import torch
from fused_ssim import fused_ssim as fast_ssim
from utils.image_utils import psnr

from gaussian_renderer import render_fastgs
from utils.loss_utils import l1_loss


def sign_log1p(x):
    return x.sign() * torch.log1p(x.abs())


def normalize_features(features):
    """对特征进行标准化归一化"""
    return (features - features.mean(0, keepdim=True)) / (features.std(0, keepdim=True) + 1e-6)
    

def get_metric_score(camlist, gaussians, pipe, bg, opt):
    """
    计算每个Gaussian点对渲染误差的贡献度
    """
    num_points = gaussians.get_xyz.shape[0]
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        gt_image = my_viewpoint_cam.original_image.cuda()
        
        with torch.no_grad():
            render_pkg2 = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=None, gt_image=gt_image)
            accum_metric_per_gs = render_pkg2["accum_metric_per_gs"]
            accum_gs_weight = render_pkg2["accum_gs_weight"]

            metric_score += accum_metric_per_gs
            gs_weights += accum_gs_weight

    visible_mask = gs_weights > 0
    metric_score /= len(camlist)
    metric_score = sign_log1p(metric_score).clamp(min=-6.0, max=6.0)
    return metric_score, visible_mask


def get_gaussians_state_for_rl(camlist, gaussians, pipe, bg, opt):
    """
    构建RL的状态特征
    """
    num_points = len(gaussians.get_xyz)
    
    xyz_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    scale_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    opacity_grads = torch.zeros(num_points, 1, device="cuda", dtype=torch.float32)
    feature_dc_grads = torch.zeros(num_points, 3, device="cuda", dtype=torch.float32)
    metric_score = torch.zeros(num_points, device="cuda", dtype=torch.float32)
    gs_weights = torch.zeros(num_points, device="cuda", dtype=torch.float32)

    n_views = len(camlist)
    
    for view in range(n_views):
        my_viewpoint_cam = camlist[view]
        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult)

        render_image = render_pkg["render"]
        gt_image = my_viewpoint_cam.original_image.cuda()

        render_image.clamp_(min=0.0, max=1.0)
        gt_image.clamp_(min=0.0, max=1.0)

        Ll1 = l1_loss(render_image, gt_image)
        ssim_value = fast_ssim(render_image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        xyz_grads += gaussians._xyz.grad
        scale_grads += gaussians._scaling.grad
        opacity_grads += gaussians._opacity.grad
        feature_dc_grads += gaussians._features_dc.grad.squeeze(1)

        gaussians.optimizer.zero_grad(set_to_none=True)
        if getattr(gaussians, "shoptimizer", None) is not None:
            gaussians.shoptimizer.zero_grad(set_to_none=True)
        gaussians.clear_grad()

        with torch.no_grad():
            render_pkg2 = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, opt.mult, get_flag=True, metric_map=None, gt_image=gt_image)
            accum_metric_per_gs = render_pkg2["accum_metric_per_gs"]
            accum_gs_weight = render_pkg2["accum_gs_weight"]

            metric_score += accum_metric_per_gs
            gs_weights += accum_gs_weight

        del render_pkg, render_image, gt_image, loss, render_pkg2, my_viewpoint_cam

    visible_mask = gs_weights > 0

    metric_score /= len(camlist)
    metric_score = sign_log1p(metric_score)
    metric_score_feature = metric_score.clone().detach().unsqueeze(-1)
    metric_score_feature = normalize_features(metric_score_feature)
    
    grad_features = torch.cat([
        xyz_grads.clone().detach(),
        scale_grads.clone().detach(),
        opacity_grads.clone().detach(),
        feature_dc_grads.clone().detach(),
    ], dim=-1)
    grad_features = normalize_features(grad_features)

    states = torch.cat([
        grad_features,
        metric_score_feature,
    ], dim=-1)

    return states, metric_score, visible_mask
