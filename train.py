#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, random, time, sys
import torch
import numpy as np
import json
import uuid
from torch_scatter import scatter_mean, scatter_add
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.camera_utils import sampling_cameras
from utils.rl_utils import get_metric_score, get_gaussians_state_for_rl
from utils.general_utils import cosine_annealing


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, rl_controller_path=None):
    if saving_iterations[-1] != opt.iterations:
        saving_iterations.append(opt.iterations)
    if len(testing_iterations) != 0 and testing_iterations[-1] != opt.iterations:
        testing_iterations.append(opt.iterations)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, training_args=opt)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    if rl_controller_path:
        gaussians.rl_controller.restore(torch.load(rl_controller_path))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    
    render_start = torch.cuda.Event(enable_timing=True)
    render_end = torch.cuda.Event(enable_timing=True)
    densify_start = torch.cuda.Event(enable_timing=True)
    densify_end = torch.cuda.Event(enable_timing=True)
    reward_start = torch.cuda.Event(enable_timing=True)
    reward_end = torch.cuda.Event(enable_timing=True)
    rl_start = torch.cuda.Event(enable_timing=True)
    rl_end = torch.cuda.Event(enable_timing=True)
    
    event_cam = torch.cuda.Event(enable_timing=True)
    event_loss = torch.cuda.Event(enable_timing=True)
    event_backward = torch.cuda.Event(enable_timing=True)
    event_log_save = torch.cuda.Event(enable_timing=True)
    event_opt = torch.cuda.Event(enable_timing=True)
    
    training_statistics = {
        "camera_picking": 0.0,
        "render": 0.0,
        "loss_compute": 0.0,
        "backward": 0.0,
        "log_and_test": 0.0,
        "densify_and_prune": 0.0,
        "reward_compute": 0.0,
        "rl_learn": 0.0,
        "optimizer_step": 0.0,
        "total": 0.0
    }
    torch.cuda.reset_peak_memory_stats()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    img_num = -1

    my_viewpoint_stack = scene.getTrainCameras().copy()
    camlist = None

    for iteration in range(first_iter, opt.iterations + 1):
        densify_executed = False
        reward_executed = False
        rl_executed = False

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

            if img_num == -1:
                img_num = len(viewpoint_stack)

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        event_cam.record()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_start.record()
        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        render_end.record()
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        event_loss.record()
        
        loss.backward()
        
        event_backward.record()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians to {}".format(iteration, dataset.model_path))
                scene.save(iteration)
                torch.save(gaussians.rl_controller.capture(), os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "rl_controller.pth"))
            
            event_log_save.record()
            optim_start.record()

            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
            
            event_opt.record()

        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                camlist = sampling_cameras(my_viewpoint_stack)

                densify_executed = True
                densify_start.record()

                gaussians_state_for_rl, metric_score, visible_mask = get_gaussians_state_for_rl(camlist, gaussians, pipe, bg, opt)
                pre_metric_score = metric_score.clone()
                pre_visible_mask = visible_mask.clone()

                with torch.no_grad():
                    gaussians.densify_and_prune_rl(0.005, radii, opt,
                        gaussians_state_for_rl, iteration=iteration, tb_writer=tb_writer, visible_mask=visible_mask)

            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                _min_opacity = 0.005
                if opt.use_prune_estimator and opt.dynamic_reset_opacity:
                    _min_opacity = cosine_annealing(
                        iteration - opt.densify_from_iter,
                        opt.densify_until_iter - opt.densify_from_iter,
                        opt.my_min_opacity_init,
                        opt.my_min_opacity_final,
                    )
                gaussians.reset_opacity(_min_opacity + 0.005)
            
            if densify_executed:
                densify_end.record()

        if iteration % 3000 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations:
            densify_executed = True
            gaussians.final_prune_rl(min_opacity = opt.my_min_opacity_final)

        delayed_iteration = iteration - opt.delay_iter_for_reward
        if delayed_iteration > opt.densify_from_iter and delayed_iteration < opt.densify_until_iter \
            and delayed_iteration % opt.densification_interval == 0 and len(gaussians.rl_controller.transition["state_list"]) > 0:

            reward_executed = True
            reward_start.record()
            torch.cuda.empty_cache()

            reward = gaussians.rl_controller.transition["reward_list"][-1].squeeze(-1).cuda()
            action = gaussians.rl_controller.transition["action_list"][-1].squeeze(-1).cuda()
            prune_mask = gaussians.rl_controller.transition["prune_mask_list"][-1].squeeze(-1).cuda()
            valid_mask = gaussians.rl_controller.transition["valid_mask_list"][-1].squeeze(-1).cuda()
            parent_mapping = gaussians.parent_mapping
            
            with torch.no_grad():
                new_metric_score, new_visible_mask = get_metric_score(camlist, gaussians, pipe, bg, opt)
                new_metric_score = scatter_add(new_metric_score, parent_mapping, dim=0, dim_size=reward.shape[0])
                new_visible_mask = scatter_add(new_visible_mask.float(), parent_mapping, dim=0, dim_size=reward.shape[0])
                final_visible_mask = torch.logical_and(pre_visible_mask, new_visible_mask.bool())
                valid_mask[(action != 3) & ~final_visible_mask] = False  # 忽略非删除点又没有同时在两次渲染中出现的点
                gaussians.rl_controller.transition["valid_mask_list"][-1] = valid_mask.cpu()

                points_improvement = new_metric_score[valid_mask] - pre_metric_score[valid_mask]
                reward[valid_mask] += points_improvement

                if getattr(opt, "rl_reward_norm", True) and pre_visible_mask.any():
                    reward_valid = reward[valid_mask]
                    if reward_valid.numel() > 0:
                        reward_mean = reward_valid.mean()
                        reward_std = reward_valid.std()
                        reward_valid = (reward_valid - reward_mean) / (reward_std + 1e-6)
                        reward[valid_mask] = reward_valid

                        if tb_writer:
                            tb_writer.add_scalar("reward/norm_mean", reward_mean.item(), iteration)
                            tb_writer.add_scalar("reward/norm_std", reward_std.item(), iteration)

                if opt.rl_use_my_value:
                    value = torch.zeros_like(reward)
                    prune_keep_mask = valid_mask & prune_mask & (action == 0)
                    non_prune_keep_mask = valid_mask & (~prune_mask) & (action == 0)
                    if prune_keep_mask.any():
                        value[valid_mask & prune_mask] = reward[prune_keep_mask].mean()
                    if non_prune_keep_mask.any():
                        value[valid_mask & (~prune_mask)] = reward[non_prune_keep_mask].mean()
                    gaussians.rl_controller.transition["value_list"].append(value.unsqueeze(-1).cpu())

                # # 记录reward统计
                if tb_writer:
                    tb_writer.add_scalar("reward/reward_mean", reward.mean().item(), iteration)
                    tb_writer.add_scalar("reward/reward_std", reward.std().item(), iteration)

                gaussians.rl_controller.transition["reward_list"][-1] = reward.unsqueeze(-1).cpu()
            
            reward_end.record()

            if len(gaussians.rl_controller.transition["state_list"]) == opt.rl_rollout_batch_size:
                rl_executed = True
                rl_start.record()
                lr = gaussians.rl_controller.update_learning_rate(delayed_iteration - opt.densify_from_iter)
                if not opt.rl_inference_only:
                    gaussians.rl_controller.learn(iteration=iteration, tb_writer=tb_writer)
                gaussians.parent_mapping = None
                gaussians.rl_controller.transition.clear()
                torch.cuda.empty_cache()
                rl_end.record()

        # record time
        optim_end.record()
        torch.cuda.synchronize()
        optim_time = optim_start.elapsed_time(optim_end)
            
        training_statistics["camera_picking"] += iter_start.elapsed_time(event_cam)
        training_statistics["render"] += event_cam.elapsed_time(render_end)
        training_statistics["loss_compute"] += render_end.elapsed_time(event_loss)
        training_statistics["backward"] += event_loss.elapsed_time(event_backward)
        training_statistics["log_and_test"] += event_backward.elapsed_time(event_log_save)
        training_statistics["optimizer_step"] += event_log_save.elapsed_time(event_opt)
        
        if densify_executed:
            training_statistics["densify_and_prune"] += densify_start.elapsed_time(densify_end)
        if reward_executed:
            training_statistics["reward_compute"] += reward_start.elapsed_time(reward_end)
        if rl_executed:
            training_statistics["rl_learn"] += rl_start.elapsed_time(rl_end)
            
        training_statistics["total"] += (iter_time + optim_time)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print("\n--- Training Time Statistics ---")
    stats_to_save = {}
    for k, v in training_statistics.items():
        time_s = v / 1000
        percentage = v / training_statistics['total'] * 100
        stats_to_save[k] = {"time_s": time_s, "percentage": percentage}
    
    print(f"Training Time: {stats_to_save['total']['time_s']:.2f} s")
    stats_to_save["GS_number"] = gaussians._xyz.shape[0]
    stats_to_save["peak_gpu_memory_mb"] = peak_memory
    
    with open(os.path.join(dataset.model_path, "training_statistics.json"), "w") as f:
        json.dump(stats_to_save, f, indent=4)

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 10):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scaling_histogram", scene.gaussians.get_scaling, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--rl_controller_path", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets,
        args.rl_controller_path
    )

    # All done
    print("\nTraining complete.")
