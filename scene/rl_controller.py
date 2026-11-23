import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_scatter import scatter_mean
from torch.utils.checkpoint import checkpoint
from utils.general_utils import get_expon_lr_func, cosine_annealing
from collections import defaultdict


class MLPStateEncoder(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64):
        super(MLPStateEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        expand_dim = hidden_dim * 4
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(hidden_dim),
                'gate': nn.Linear(hidden_dim, expand_dim),
                'up': nn.Linear(hidden_dim, expand_dim),
                'down': nn.Linear(expand_dim, hidden_dim),
            }) for _ in range(3)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, state_features, xyz_coords=None):
        x = self.input_proj(state_features)
        
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            gate = F.silu(layer['gate'](x))
            up = layer['up'](x)
            x = layer['down'](gate * up)
            x = x + residual
        
        return self.output_norm(x)


class PPOActor(nn.Module):
    """
    PPO策略网络 - 输出动作概率分布
    """
    def __init__(self, hidden_dim=64, action_dim=4):
        super(PPOActor, self).__init__()
        self.action_dim = action_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 输出层零初始化，使初始策略接近均匀分布
        self._init_output_layer()
    
    def _init_output_layer(self):
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
        # 抑制初始的delete概率，使初始策略更倾向于保留点，避免崩溃
        if self.mlp[-1].bias.shape[0] == 4:
            self.mlp[-1].bias[-1].data -= 2

    def forward(self, x, temperature=1.0):
        logits = self.mlp(x) / temperature
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def get_action(self, state):
        """
        获取每个Gaussian点的动作
        
        Args:
            state: (n, hidden_dim) 每个Gaussian点的编码特征
            
        Returns:
            action: (n, 1) 每个点的动作索引
        """
        probs = self.forward(state)  # (n, action_dim)
        action = torch.argmax(probs, dim=-1, keepdim=True)  # (n, 1)
        return action


class PPOPruneEstimator(nn.Module):
    """
    Prune估计器 - 专门负责判断是否删除点
    """
    def __init__(self, hidden_dim=64):
        super(PPOPruneEstimator, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self._init_output_layer()
    
    def _init_output_layer(self):
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def forward(self, x):
        return torch.sigmoid(self.mlp(x))

    def get_action(self, state):
        return (self.forward(state) > 0.5).int()


class PPOCritic(nn.Module):
    """
    PPO价值网络 - 评估状态价值
    """
    def __init__(self, hidden_dim=64):
        super(PPOCritic, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class GaussianDensificationController:
    """使用PPO算法控制Gaussian点的densification过程"""
    def __init__(self,
                 training_args,
                 device="cuda"):
        self.device = device
        self.gamma = training_args.rl_gamma
        self.policy_clip = training_args.rl_policy_clip
        self.gae_lambda = training_args.rl_gae_lambda
        self.rl_rollout_batch_size = training_args.rl_rollout_batch_size
        self.lr_scheduler = None
        self.training_args = training_args

        self.entropy_coef_init = getattr(training_args, 'entropy_loss_init', 0.1)
        self.entropy_coef_final = getattr(training_args, 'entropy_loss_final', 0.01)
        self.use_my_value = getattr(training_args, 'rl_use_my_value', False)

        state_dim = training_args.rl_state_dim
        hidden_dim = training_args.rl_net_hidden_dim

        self.actor = PPOActor(hidden_dim, action_dim=4 if training_args.use_delete_action else 3).to(device)

        if training_args.use_prune_estimator:
            self.prune_estimator = PPOPruneEstimator(hidden_dim).to(device)
            if not self.use_my_value:
                self.prune_critic = PPOCritic(hidden_dim).to(device)

        if not self.use_my_value:
            self.critic = PPOCritic(hidden_dim).to(device)
        self.state_encoder = MLPStateEncoder(state_dim, hidden_dim).to(device)

        if getattr(training_args, "rl_inference_only", False):
            self.actor.eval()
            for param in self.actor.parameters():
                param.requires_grad = False
            if training_args.use_prune_estimator:
                self.prune_estimator.eval()
                for param in self.prune_estimator.parameters():
                    param.requires_grad = False
                if not self.use_my_value:
                    self.prune_critic.eval()
                    for param in self.prune_critic.parameters():
                        param.requires_grad = False
            if not self.use_my_value:
                self.critic.eval()
                for param in self.critic.parameters():
                    param.requires_grad = False
            self.state_encoder.eval()
            for param in self.state_encoder.parameters():
                param.requires_grad = False

        # 打印网络参数大小，换算成MB单位
        if self.training_args.verbose:
            print(f"Actor parameters size: {sum(p.numel() for p in self.actor.parameters()) / 1e6:.4f}MB")
            if not self.use_my_value:
                print(f"Critic parameters size: {sum(p.numel() for p in self.critic.parameters()) / 1e6:.4f}MB")
            print(f"State encoder parameters size: {sum(p.numel() for p in self.state_encoder.parameters()) / 1e6:.4f}MB")
            if training_args.use_prune_estimator:
                print(f"prune_estimator parameters size: {sum(p.numel() for p in self.prune_estimator.parameters()) / 1e6:.4f}MB")
                if not self.use_my_value:
                    print(f"prune_critic parameters size: {sum(p.numel() for p in self.prune_critic.parameters()) / 1e6:.4f}MB")
        
        # 优化器
        self.actor_optimizer = optim.AdamW(list(self.actor.parameters()), weight_decay=1e-4)
        if not self.use_my_value:
            self.critic_optimizer = optim.AdamW(list(self.critic.parameters()), weight_decay=1e-4)
        self.state_encoder_optimizer = optim.AdamW(list(self.state_encoder.parameters()), weight_decay=1e-4)

        if training_args.use_prune_estimator:
            self.prune_estimator_optimizer = optim.AdamW(list(self.prune_estimator.parameters()), weight_decay=1e-4)
            if not self.use_my_value:
                self.prune_critic_optimizer = optim.AdamW(list(self.prune_critic.parameters()), weight_decay=1e-4)

        self.transition = defaultdict(list)
        
    def store_transition(self, state, action, reward, parent_mapping, **kwargs):
        self.transition["state_list"].append(state.cpu())
        self.transition["action_list"].append(action.cpu())
        self.transition["reward_list"].append(reward.cpu())
        self.transition["parent_mapping_list"].append(parent_mapping.cpu() if parent_mapping is not None else None)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                self.transition[key + "_list"].append(value.cpu())
            else:
                self.transition[key + "_list"].append(value)

    def _batch_encode(self, state, chunk_size=None, use_checkpoint=False):
        if chunk_size is None:
            chunk_size = self.training_args.rl_chunk_size
        if chunk_size is None:
            chunk_size = state.shape[0]

        encoded_list = []
        for i in range(0, state.shape[0], chunk_size):
            chunk = state[i:i+chunk_size]
            if use_checkpoint:
                chunk = chunk.detach().requires_grad_(True)
                def run_encoder(x):
                    return self.state_encoder(x)
                encoded_chunk = checkpoint(run_encoder, chunk, use_reentrant=False)
            else:
                encoded_chunk = self.state_encoder(chunk)
            encoded_list.append(encoded_chunk)
        return torch.cat(encoded_list, dim=0)
        
    def learn(self, iteration=None, tb_writer=None):
        """PPO学习过程"""
        state_list = self.transition["state_list"]
        action_list = self.transition["action_list"]
        reward_list = self.transition["reward_list"]
        parent_mapping_list = self.transition["parent_mapping_list"]
        valid_mask_list = self.transition.get("valid_mask_list", [torch.ones(x.shape[0], dtype=bool) for x in state_list])
        prune_mask_list = self.transition.get("prune_mask_list", [torch.zeros(x.shape[0], dtype=bool) for x in state_list])

        if not self.use_my_value:
            value_list = []
        old_log_prob_list = []

        with torch.no_grad():
            for state, action, valid_mask, prune_mask in zip(state_list, action_list, valid_mask_list, prune_mask_list):
                state = state.cuda()
                action = action.cuda()
                valid_mask = valid_mask.cuda()
                prune_mask = prune_mask.cuda()

                n_points = state.shape[0]

                state = state[valid_mask]
                action = action[valid_mask]
                prune_mask = prune_mask[valid_mask]

                encoded = self._batch_encode(state, chunk_size=self.training_args.rl_chunk_size, use_checkpoint=False)

                if not self.use_my_value:
                    value = torch.zeros(n_points, 1, device="cuda", dtype=encoded.dtype)
                    if self.training_args.use_prune_estimator:
                        if (~prune_mask).any():
                            value_non_prune = self.critic(encoded[~prune_mask])
                        if prune_mask.any():
                            value_prune = self.prune_critic(encoded[prune_mask])

                        valid_indices = torch.where(valid_mask)[0]
                        if (~prune_mask).any():
                            value[valid_indices[~prune_mask]] = value_non_prune
                        if prune_mask.any():
                            value[valid_indices[prune_mask]] = value_prune
                    else:
                        value[valid_mask] = self.critic(encoded)
                    value_list.append(value)

                if self.training_args.use_prune_estimator:
                    probs = torch.zeros(encoded.shape[0], 1, device="cuda", dtype=encoded.dtype)

                    if prune_mask.any():
                        prune_encoded_state = encoded[prune_mask]
                        prune_probs = self.prune_estimator(prune_encoded_state)
                        prune_probs = torch.where(action[prune_mask] == 3, prune_probs, 1 - prune_probs)
                        probs[prune_mask] = prune_probs.to(probs.dtype)
                    
                    if (~prune_mask).any():
                        non_prune_encoded_state = encoded[~prune_mask]
                        non_prune_probs = self.actor(non_prune_encoded_state)
                        non_prune_probs = non_prune_probs.gather(-1, action[~prune_mask])
                        probs[~prune_mask] = non_prune_probs.to(probs.dtype)
                else:
                    probs = self.actor(encoded)
                    probs = probs.gather(-1, action)

                log_prob = torch.log(probs)
                old_log_prob_list.append(log_prob.detach())

            if self.use_my_value:
                value_list = self.transition["value_list"]

            lastgaelam = 0
            advantage_list_reversed = []
            t_length = len(reward_list)
            
            for t in reversed(range(t_length)):
                reward_t = reward_list[t].cuda()
                # 注意parent mapping是当前点对应的父节点
                value_t = value_list[t].cuda()
                if t < t_length - 1:
                    value_t_next = value_list[t + 1].cuda()
                    # 把 s_{t+1} 的点聚合回 s_t 的父点
                    parent_mapping_t_next = parent_mapping_list[t + 1]
                    if parent_mapping_t_next is not None:
                        parent_mapping_t_next = parent_mapping_t_next.cuda()

                    next_value_aggregated = scatter_mean(
                        value_t_next,
                        parent_mapping_t_next.unsqueeze(-1),
                        dim=0,
                        dim_size=value_t.shape[0],
                    )
                    
                    # TD误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                    delta = reward_t + self.gamma * next_value_aggregated - value_t

                    # GAE: A_t = δ_t + (γλ) * A_{t+1}
                    lastgaelam_aggregated = scatter_mean(
                        lastgaelam,
                        parent_mapping_t_next.unsqueeze(-1),
                        dim=0,
                        dim_size=value_t.shape[0],
                    )
                    lastgaelam = delta + self.gamma * self.gae_lambda * lastgaelam_aggregated
                else:
                    lastgaelam = reward_t - value_t
                advantage_list_reversed.append(lastgaelam.cpu())
            advantage_list = advantage_list_reversed[::-1]  # 反转得到正序

        filtered_state_list = []
        filtered_action_list = []
        filtered_advantage_list = []
        filtered_value_list = []
        filtered_prune_mask_list = []

        for state, action, adv, val, p_mask, v_mask in zip(state_list, action_list, advantage_list, value_list, prune_mask_list, valid_mask_list):
            v_mask_cpu = v_mask.cpu()
            filtered_state_list.append(state.cpu()[v_mask_cpu])
            filtered_action_list.append(action.cpu()[v_mask_cpu])
            filtered_advantage_list.append(adv.cpu()[v_mask_cpu].detach())
            filtered_value_list.append(val.cpu()[v_mask_cpu])
            filtered_prune_mask_list.append(p_mask.cpu()[v_mask_cpu])

        state_list = filtered_state_list
        action_list = filtered_action_list
        advantage_list = filtered_advantage_list
        value_list = filtered_value_list
        prune_mask_list = filtered_prune_mask_list
        old_log_prob_list = [x.cpu() for x in old_log_prob_list]

        if len(action_list) == 0 or all(x.numel() == 0 for x in action_list):
            assert False

        all_action = torch.cat(action_list).squeeze(-1)
        all_advantage = torch.cat(advantage_list).squeeze(-1)

        if tb_writer:
            tb_writer.add_scalar("rl/clone_advantage_mean", all_advantage[(all_action == 1)].mean().item(), iteration)
            tb_writer.add_scalar("rl/split_advantage_mean", all_advantage[(all_action == 2)].mean().item(), iteration)
            tb_writer.add_scalar("rl/delete_advantage_mean", all_advantage[(all_action == 3)].mean().item(), iteration)
        
        n_epochs = self.training_args.ppo_n_epochs
        n_rollout = len(state_list)
        pg_loss_avg = 0.
        vf_loss_avg = 0.
        entropy_loss_avg = 0.
        ratio_avg = 0.

        for _i in range(n_epochs):
            self.actor_optimizer.zero_grad(set_to_none=True)
            if not self.use_my_value:
                self.critic_optimizer.zero_grad(set_to_none=True)
            self.state_encoder_optimizer.zero_grad(set_to_none=True)
            if self.training_args.use_prune_estimator:
                self.prune_estimator_optimizer.zero_grad(set_to_none=True)
                if not self.use_my_value:
                    self.prune_critic_optimizer.zero_grad(set_to_none=True)
            
            # 分批次累积梯度，进一步降低显存
            mini_batch_size = self.training_args.rl_mini_batch_size
            chunk_size = self.training_args.rl_chunk_size

            for start_idx in range(0, n_rollout, mini_batch_size):
                end_idx = min(start_idx + mini_batch_size, n_rollout)
                
                for state_full, action_full, advantage_full, old_value_full, old_log_prob_full, prune_mask_full in zip(
                    state_list[start_idx:end_idx], action_list[start_idx:end_idx], 
                    advantage_list[start_idx:end_idx], value_list[start_idx:end_idx], 
                    old_log_prob_list[start_idx:end_idx], prune_mask_list[start_idx:end_idx]
                ):
                    n_points_in_step = state_full.shape[0]

                    for chunk_start in range(0, n_points_in_step, chunk_size if chunk_size is not None else n_points_in_step):
                        chunk_end = min(chunk_start + chunk_size if chunk_size is not None else n_points_in_step, n_points_in_step)

                        state = state_full[chunk_start:chunk_end].cuda()
                        action = action_full[chunk_start:chunk_end].cuda()
                        advantage = advantage_full[chunk_start:chunk_end].cuda()
                        old_value = old_value_full[chunk_start:chunk_end].cuda()
                        old_log_prob = old_log_prob_full[chunk_start:chunk_end].cuda()
                        prune_mask = prune_mask_full[chunk_start:chunk_end].cuda()

                        chunk_weight = (chunk_end - chunk_start) / n_points_in_step
                        
                        with torch.amp.autocast('cuda', enabled=self.training_args.rl_use_mixed_precision):
                            def run_encoder(x):
                                return self.state_encoder(x)

                            state_enc = checkpoint(run_encoder, state.detach().requires_grad_(True), use_reentrant=False)
                            actual_value = (advantage + old_value)

                            if self.training_args.use_prune_estimator:
                                dtype = state_enc.dtype
                                probs = torch.zeros(state_enc.shape[0], 1, device="cuda", dtype=dtype)
                                if not self.use_my_value:
                                    value = torch.zeros(state_enc.shape[0], 1, device="cuda", dtype=dtype)
                                
                                if prune_mask.any():
                                    prune_encoded_state = state_enc[prune_mask]
                                    prune_probs = self.prune_estimator(prune_encoded_state)
                                    prune_probs = torch.where(action[prune_mask] == 3, prune_probs, 1 - prune_probs)
                                    probs[prune_mask] = prune_probs.to(probs.dtype).view(-1, 1)

                                    if not self.use_my_value:
                                        prune_value = self.prune_critic(prune_encoded_state)
                                        value[prune_mask] = prune_value.to(value.dtype).view(-1, 1)
                                
                                if (~prune_mask).any():
                                    non_prune_encoded_state = state_enc[~prune_mask]
                                    non_prune_probs = self.actor(non_prune_encoded_state)
                                    non_prune_probs = non_prune_probs.gather(-1, action[~prune_mask])
                                    probs[~prune_mask] = non_prune_probs.to(probs.dtype).view(-1, 1)

                                    if not self.use_my_value:
                                        non_prune_value = self.critic(non_prune_encoded_state)
                                        value[~prune_mask] = non_prune_value.to(value.dtype).view(-1, 1)
                            else:
                                if not self.use_my_value:
                                    value = self.critic(state_enc)
                                probs = self.actor(state_enc)
                                probs = probs.gather(-1, action)

                            log_prob = torch.log(probs + 1e-8)
                            ratio = torch.exp(log_prob - old_log_prob.to(log_prob.dtype))

                            pg_loss1 = -advantage.to(ratio.dtype) * ratio
                            pg_loss2 = -advantage.to(ratio.dtype) * torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip)
                            pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                            if not self.use_my_value:
                                vf_loss = 0.5 * torch.mean((value - actual_value.to(value.dtype)) ** 2)
                            else:
                                vf_loss = 0.

                            # if torch.isnan(pg_loss) or torch.isinf(pg_loss):
                            #     print(f"警告: 检测到无效的loss值 (pg_loss={pg_loss.item()}, vf_loss={vf_loss.item()})，跳过此batch")
                            #     print("log_prob:", log_prob.mean().item(), "old_log_prob:", old_log_prob.mean().item())
                            #     raise ValueError("Invalid loss values")

                            entropy = 0.
                            if self.training_args.use_prune_estimator:
                                # 处理 prune 点的熵
                                if prune_mask.any():
                                    prune_probs = self.prune_estimator(state_enc[prune_mask])
                                    prune_probs = torch.cat([prune_probs, 1 - prune_probs], dim=-1)
                                    entropy += -torch.sum(prune_probs * torch.log(prune_probs + 1e-8), dim=-1).mean()
                                
                                if (~prune_mask).any():
                                    non_prune_probs = self.actor(state_enc[~prune_mask])
                                    entropy += -torch.sum(non_prune_probs * torch.log(non_prune_probs + 1e-8), dim=-1).mean()
                            else:
                                probs = self.actor(state_enc)
                                entropy += -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                            
                            entropy_coef = cosine_annealing(
                                iteration - self.training_args.densify_from_iter,
                                self.training_args.densify_until_iter - self.training_args.densify_from_iter,
                                initial_temp=self.entropy_coef_init, 
                                final_temp=self.entropy_coef_final
                            )
                            entropy_loss = -entropy_coef * entropy  # 最大化熵

                            pg_loss_avg += pg_loss.item() * chunk_weight
                            if not self.use_my_value:
                                vf_loss_avg += vf_loss.item() * chunk_weight
                            entropy_loss_avg += entropy_loss.item() * chunk_weight
                            ratio_avg += ratio.mean().item() * chunk_weight

                            chunk_loss = (pg_loss + vf_loss + entropy_loss) * chunk_weight / n_rollout

                        chunk_loss.backward()
                        del state, action, advantage, old_value, old_log_prob, prune_mask, probs, log_prob, ratio, state_enc, chunk_loss

            self.actor_optimizer.step()
            if not self.use_my_value:
                self.critic_optimizer.step()
            self.state_encoder_optimizer.step()
            if self.training_args.use_prune_estimator:
                self.prune_estimator_optimizer.step()
                if not self.use_my_value:
                    self.prune_critic_optimizer.step()

        pg_loss_avg /= n_epochs * n_rollout
        vf_loss_avg /= n_epochs * n_rollout
        entropy_loss_avg /= n_epochs * n_rollout
        ratio_avg /= n_epochs * n_rollout
        
        if tb_writer:
            all_values = torch.cat(value_list)
            tb_writer.add_scalar("rl/pg_loss_avg", pg_loss_avg, iteration)
            tb_writer.add_scalar("rl/vf_loss_avg", vf_loss_avg, iteration)
            tb_writer.add_scalar("rl/ratio_avg", ratio_avg, iteration)
            tb_writer.add_scalar("rl/value_mean", all_values.mean().item(), iteration)
            tb_writer.add_scalar("rl/value_std", all_values.std().item(), iteration)

    def training_setup(self, training_args):
        self.actor_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_actor_lr_init,
            lr_final=training_args.rl_actor_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        self.critic_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_critic_lr_init,
            lr_final=training_args.rl_critic_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )

        self.state_encoder_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_state_encoder_lr_init,
            lr_final=training_args.rl_state_encoder_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        
        self.prune_estimator_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_prune_estimator_lr_init,
            lr_final=training_args.rl_prune_estimator_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )
        
        self.prune_critic_lr_scheduler = get_expon_lr_func(
            lr_init=training_args.rl_critic_lr_init,
            lr_final=training_args.rl_critic_lr_final,
            lr_delay_steps=training_args.rl_lr_delay_steps,
            lr_delay_mult=training_args.rl_lr_delay_mult,
            max_steps=training_args.densify_until_iter - training_args.densify_from_iter,
        )

    def update_learning_rate(self, iteration):
        lr = self.actor_lr_scheduler(iteration)
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

        if not self.use_my_value:
            lr = self.critic_lr_scheduler(iteration)
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr

        lr = self.state_encoder_lr_scheduler(iteration)
        for param_group in self.state_encoder_optimizer.param_groups:
            param_group['lr'] = lr

        if self.training_args.use_prune_estimator:
            lr = self.prune_estimator_lr_scheduler(iteration)
            for param_group in self.prune_estimator_optimizer.param_groups:
                param_group['lr'] = lr
            if not self.use_my_value:
                lr = self.prune_critic_lr_scheduler(iteration)
                for param_group in self.prune_critic_optimizer.param_groups:
                    param_group['lr'] = lr

        return lr


    def capture(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": None if self.use_my_value else self.critic.state_dict(),
            "state_encoder": self.state_encoder.state_dict(),
            "prune_estimator": self.prune_estimator.state_dict() if getattr(self, "prune_estimator", None) is not None else None,
            "prune_critic": self.prune_critic.state_dict() if getattr(self, "prune_critic", None) is not None else None,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": None if self.use_my_value else self.critic_optimizer.state_dict(),
            "state_encoder_optimizer": self.state_encoder_optimizer.state_dict(),
            "prune_estimator_optimizer": self.prune_estimator_optimizer.state_dict() if getattr(self, "prune_estimator_optimizer", None) is not None else None,
            "prune_critic_optimizer": self.prune_critic_optimizer.state_dict() if getattr(self, "prune_critic_optimizer", None) is not None else None,
        }

    def restore(self, rl_controller_state):
        if isinstance(rl_controller_state, tuple):
            self.actor.load_state_dict(rl_controller_state[0])
            if not self.use_my_value and len(rl_controller_state) > 1 and rl_controller_state[1] is not None:
                self.critic.load_state_dict(rl_controller_state[1])
            if len(rl_controller_state) > 2:
                self.state_encoder.load_state_dict(rl_controller_state[2])
            return

        self.actor.load_state_dict(rl_controller_state["actor"])
        if not self.use_my_value and rl_controller_state.get("critic") is not None:
            self.critic.load_state_dict(rl_controller_state["critic"])
        self.state_encoder.load_state_dict(rl_controller_state["state_encoder"])
        if getattr(self, "prune_estimator", None) is not None and rl_controller_state.get("prune_estimator") is not None:
            self.prune_estimator.load_state_dict(rl_controller_state["prune_estimator"])
        if getattr(self, "prune_critic", None) is not None and rl_controller_state.get("prune_critic") is not None:
            self.prune_critic.load_state_dict(rl_controller_state["prune_critic"])
        if rl_controller_state.get("actor_optimizer") is not None:
            self.actor_optimizer.load_state_dict(rl_controller_state["actor_optimizer"])
        if not self.use_my_value and rl_controller_state.get("critic_optimizer") is not None:
            self.critic_optimizer.load_state_dict(rl_controller_state["critic_optimizer"])
        if rl_controller_state.get("state_encoder_optimizer") is not None:
            self.state_encoder_optimizer.load_state_dict(rl_controller_state["state_encoder_optimizer"])
        if getattr(self, "prune_estimator_optimizer", None) is not None and rl_controller_state.get("prune_estimator_optimizer") is not None:
            self.prune_estimator_optimizer.load_state_dict(rl_controller_state["prune_estimator_optimizer"])
        if getattr(self, "prune_critic_optimizer", None) is not None and rl_controller_state.get("prune_critic_optimizer") is not None:
            self.prune_critic_optimizer.load_state_dict(rl_controller_state["prune_critic_optimizer"])
