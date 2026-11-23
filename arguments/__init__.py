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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.random_background = False

        # fastgs parameters
        self.grad_abs_thresh = 0.0012
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        ### rl parameters
        self.grad_thresh = 0.0001
        self.grad_abs_thresh = 0.0002

        self.rl_lr_delay_mult = 0.01
        self.rl_lr_delay_steps = 0
        self.rl_actor_lr_init = 1e-3
        self.rl_actor_lr_final = 1e-5
        self.rl_prune_estimator_lr_init = 1e-3
        self.rl_prune_estimator_lr_final = 1e-5
        self.rl_state_encoder_lr_init = 1e-3
        self.rl_state_encoder_lr_final = 1e-5
        self.rl_critic_lr_init = 1e-3
        self.rl_critic_lr_final = 1e-5
        self.delay_iter_for_reward = 50

        self.rl_use_my_value = True
        self.use_delete_action = False
        self.use_prune_estimator = True
        self.rl_inference_only = False
        self.dynamic_reset_opacity = False
        self.rl_reward_norm = True
        self.rl_use_mixed_precision = True
        self.verbose = False
        self.rl_inference_only = False

        self.rl_state_dim = 11
        self.rl_net_hidden_dim = 64
        self.rl_rollout_batch_size = 2
        self.ppo_n_epochs = 2
        self.rl_mini_batch_size = 1
        self.rl_chunk_size = 500000
        self.rl_gamma = 0.99
        self.rl_gae_lambda = 0.95
        self.rl_policy_clip = 0.2
        self.entropy_loss_init = 0
        self.entropy_loss_final = 0
        self.my_min_opacity_init = 0.005
        self.my_min_opacity_final = 0.1

        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
