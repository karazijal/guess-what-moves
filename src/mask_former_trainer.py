# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import json
import logging
import os
import sys
from typing import Any, Dict, List, Set

import detectron2.utils.comm as comm
import torch
import wandb
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

import utils
# MaskFormer
from config import add_gwm_config

logger = logging.getLogger('gwm')


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "RMSProp":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.RMSprop)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """

    wandb_inited = False
    if 'CONFIG_FILE' in args.opts and not args.wandb_sweep_mode:
        logger.warning(
            f"Found CONFIG_FILE key in OPT args and using {args.opts[args.opts.index('CONFIG_FILE') + 1]} instead of {args.config_file}")
        args.config_file = args.opts[args.opts.index('CONFIG_FILE') + 1]
    else:
        cfg = get_cfg()
        add_gwm_config(cfg)
        wandb_basedir = cfg.WANDB.BASEDIR
        cfg_dict = CfgNode.load_yaml_with_base(args.config_file, allow_unsafe=True)
        if 'WANDB' in cfg_dict and 'BASEDIR' in cfg_dict['WANDB']:
            wandb_basedir = cfg_dict['WANDB']['BASEDIR']
        if 'CONFIG_FILE' in cfg_dict and cfg_dict['CONFIG_FILE'] is not None:
            logger.warning(
                f"Found CONFIG_FILE key in the config.yaml file and using {cfg_dict['CONFIG_FILE']} instead of {args.config_file}")
            args.config_file = cfg_dict['CONFIG_FILE']

        if args.wandb_sweep_mode:
            if PathManager.isfile('wandb.yaml'):
                wandb_cfg = CfgNode.load_yaml_with_base('wandb.yaml', allow_unsafe=False)
                wandb.init(project=wandb_cfg['PROJECT'], entity=wandb_cfg['USER'], dir=wandb_basedir)
                wandb_inited = True
                if wandb.run.sweep_id:  # sweep active
                    sweep_dict = dict(wandb.config)
                    if 'CONFIG_FILE' in sweep_dict:
                        args.config_file = sweep_dict['CONFIG_FILE']
                        logger.warning(f"Loading CONFIG_FILE as set in sweep config: {args.config_file}")
                    elif 'CONFIG_FILE' in args.opts:
                        args.config_file = args.opts[args.opts.index('CONFIG_FILE') + 1]
                        logger.warning(f"Loading CONFIG_FILE as set in the optional arguments: {args.config_file}")

    if 'GWM.MODEL' in args.opts and not args.wandb_sweep_mode:
        logger.warning(
            "It is advised to not set GWM.MODEL in OPT args and instead set it in the config.yaml file")
        model = args.opts[args.opts.index('GWM.MODEL') + 1]
    else:
        cfg = get_cfg()
        add_gwm_config(cfg)
        model = cfg.GWM.MODEL
        wandb_basedir = cfg.WANDB.BASEDIR
        cfg_dict = CfgNode.load_yaml_with_base(args.config_file, allow_unsafe=True)
        if 'GWM' in cfg_dict and 'MODEL' in cfg_dict['GWM']:
            model = cfg_dict['GWM']['MODEL']
        if 'WANDB' in cfg_dict and 'BASEDIR' in cfg_dict['WANDB']:
            wandb_basedir = cfg_dict['WANDB']['BASEDIR']

        if args.wandb_sweep_mode:
            if PathManager.isfile('wandb.yaml'):
                if not wandb_inited:
                    wandb_cfg = CfgNode.load_yaml_with_base('wandb.yaml', allow_unsafe=False)
                    wandb.init(project=wandb_cfg['PROJECT'], entity=wandb_cfg['USER'], dir=wandb_basedir)
                    wandb_inited = True

                if args.wandb_sweep_mode:
                    sweep_dict = dict(wandb.config)
                    if 'GWM.MODEL' in sweep_dict:
                        logger.warning(
                            "It is advised to not set GWM.MODEL in sweep config and instead set it in the config.yaml file")
                        model = sweep_dict['GWM.MODEL']
                    elif 'GWM.MODEL' in args.opts:
                        logger.warning(
                            "It is advised to not set GWM.MODEL in optional arguments and instead set it in the config.yaml file")
                        model = args.opts[args.opts.index('GWM.MODEL') + 1]

    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    if model == "MASKFORMER":
        from mask_former import add_mask_former_config
        add_mask_former_config(cfg)
    else:
        logger.error(f'Unknown Model: {model}. Exiting..')
        sys.exit(0)

    add_gwm_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.WANDB.ENABLE = (cfg.WANDB.ENABLE or args.wandb_sweep_mode) and not args.eval_only

    datestring = utils.log.get_datestring_for_the_run()
    if cfg.WANDB.ENABLE:
        if PathManager.isfile('wandb.yaml'):
            if not wandb_inited:
                wandb_cfg = CfgNode.load_yaml_with_base('wandb.yaml', allow_unsafe=False)
                wandb.init(project=wandb_cfg['PROJECT'], entity=wandb_cfg['USER'], dir=cfg.WANDB.BASEDIR)

            if args.wandb_sweep_mode:  # sweep active
                sweep_list = [(k, v) for k, v in dict(wandb.config).items()]
                sweep_list = [item for items in sweep_list for item in items]
                cfg.merge_from_list(sweep_list)

            if cfg.LOG_ID is not None:
                api = wandb.Api()
                run = api.run(path=f"{wandb_cfg['USER']}/{wandb_cfg['PROJECT']}/{wandb.run.id}")
                run.name = f'{cfg.LOG_ID}/{datestring}-{wandb.run.id}'
                run.save()

        else:
            logger.error("W&B config file 'src/wandb.yaml' does not exist!")
            cfg.WANDB.ENABLE = False

    if args.resume_path:
        cfg.OUTPUT_DIR = "/".join(args.resume_path.split('/')[:-2])  # LOG_ID/datestring/checkpoints/checkpoints.pth

        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval', datestring)

    else:
        if cfg.LOG_ID and not cfg.SLURM:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_BASEDIR, cfg.LOG_ID)
        else:
            cfg.OUTPUT_DIR = cfg.OUTPUT_BASEDIR

        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval', datestring)
        else:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datestring)
            os.makedirs(f'{cfg.OUTPUT_DIR}/checkpoints', exist_ok=True)


    if cfg.WANDB.ENABLE:
        wandb.config.update(cfg, allow_val_change=True)

    if cfg.GWM.LOSS == 'OG':
        cfg.FLAGS.EXTENDED_FLOW_RECON_VIS = False
        cfg.FLAGS.COMP_NLL_FOR_GT = False

    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "gwm" module
    setup_logger(output=f'{cfg.OUTPUT_DIR}/main.log', distributed_rank=comm.get_rank(), name="gwm")
    with open(f'{cfg.OUTPUT_DIR}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    return cfg
