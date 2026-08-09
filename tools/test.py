# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import os
import warnings

import pytorch_lightning as pl
import torch
import json
import numpy as np

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="val", help="task to run, test or val"
    )
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional override for cfg.data.val.root (path to dataset)",
    )
    # debug flag removed; run interactive notebook for single-batch debugging
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = NanoDetLightningLogger(cfg.save_dir)

    assert args.task in ["val", "test","cp_cal","cp_test"]
    cfg.update({"test_mode": args.task})

    logger.info("Setting up data...")
    # Prefer cfg.data.test when running --task test, fallback to cfg.data.val
    data_cfg = None
    if args.task == "test" and hasattr(cfg.data, "test") and cfg.data.test is not None:
        data_cfg = cfg.data.test
    elif args.task == "cp_cal" and hasattr(cfg.data, "cp_cal") and cfg.data.cp_cal is not None:
        data_cfg = cfg.data.cp_cal
    elif args.task == "cp_test" and hasattr(cfg.data, "cp_test") and cfg.data.cp_test is not None:
        data_cfg = cfg.data.cp_test
    else:
        data_cfg = cfg.data.val

    # Allow optional override of dataset root
    if args.data_root is not None:
        data_cfg = data_cfg.clone()
        data_cfg.update({"root": args.data_root})

    dataset = build_dataset(data_cfg, args.task)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )
    evaluator = build_evaluator(cfg.evaluator, dataset)

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    # Load checkpoint safely onto CPU first to avoid GPU OOM on load
    ckpt = torch.load(args.model, map_location="cpu")

    # Helper to remove common DataParallel 'module.' prefix
    def _strip_module_prefix(state_dict):
        return {k[len('module.') :] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    # Robust checkpoint handling
    if isinstance(ckpt, dict) and "pytorch-lightning_version" in ckpt:
        # PyTorch Lightning .ckpt – most reliable path
        sd = ckpt.get("state_dict", {})
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        # torch.save({'state_dict': model.state_dict(), ...})
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "epoch" in ckpt and "iter" in ckpt:
        # Old NanoDet .pth format: {"epoch": ..., "iter": ..., "state_dict": {...}}
        warnings.warn(
            "Warning! Old .pth checkpoint is deprecated. "
            "Convert the checkpoint with tools/convert_old_checkpoint.py"
        )
        new_ckpt = convert_old_model(ckpt)
        sd = new_ckpt.get("state_dict", {})
    elif isinstance(ckpt, dict):
        # Raw state dict saved directly: torch.save(model.state_dict(), path)
        # All values should be tensors; treat the dict itself as the state_dict
        warnings.warn(
            "Warning! Checkpoint appears to be a raw state_dict (.pth). "
            "Loading directly into model."
        )
        sd = ckpt
    else:
        raise RuntimeError("Unrecognized checkpoint format: could not extract state_dict")

    # Strip 'module.' if present
    sd = _strip_module_prefix(sd)

    # Decide whether sd contains top-level TrainingTask keys (model.*, avg_model.*)
    sd_keys = list(sd.keys())
    if any(k.startswith("model.") or k.startswith("avg_model.") for k in sd_keys):
        # Full PL-style state dict — load everything (model + avg_model)
        try:
            task.load_state_dict(sd)
        except Exception:
            task.load_state_dict(sd, strict=False)
    else:
        # Model-only state dict (keys like 'backbone.*') — load into task.model only
        try:
            task.model.load_state_dict(sd, strict=False)
        except Exception:
            # try prefixing keys with 'model.' and load into task
            prefixed = {"model." + k: v for k, v in sd.items()}
            try:
                task.load_state_dict(prefixed, strict=False)
            except Exception:
                raise RuntimeError("Failed to load checkpoint into task or task.model")

        # ── Critical fix for .pth + weight_averager ───────────────────────────
        # When loaded from .pth the EMA weights go into task.model, but
        # task.avg_model still holds random (initial) weights.  on_test_epoch_start
        # then overwrites task.model with avg_model, discarding the loaded weights.
        # Sync avg_model now so the overwrite becomes a no-op.
        if hasattr(task, "avg_model"):
            task.avg_model.load_state_dict(task.model.state_dict())
            logger.info("Synced avg_model with loaded .pth weights.")
        # Also restore weight_averager state so it tracks the loaded weights
        if task.weight_averager is not None:
            task.weight_averager.load_from(task.model)
            logger.info("Re-initialised weight_averager from loaded .pth weights.")

    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices = "cpu", 1
    else:
        accelerator, devices = "gpu", cfg.device.gpu_ids


    # If user asked for debug mode, run a single batch through the model and post_process
    # interactive debugging was removed from the script; use the notebook helper

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        logger=logger,
    )
    logger.info("Starting testing...")
    # PyTorch Lightning v2 prefers the keyword 'dataloaders' for test
    try:
        trainer.test(task, dataloaders=dataloader)
    except TypeError:
        # fall back to positional arg for older PL versions
        trainer.test(task, dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)