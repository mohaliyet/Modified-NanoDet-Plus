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

"""
Export NanoDet-Plus model for deployment.

Supported formats:
  --format onnx        → model.onnx  (for TensorRT, OpenVINO, ONNX Runtime, etc.)
  --format torchscript → model.torchscript.pt  (for PyTorch C++ / mobile)
  --format both        → both of the above (default)

Usage:
  python tools/export.py \\
      --config config/parasight_plus.yml \\
      --model  /path/to/nanodet_model_best.ckpt \\
      --output /path/to/export_dir \\
      --format both \\
      --input-size 320 320   # H W  (overrides config)
      --batch-size 1
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn

from nanodet.model.arch import build_model
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
)


# ---------------------------------------------------------------------------
# Deployment wrapper
# ---------------------------------------------------------------------------

class DeployModel(nn.Module):
    """Sigmoid-only deployment wrapper.

    Strips aux_fpn and aux_head (training-only).  Returns main-head
    predictions ``(B, N, num_classes + 4*(reg_max+1))``, activated with
    sigmoid inside the head's ``_forward_onnx``.

    Use ``--variant sigmoid`` (default).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn
        self.head = model.head

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.fpn(x)
        x = self.head(x)
        return x


class DeployModelWithAux(nn.Module):
    """Full deployment wrapper that includes the auxiliary head.

    Returns a **2-tuple**:
      * ``output_main`` — main head predictions (sigmoid-activated),
        shape ``(B, N, num_classes + 4*(reg_max+1))``
      * ``output_aux``  — auxiliary head raw logits (NOT activated),
        shape ``(B, N, num_classes + 4*(reg_max+1))``

    ``output_aux`` carries the softmax class logits needed for
    conformal prediction / full-score inference.  The ONNX runtime
    consumer (``OnnxPredictor``) detects the second output and passes
    it as ``aux_preds`` to ``post_process``.

    Use ``--variant full``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.fpn = model.fpn
        self.head = model.head
        self.aux_fpn = model.aux_fpn
        self.aux_head = model.aux_head

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        fpn_feat = self.fpn(feat)
        aux_fpn_feat = self.aux_fpn(feat)
        dual_fpn_feat = tuple(
            torch.cat([f, aux_f], dim=1)
            for f, aux_f in zip(fpn_feat, aux_fpn_feat)
        )
        main_out = self.head(fpn_feat)       # sigmoid-activated via _forward_onnx
        aux_out  = self.aux_head(dual_fpn_feat)  # raw logits
        return main_out, aux_out


# ---------------------------------------------------------------------------
# Checkpoint loading  (mirrors robust logic from test.py)
# ---------------------------------------------------------------------------

def _strip_module_prefix(state_dict):
    return {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


def load_checkpoint(task: TrainingTask, model_path: str, logger) -> None:
    """Load .ckpt or .pth weights into *task*, then sync avg_model."""
    ckpt = torch.load(model_path, map_location="cpu")

    # ── detect format ──────────────────────────────────────────────────────
    if isinstance(ckpt, dict) and "pytorch-lightning_version" in ckpt:
        sd = ckpt.get("state_dict", {})

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]

    elif isinstance(ckpt, dict) and "epoch" in ckpt and "iter" in ckpt:
        warnings.warn(
            "Old .pth checkpoint format detected. "
            "Consider converting with tools/convert_old_checkpoint.py"
        )
        new_ckpt = convert_old_model(ckpt)
        sd = new_ckpt.get("state_dict", {})

    elif isinstance(ckpt, dict):
        # Raw state_dict saved with torch.save(model.state_dict(), path)
        warnings.warn("Treating checkpoint as raw state_dict.")
        sd = ckpt

    else:
        raise RuntimeError(f"Unrecognised checkpoint format: {model_path}")

    sd = _strip_module_prefix(sd)

    # ── load into task ──────────────────────────────────────────────────────
    if any(k.startswith("model.") or k.startswith("avg_model.") for k in sd):
        try:
            task.load_state_dict(sd)
        except Exception:
            task.load_state_dict(sd, strict=False)
    else:
        task.model.load_state_dict(sd, strict=False)
        # Sync avg_model so on_test_epoch_start doesn't overwrite loaded weights
        if hasattr(task, "avg_model"):
            task.avg_model.load_state_dict(task.model.state_dict())
        if task.weight_averager is not None:
            task.weight_averager.load_from(task.model)

    logger.info(f"Loaded weights from {model_path}")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_onnx(
    deploy_model: nn.Module,
    dummy: torch.Tensor,
    out_path: str,
    opset: int = 18,
    dual_output: bool = False,
):
    """Export model to ONNX using the legacy exporter (avoids dynamo/version issues).

    Args:
        dual_output: When True, the model returns (main_out, aux_out) and
            both are registered as named ONNX outputs.
    """
    import onnx

    if dual_output:
        output_names = ["output_main", "output_aux"]
        dynamic_axes = {
            "input":       {0: "batch"},
            "output_main": {0: "batch"},
            "output_aux":  {0: "batch"},
        }
    else:
        output_names = ["output"]
        dynamic_axes = {
            "input":  {0: "batch"},
            "output": {0: "batch"},
        }

    torch.onnx.export(
        deploy_model,
        dummy,
        out_path,
        opset_version=opset,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,   # use legacy TorchScript exporter
    )
    # Validate
    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    print(f"[Export] ONNX model saved to: {out_path}  (opset {opset})")


def export_torchscript(deploy_model: nn.Module, dummy: torch.Tensor, out_path: str):
    """Export model to TorchScript via tracing."""
    with torch.no_grad():
        traced = torch.jit.trace(deploy_model, dummy)
    traced.save(out_path)
    print(f"[Export] TorchScript model saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Export NanoDet-Plus for deployment")
    parser.add_argument("--config",     type=str, required=True,  help="YAML config file")
    parser.add_argument("--model",      type=str, required=True,  help=".ckpt or .pth checkpoint")
    parser.add_argument("--output",     type=str, default="./exports", help="Output directory")
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["onnx", "torchscript", "both"],
        help="Export format",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "full"],
        help=(
            "sigmoid: export main head only (no aux head, smaller model). "
            "full: export main + aux head; ONNX has two outputs "
            "(output_main, output_aux) so OnnxPredictor can attach softmax scores."
        ),
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Input image size (H W). Defaults to cfg.data.val.input_size.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input")
    parser.add_argument("--opset",      type=int, default=18,    help="ONNX opset version (>=18 recommended for PyTorch 2.x)")
    parser.add_argument("--device",     type=str, default="cpu", help="Device for export (cpu / cuda:0)")
    return parser.parse_args()


def main(args):
    load_config(cfg, args.config)
    mkdir(-1, args.output)
    logger = NanoDetLightningLogger(args.output)

    # ── resolve input size ─────────────────────────────────────────────────
    if args.input_size:
        h, w = args.input_size
    else:
        hw = cfg.data.val.input_size  # e.g. [320, 320]
        if isinstance(hw, (list, tuple)):
            h, w = hw[0], hw[1]
        else:
            h = w = int(hw)

    logger.info(f"Input size: {h} x {w}   device: {args.device}")

    # ── build model & load weights ─────────────────────────────────────────
    logger.info("Building model...")
    task = TrainingTask(cfg, evaluator=None)
    load_checkpoint(task, args.model, logger)

    # ── build deployment wrapper ───────────────────────────────────────────
    src_model = task.model
    src_model.eval()

    dual_output = args.variant == "full"
    if dual_output:
        deploy = DeployModelWithAux(src_model).to(args.device).eval()
        logger.info("Export variant: full (main head + aux head, dual ONNX output)")
    else:
        deploy = DeployModel(src_model).to(args.device).eval()
        logger.info("Export variant: sigmoid (main head only)")

    dummy = torch.zeros(args.batch_size, 3, h, w, device=args.device)

    # ── sanity forward pass ────────────────────────────────────────────────
    logger.info("Running sanity forward pass...")
    with torch.no_grad():
        out = deploy(dummy)
    if isinstance(out, (list, tuple)) and isinstance(out[0], torch.Tensor) and out[0].dim() == 3:
        logger.info(f"Output: {len(out)} tensors, shapes: {[o.shape for o in out]}")
    elif isinstance(out, torch.Tensor):
        logger.info(f"Output shape: {out.shape}")
    else:
        logger.info(f"Output: {out}")

    # ── export ────────────────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(args.model))[0]
    variant_suffix = "_full" if dual_output else ""

    if args.format in ("onnx", "both"):
        try:
            import onnx  # noqa: F401 — check early so error is clear
        except ImportError:
            print(
                "\n[Export] ERROR: 'onnx' package is not installed.\n"
                "         Install it with:  pip install onnx\n"
                "         Skipping ONNX export.\n"
            )
            logger.info("onnx package not installed — skipping ONNX export.")
        else:
            try:
                onnx_path = os.path.join(args.output, f"{stem}{variant_suffix}.onnx")
                export_onnx(deploy, dummy, onnx_path, opset=args.opset, dual_output=dual_output)
            except Exception as e:
                print(f"\n[Export] ERROR during ONNX export: {e}\n")
                logger.info(f"ONNX export failed: {e}")

    if args.format in ("torchscript", "both"):
        ts_path = os.path.join(args.output, f"{stem}{variant_suffix}.torchscript.pt")
        export_torchscript(deploy, dummy, ts_path)

    logger.info("Export complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
