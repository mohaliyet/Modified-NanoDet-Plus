# In nanodet/trainer/task.py

# (Keep all existing imports at the top)
import copy
import json
import os
import warnings
from typing import Any, Dict, List
import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from nanodet.data.batch_process import stack_batch_img
from nanodet.optim import build_optimizer
from nanodet.util import convert_avg_params, gather_results, mkdir
from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager


class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        self.validation_step_outputs = []
        self.test_step_outputs = []
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        preds, loss, loss_states = self.model.forward_train(batch)

        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return loss

    def on_train_epoch_end(self) -> None:
        if not (dist.is_available() and dist.is_initialized()) or self.local_rank < 1:
            self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        
        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch)
        else:
            preds, loss, loss_states = self.model.forward_train(batch)

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        self.validation_step_outputs.append(dets)
        return dets

    def on_validation_epoch_end(self):
        self._run_evaluation_epoch_end(self.validation_step_outputs, "Validation")
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        preds = self.model(batch["img"])
        dets = self.model.head.post_process(preds, batch)
        self.test_step_outputs.append(dets)
        return dets

    def on_test_epoch_end(self) -> None:
        self._run_evaluation_epoch_end(self.test_step_outputs, "Test")
        self.test_step_outputs = []

    def _run_evaluation_epoch_end(self, step_outputs, phase):
        """
        Shared logic for processing and evaluating results at the end of
        a validation or test epoch.
        """
        results = {}
        for res in step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            num_eval_classes = self.cfg.model.arch.head.num_classes
            for img_id in list(all_results.keys()):
                for class_id in list(all_results[img_id].keys()):
                    if class_id >= num_eval_classes:
                        del all_results[img_id][class_id]

            try:
                n_images = len(all_results)
                sample_ids = list(all_results.keys())[:5]
                self.logger.info(f"{phase}: collected results for {n_images} images. sample ids: {sample_ids}")
                if n_images > 0:
                    first = all_results[sample_ids[0]]
                    try:
                        counts = {int(k): len(v) for k, v in first.items()}
                        self.logger.info(f"{phase} sample {sample_ids[0]} per-class counts: {counts}")
                    except Exception:
                        self.logger.info(f"{phase} sample structure not standard for per-class counts.")
            except Exception:
                pass

            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )

            if phase == "Validation":
                save_key_cfg = getattr(self.cfg.evaluator, "save_key", None)
                metric_key = None
                if save_key_cfg and save_key_cfg in eval_results:
                    metric_key = save_key_cfg
                else:
                    common_keys = ["val/mAP", "mAP", "map", "val_map", "map_0"]
                    for k in common_keys:
                        if k in eval_results:
                            metric_key = k
                            break
                    if metric_key is None and len(eval_results) > 0:
                        metric_key = next(iter(eval_results.keys()))

                if metric_key is not None:
                    metric = eval_results[metric_key]
                    if metric > self.save_flag:
                        self.save_flag = metric
                        best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                        mkdir(self.local_rank, best_save_path)
                        if not (dist.is_available() and dist.is_initialized()) or self.local_rank < 1:
                            self.trainer.save_checkpoint(os.path.join(best_save_path, "nanodet_model_best.ckpt"))
                            self.save_model_state(os.path.join(best_save_path, "nanodet_model_best.pth"))
                            txt_path = os.path.join(best_save_path, "eval_results.txt")
                            with open(txt_path, "a") as f:
                                f.write("Epoch:{}\n".format(self.current_epoch + 1))
                                f.write(f"Metric used for saving: {metric_key} = {metric}\n")
                                for k, v in eval_results.items():
                                    f.write("{}: {}\n".format(k, v))
            
            try:
                self.logger.log_metrics(eval_results, self.current_epoch + 1)
            except Exception:
                self.logger.info(f"Eval results: {eval_results}")
        else:
            self.logger.info(f"Skip {phase} on rank {self.local_rank}")
        
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)
        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs.get("optimizer") or kwargs.get("optim")
        optimizer_closure = kwargs.get("optimizer_closure") or kwargs.get("closure")
        for a in args:
            if optimizer is None and hasattr(a, "param_groups"):
                optimizer = a
            if optimizer_closure is None and callable(a) and not isinstance(a, (int, float, str, bytes)):
                optimizer_closure = a
        try:
            if optimizer is not None and hasattr(self, "trainer") and self.trainer.global_step <= self.cfg.schedule.warmup.steps:
                if self.cfg.schedule.warmup.name == "constant":
                    k = self.cfg.schedule.warmup.ratio
                elif self.cfg.schedule.warmup.name == "linear":
                    k = 1 - (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (1 - self.cfg.schedule.warmup.ratio)
                elif self.cfg.schedule.warmup.name == "exp":
                    k = self.cfg.schedule.warmup.ratio ** (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps)
                else:
                    raise Exception("Unsupported warm up type!")
                for pg in optimizer.param_groups:
                    if "initial_lr" not in pg:
                        pg["initial_lr"] = pg.get("lr", 0)
                    pg["lr"] = pg["initial_lr"] * k
        except Exception:
            pass
        if optimizer_closure is not None and callable(optimizer_closure):
            optimizer_closure()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        else:
            raise RuntimeError("optimizer_step hook could not find an optimizer to step")

    def scalar_summary(self, tag, phase, value, step):
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    # ------------Hooks-----------------
    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)
        if self.weight_averager is not None and hasattr(self.avg_model, "set_epoch"):
            self.avg_model.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self) -> None:
        """
        Applies averaged weights to the main model before testing to ensure
        consistency with validation performance.
        """
        if self.weight_averager:
            try:
                self.model.load_state_dict(self.avg_model.state_dict())
                self.logger.info("Successfully applied averaged weights to model for testing.")
            except Exception as e:
                self.logger.info(f"Failed to apply averaged weights: {e}; continuing with current model.")
        else:
            self.logger.info("No weight averager found. Testing with standard model weights.")

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) > 0 and len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not match the model"
                )
            elif len(avg_params) > 0:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")
            else:
                self.logger.info("Checkpoint has no average state. Weight averager will be initialized from scratch.")
