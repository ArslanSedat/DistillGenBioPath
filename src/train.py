#TODO : Check Notion notes (13/10)

from __future__ import annotations
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import cv2
import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from wsistream.torch import MonitoredLoader
from wsistream.types import resolve_slide_paths
from config import TrainConfig, get_device, get_paths, seed_everything, validate
from data import TILE_VIEW, build_train_dataset, build_val_dataset
from distiller import Distiller
from export import export_student_weights, snapshot_name
from models import REGISTRY, load_model

logger = logging.getLogger(__name__)

_AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
_NO_DECAY_SUFFIXES = ("cls_token", "pos_embed", "reg_token", "dist_token")

def _worker_init(worker_id: int) -> None:
    # Disable OpenCV's internal threadpool inside each forked data worker. Otherwise a
    # worker can inherit OpenCV's pthread mutexes in a locked state and deadlock on its
    # first tissue-detection / resize call (the data pipeline is pure OpenCV). The data
    # workers already provide the parallelism, so per-worker threading only oversubscribes.
    cv2.setNumThreads(0)

def build_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr: float):
    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=min_lr)

def build_param_groups(modules, weight_decay: float) -> list[dict]:
    """Apply weight decay only to >=2-D weight matrices; exclude biases, norm gains,
    and ViT token / positional embeddings (standard ViT-training practice)."""
    decay, no_decay, seen = [], [], set()
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param.ndim <= 1 or name.endswith(_NO_DECAY_SUFFIXES):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
        ]

def load_checkpoint(path, student, distiller, optimizer, scheduler, device) -> int:
    """Restore training state from a checkpoint and return the step it was saved at."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    distiller.load_state_dict(ckpt["distiller"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"]

def scalar_metrics(losses: dict, prefix: str) -> dict:
    return {f"{prefix}/{k}": v.item() for k, v in losses.items() if torch.is_tensor(v)}

def contribution_metrics(losses: dict, distiller: Distiller, prefix: str) -> dict[str, float]:
    """Report realized weighted loss contributions without changing optimization."""

    def value(name: str) -> float | None:
        tensor = losses.get(name)
        return tensor.detach().item() if torch.is_tensor(tensor) else None

    metrics: dict[str, float] = {}
    cls = value("cls_loss")
    patch = value("patch_loss")
    weighted_cls = distiller.cls_weight * cls if cls is not None else None
    weighted_patch = distiller.patch_weight * patch if patch is not None else None
    total = (weighted_cls or 0.0) + (weighted_patch or 0.0)

    if weighted_cls is not None:
        metrics[f"{prefix}/contrib/cls_weighted"] = weighted_cls
    if weighted_patch is not None:
        metrics[f"{prefix}/contrib/patch_weighted"] = weighted_patch
    if total > 0:
        if weighted_cls is not None:
            metrics[f"{prefix}/contrib/cls_fraction_total"] = weighted_cls / total
        if weighted_patch is not None:
            metrics[f"{prefix}/contrib/patch_fraction_total"] = weighted_patch / total

    cls_objective = getattr(distiller, "cls_objective", None)
    component_weights = getattr(cls_objective, "weights", {})
    cls_component_total = 0.0
    cls_components: dict[str, float] = {}
    for name, weight in component_weights.items():
        raw = value(f"cls_{name}")
        if raw is None:
            continue
        weighted = distiller.cls_weight * weight * raw
        cls_components[name] = weighted
        cls_component_total += weighted
        metrics[f"{prefix}/contrib/cls_{name}_weighted"] = weighted

    if cls_component_total > 0:
        for name, weighted in cls_components.items():
            metrics[f"{prefix}/contrib/cls_{name}_fraction_cls"] = weighted / cls_component_total
            if total > 0:
                metrics[f"{prefix}/contrib/cls_{name}_fraction_total"] = weighted / total

    return metrics

@torch.no_grad()
def evaluate(
    teacher,
    student,
    distiller,
    val_slide_paths,
    num_val_slides,
    cfg,
    device,
    use_amp,
    amp_dtype,
    step
    ):
    student.eval()
    distiller.eval()

    dataset = build_val_dataset(val_slide_paths, cfg.data, seed=cfg.seed, num_slides=num_val_slides)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.val_num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.val_num_workers > 0 else None,
        worker_init_fn=_worker_init if cfg.data.val_num_workers > 0 else None
        )
    teacher_prefix = teacher.num_prefix_tokens
    student_prefix = student.num_prefix_tokens

    totals: dict[str, float] = {}
    num_batches = 0
    for batch in loader:
        images = batch[TILE_VIEW].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            teacher_tokens = teacher(images).detach()
            student_tokens = student(images)
            losses = distiller(student_tokens, teacher_tokens, student_prefix, teacher_prefix)
        for key, value in losses.items():
            if torch.is_tensor(value):
                totals[key] = totals.get(key, 0.0) + value.item()
        num_batches += 1

    n = max(num_batches, 1)
    averaged = {k: v / n for k, v in totals.items()}
    metrics = {f"val/{k}": v for k, v in averaged.items()}
    metrics.update(
        contribution_metrics(
            {k: torch.tensor(v, device=device) for k, v in averaged.items()}, distiller, "val"
            )
        )
    wandb.log(metrics, step=step)
    logger.info(
        "VALIDATION step=%d %s",
        step,
        " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
        )

    student.train()
    distiller.train()

def save_checkpoint(path: Path, step, loss, student, distiller, optimizer, scheduler):
    torch.save(
        {
            "step": step,
            "loss": loss,
            "student": student.state_dict(),
            "distiller": distiller.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
            },
        path
        )
    logger.info("Saved checkpoint: %s", path)

def train(cfg: TrainConfig, run_name: str) -> None:
    cv2.setNumThreads(0)
    device = get_device()
    paths = get_paths()
    seed_everything(cfg.seed)
    run_dir = paths["output_dir"] / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "distillpath"),
        name=run_name,
        config=dataclasses.asdict(cfg),
        dir=str(run_dir)
        )

    teacher = load_model(cfg.teacher.name, device, cfg.teacher.weights, require_weights=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student_requires_weights = REGISTRY[cfg.student.name].info.requires_pretrained
    student = load_model(cfg.student.name, device, cfg.student.weights, require_weights=student_requires_weights)
    student_dim = REGISTRY[cfg.student.name].info.feature_dim
    teacher_dim = REGISTRY[cfg.teacher.name].info.feature_dim
    distiller = Distiller(cfg, student_dim, teacher_dim).to(device)
    teacher_prefix = teacher.num_prefix_tokens
    student_prefix = student.num_prefix_tokens
    amp_dtype = _AMP_DTYPES[cfg.precision] if device.type == "cuda" else None
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None
    trainable_params = list(student.parameters()) + list(distiller.parameters())
    optimizer = torch.optim.AdamW(build_param_groups([student, distiller], cfg.optim.weight_decay), lr=cfg.optim.lr)
    scheduler = build_scheduler(optimizer, cfg.optim.warmup_steps, cfg.max_steps, cfg.optim.min_lr)

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.resume, student, distiller, optimizer, scheduler, device)
        logger.info("Resumed from %s at step %d", cfg.resume, start_step)

    logger.info(
        "Teacher %s (%d-dim) -> Student %s (%d-dim)",
        cfg.teacher.name,
        teacher_dim,
        cfg.student.name,
        student_dim
        )
    logger.info(
        "CLS loss=%s (w=%.3f) | patch loss=%s (w=%.3f) | projector=%s",
        cfg.cls_loss.type,
        cfg.cls_loss.weight,
        cfg.patch_loss.type,
        cfg.patch_loss.weight,
        cfg.projector.variant if distiller.projector is not None else "none"
        )
    logger.info(
        "batch_size=%d precision=%s max_steps=%d lr=%.2e",
        cfg.batch_size,
        cfg.precision,
        cfg.max_steps,
        cfg.optim.lr
        )

    train_dataset = build_train_dataset(str(paths["data_dir"]), cfg.data, seed=cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        worker_init_fn=_worker_init if cfg.data.num_workers > 0 else None
        )
    monitored_loader = MonitoredLoader(train_loader, dataset=train_dataset, device=device, log_every=cfg.log_every)

    val_slide_paths = None
    num_val_slides = 0
    if paths["val_data_dir"] is not None:
        val_slide_paths = str(paths["val_data_dir"])
        num_val_slides = len(resolve_slide_paths(val_slide_paths))
        logger.info("Validation: %s (%d slides)", val_slide_paths, num_val_slides)
    else:
        logger.info("Validation disabled (VAL_DATA_DIR not set)")

    run_info = dataclasses.asdict(cfg)
    run_info["run_name"] = run_name
    run_info["val_data_dir"] = val_slide_paths
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    student.train()
    distiller.train()

    if val_slide_paths is not None and start_step == 0:
        evaluate(
            teacher,
            student,
            distiller,
            val_slide_paths,
            num_val_slides,
            cfg,
            device,
            use_amp,
            amp_dtype,
            step=0
            )

    step_iter = iter(monitored_loader)
    last_loss = float("nan")
    accum = cfg.grad_accum_steps
    for step in range(start_step + 1, cfg.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = None
        nonfinite = False
        for _ in range(accum):
            batch = next(step_iter)
            teacher_input = batch[TILE_VIEW].to(device, non_blocking=True)
            student_input = teacher_input
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    teacher_tokens = teacher(teacher_input)
                student_tokens = student(student_input)
                losses = distiller(student_tokens, teacher_tokens.detach(), student_prefix, teacher_prefix)
                loss = losses["total_loss"] / accum
            if not torch.isfinite(loss):
                nonfinite = True
                break
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if nonfinite:
            logger.warning("step=%d: non-finite loss, skipping optimizer update", step)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.optim.clip_grad).item()
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        last_loss = losses["total_loss"].item()
        current_lr = optimizer.param_groups[0]["lr"]
        payload = monitored_loader.mark_step(
            extra={
                **scalar_metrics(losses, "train"),
                **contribution_metrics(losses, distiller, "train"),
                "train/lr": current_lr,
                "train/grad_norm": grad_norm
                }
            )
        if payload is not None:
            payload["train/step"] = step
            wandb.log(payload, step=step)
            # Append the weighted cls/patch loss split to the step log.
            comp = ""
            if "cls_loss" in losses and "patch_loss" in losses:
                c = losses["cls_loss"].item()
                p = losses["patch_loss"].item()
                cls_c = distiller.cls_weight * c
                patch_c = distiller.patch_weight * p
                tot = cls_c + patch_c or 1.0
                cp, pp = 100 * cls_c / tot, 100 * patch_c / tot
                comp = f" | cls={c:.4f} patch={p:.4f} share={cp:.0f}/{pp:.0f}"
                cls_objective = getattr(distiller, "cls_objective", None)
                cls_weights = getattr(cls_objective, "weights", {})
                if "cosine" in cls_weights and "rkd" in cls_weights:
                    cos = losses.get("cls_cosine")
                    rkd = losses.get("cls_rkd")
                    if torch.is_tensor(cos) and torch.is_tensor(rkd):
                        cos_c = distiller.cls_weight * cls_weights["cosine"] * cos.item()
                        rkd_c = distiller.cls_weight * cls_weights["rkd"] * rkd.item()
                        cls_tot = cos_c + rkd_c or 1.0
                        cos_pct = 100 * cos_c / cls_tot
                        rkd_pct = 100 * rkd_c / cls_tot
                        comp += f" cls_cos/rkd={cos_pct:.0f}/{rkd_pct:.0f}"
            elif "cls_loss" in losses:
                comp = f" | cls={losses['cls_loss'].item():.4f}"
            elif "patch_loss" in losses:
                comp = f" | patch={losses['patch_loss'].item():.4f}"
            logger.info(
                "step=%d loss=%.4f lr=%.2e grad=%.2f%s | data_wait=%.0f%% %.0f patch/s slides=%d",
                step,
                last_loss,
                current_lr,
                grad_norm,
                comp,
                payload.get("loader/data_fraction", 0.0) * 100,
                payload.get("loader/patches_per_sec", 0.0),
                int(payload.get("pipeline/slides_processed", 0))
                )

        if val_slide_paths is not None and step % cfg.val_every == 0:
            evaluate(
                teacher,
                student,
                distiller,
                val_slide_paths,
                num_val_slides,
                cfg,
                device,
                use_amp,
                amp_dtype,
                step
                )

        if step % cfg.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step}.pt",
                step,
                last_loss,
                student,
                distiller,
                optimizer,
                scheduler
                )
            export_student_weights(student.state_dict(), snapshot_name(cfg.export_name, step))

    save_checkpoint(
        ckpt_dir / "final.pt",
        cfg.max_steps,
        last_loss,
        student,
        distiller,
        optimizer,
        scheduler
        )
    export_student_weights(student.state_dict(), cfg.export_name)
    logger.info(
        "Training complete (%d steps). Loader stats: %s",
        cfg.max_steps,
        monitored_loader.lifetime_stats()
        )
    wandb.finish()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
        )
    train_cfg: TrainConfig = OmegaConf.to_object(cfg)
    validate(train_cfg)
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    experiment = HydraConfig.get().runtime.choices.get("experiment", "run")
    run_name = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(train_cfg, run_name)

if __name__ == "__main__":
    main()