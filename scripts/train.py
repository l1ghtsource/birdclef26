import argparse
import importlib.util
import json
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, WeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.swa_utils import get_ema_avg_fn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.create_folds import get_folds  # noqa: E402
from src.lightning import (  # noqa: E402
    BirdClefLightningModule,
    make_trainer_train_only_dataloader,
    make_trainer_val_dataloaders,
)
from src.seed import set_seed  # noqa: E402
from src.utils import cfg_to_dict  # noqa: E402


def load_config(path: Path) -> dict:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_train_cfg_", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cfg_to_dict(mod.cfg)


def parse_args():
    p = argparse.ArgumentParser(description="birdclef training script")
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "config1.py",
        help="path to config file",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        cfg["data_root"] = str((ROOT / data_root).resolve())
    model_cfg = cfg["model"]
    mt = str(model_cfg.get("model_type", "sed")).lower()
    if mt == "perch_sed":
        for key in ("perch_frontend_ckpt", "perch_backbone_ckpt"):
            raw = model_cfg.get(key)
            if raw is None:
                continue
            raw = str(raw)
            bb_path = Path(raw)
            if not bb_path.is_absolute() and (
                raw.startswith("models") or raw.startswith(".") or "/" in raw or "\\" in raw
            ):
                model_cfg[key] = str((ROOT / bb_path).resolve())
    else:
        bb_raw = model_cfg["backbone"]["backbone_name"]
        bb_path = Path(bb_raw)
        if not bb_path.is_absolute() and (
            bb_raw.startswith("models") or bb_raw.startswith(".") or "/" in bb_raw or "\\" in bb_raw
        ):
            model_cfg["backbone"]["backbone_name"] = str((ROOT / bb_path).resolve())
        ck_raw = model_cfg["backbone"].get("init_checkpoint")
        if ck_raw:
            ck_str = str(ck_raw)
            ck_path = Path(ck_str)
            if not ck_path.is_absolute() and (
                ck_str.startswith("models") or ck_str.startswith(".") or "/" in ck_str or "\\" in ck_str
            ):
                model_cfg["backbone"]["init_checkpoint"] = str((ROOT / ck_path).resolve())
    set_seed(cfg["seed"])

    folds = get_folds(cfg)
    train_datasets = folds["train"]
    val_datasets = folds["val"]

    model_dir = Path(cfg["model_dir"]) / cfg["exp_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(cfg["log_dir"])

    fold_ids = cfg["curr_folds"] if not cfg["do_full_retrain"] else [0]

    for i, train_ds in enumerate(train_datasets):
        fold_id = fold_ids[i] if i < len(fold_ids) else i
        has_val = val_datasets is not None

        if has_val:
            val_ds = val_datasets[i]
            train_loader, val_loader, total_steps = make_trainer_val_dataloaders(cfg, train_ds, val_ds)
        else:
            train_loader, total_steps = make_trainer_train_only_dataloader(cfg, train_ds)
            val_loader = None

        lm_cfg = dict(cfg)
        lm_cfg["ss_bank_train_indices"] = list(train_ds.indices)
        lm = BirdClefLightningModule(lm_cfg, total_optimizer_steps=total_steps, fold_id=fold_id)

        callbacks = [LearningRateMonitor(logging_interval="step")]
        if cfg["use_ema"]:
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.WeightAveraging.html
            callbacks.append(WeightAveraging(avg_fn=get_ema_avg_fn(cfg["ema_decay"])))

        k = cfg["checkpoint_save_last_k"]
        if has_val:
            ckpt_cb = ModelCheckpoint(
                dirpath=str(model_dir),
                filename=f"fold{fold_id}_{{epoch:02d}}-{{val_macro_auc:.4f}}",
                monitor="epoch",
                mode="max",
                save_top_k=k,
                save_last=True,
            )
        else:
            ckpt_cb = ModelCheckpoint(
                dirpath=str(model_dir),
                filename=f"fold{fold_id}_{{epoch:02d}}-train",
                monitor="epoch",
                mode="max",
                save_top_k=k,
                save_last=True,
            )

        callbacks.append(ckpt_cb)

        logger = None
        if cfg["do_tensorboard_log"]:
            logger = TensorBoardLogger(
                save_dir=str(log_root),
                name=cfg["tensorboard_project"],
                version=f"{cfg['exp_name']}_fold{fold_id}",
                default_hp_metric=False,
            )

        devices = cfg["devices"]
        devices = [int(x) for x in devices.split(",") if x.strip()]
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if accelerator == "cpu":
            devices = 1

        trainer_kwargs = {
            "max_epochs": cfg["n_epochs"],
            "accelerator": accelerator,
            "devices": devices,
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": cfg["log_dir_steps"],
            "enable_checkpointing": True,
        }
        if lm.automatic_optimization:
            trainer_kwargs["gradient_clip_val"] = cfg["max_norm"]

        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)

        run_info = {
            "exp_name": cfg["exp_name"],
            "fold_id": fold_id,
            "config_path": str(args.config.resolve()),
            "best_model_path": ckpt_cb.best_model_path,
            "last_model_path": ckpt_cb.last_model_path,
        }
        (model_dir / f"fold{fold_id}_run.json").write_text(json.dumps(run_info, indent=2, default=str))


if __name__ == "__main__":
    main()
