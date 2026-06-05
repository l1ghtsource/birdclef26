import argparse
import importlib.util
import json
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_old_manifest import build_old_manifest

from src.create_folds import get_folds
from src.lightning import (
    BirdClefLightningModule,
    make_trainer_train_only_dataloader,
    make_trainer_val_dataloaders,
)
from src.seed import set_seed
from src.taxonomy_merge import merged_num_classes
from src.utils import cfg_to_dict


def load_config(path: Path) -> dict:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_pretrain_cfg_", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cfg_to_dict(mod.cfg)


def parse_args():
    from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS

    p = argparse.ArgumentParser(description="Pretrain on merged old_extra_data; save EfficientNet encoder .pt weights")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "pretrain.py", help="path to config file")
    p.add_argument(
        "--build-manifest",
        action="store_true",
        help="rebuild folder manifest (see dataset.build_manifest_out or dataset.train_csv) under data_root",
    )
    p.add_argument(
        "--validate-audio",
        action="store_true",
        help="with --build-manifest: drop unreadable files (ffmpeg/decode); writes *_rejected_audio.txt",
    )
    p.add_argument(
        "--validate-extra-sources",
        action="store_true",
        help="scan dataset.extra_sources_data (inat/tsa/xc) with validate_audio_file; "
        "writes old_extra_data/extra_sources_rejected_audio.txt (use audio_blocklist_rels in config)",
    )
    p.add_argument(
        "--validate-audio-workers",
        type=int,
        default=VALIDATE_AUDIO_DEFAULT_WORKERS,
        help="thread pool for --validate-audio (manifest) and --validate-extra-sources",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        cfg["data_root"] = str((ROOT / data_root).resolve())
    data_root = Path(cfg["data_root"])

    if args.build_manifest:
        ds = cfg["dataset"]
        manifest_rel = ds.get("build_manifest_out") or ds["train_csv"]
        om = data_root / manifest_rel
        ot = Path(cfg["dataset"]["taxonomy_csv"])
        scan_rel_roots = ds.get("build_manifest_scan_rel_roots")
        if not ot.is_absolute():
            ot = (data_root / ot).resolve()
        stats_json = Path(cfg.get("pretrain_folder_stats_json", data_root / "pretrain_folder_stats.json"))
        reject_log = om.parent / f"{om.stem}_rejected_audio.txt" if args.validate_audio else None
        stats = build_old_manifest(
            data_root,
            out_manifest=om,
            out_taxonomy=ot,
            out_stats_json=stats_json,
            scan_rel_roots=list(scan_rel_roots) if scan_rel_roots else None,
            validate_audio=bool(args.validate_audio),
            validate_rejected_log=reject_log,
            validate_audio_workers=int(args.validate_audio_workers),
        )
        if int(stats.get("num_files", 0) or 0) == 0:
            raise SystemExit(
                "[pretrain] --build-manifest found 0 audio files (no scan anchors under data_root). "
                "Refusing to overwrite manifest/taxonomy. Put archives under data/old_extra_data/... or fix paths; "
                "see scripts/create_old_manifest.py discover_scan_rel_roots()."
            )
        missing_anchors = list(stats.get("missing_anchors") or [])
        if missing_anchors:
            raise SystemExit(
                "[pretrain] --build-manifest: configured scan roots were not found:\n - "
                + "\n - ".join(missing_anchors)
            )
        cfg["num_classes"] = stats["num_classes"]
        print(f"[pretrain] rebuilt folder manifest: {stats}")

    tax_rel = cfg["dataset"]["taxonomy_csv"]
    tax_path = (data_root / tax_rel).resolve() if not Path(tax_rel).is_absolute() else Path(tax_rel).resolve()
    if not tax_path.is_file():
        raise SystemExit(
            f"[pretrain] taxonomy not found: {tax_path}\n Run: python scripts/create_old_manifest.py\n"
            f" Or: python scripts/pretrain.py --config ... --build-manifest"
        )
    cfg["num_classes"] = merged_num_classes(Path(cfg["data_root"]), cfg["dataset"])

    if args.validate_extra_sources:
        from src.extra_source_audio_validate import validate_extra_source_audio_files

        st = validate_extra_source_audio_files(cfg, workers=int(args.validate_audio_workers))
        print(f"[pretrain] validate_extra_sources: {st}")

    bb_raw = cfg["model"]["backbone"]["backbone_name"]
    bb_path = Path(bb_raw)
    if not bb_path.is_absolute() and (
        bb_raw.startswith("models") or bb_raw.startswith(".") or "/" in bb_raw or "\\" in bb_raw
    ):
        cfg["model"]["backbone"]["backbone_name"] = str((ROOT / bb_path).resolve())
    set_seed(cfg["seed"])

    folds = get_folds(cfg)
    train_datasets = folds["train"]
    val_datasets = folds["val"]

    model_dir = Path(cfg["model_dir"]) / cfg["exp_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(cfg["log_dir"])

    encoder_name = cfg.get("pretrain_encoder_filename", "encoder_efficientnet_b0.pt")
    save_lightning_ckpt = bool(cfg.get("pretrain_save_lightning_checkpoints", False))

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
        if cfg.get("use_ema"):
            raise NotImplementedError(
                "EMA pretrain: set pretrain_save_lightning_checkpoints and add WeightAveraging manually"
            )

        ckpt_cb = None
        enable_ckpt = save_lightning_ckpt
        if save_lightning_ckpt:
            from lightning.pytorch.callbacks import ModelCheckpoint

            k = cfg["checkpoint_save_last_k"]
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

        trainer = pl.Trainer(
            max_epochs=cfg["n_epochs"],
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=cfg["max_norm"],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=cfg["log_dir_steps"],
            enable_checkpointing=enable_ckpt,
        )

        trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=val_loader)

        encoder_path = model_dir / encoder_name
        torch.save(lm.model.encoder.state_dict(), encoder_path)
        print(f"[pretrain] saved backbone encoder state_dict -> {encoder_path}")

        run_info = {
            "exp_name": cfg["exp_name"],
            "fold_id": fold_id,
            "config_path": str(args.config.resolve()),
            "encoder_pt": str(encoder_path.resolve()),
            "lightning_best": ckpt_cb.best_model_path if ckpt_cb else None,
            "lightning_last": ckpt_cb.last_model_path if ckpt_cb else None,
        }
        (model_dir / f"fold{fold_id}_run.json").write_text(json.dumps(run_info, indent=2, default=str))


if __name__ == "__main__":
    main()
