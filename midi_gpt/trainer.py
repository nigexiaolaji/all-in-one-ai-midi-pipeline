#!/usr/bin/env python3
"""
在 MIDI-GPT 预训练权重基础上微调（小数据量微调专用）。

用法:
    python midi_gpt/trainer.py \
        --init-from models/midigpt/yellow_medium-final.safetensors \
        --train-data "data/midi_gpt/train.parquet" \
        --eval-data "data/midi_gpt/valid.parquet" \
        --output-dir checkpoints/run_001

说明:
    - 从预训练 .safetensors 加载权重后继续训练（官方 train() 只支持随机初始化）
    - 小数据集建议直接按 --max-steps 控制训练步数
    - num_workers 必须为 0（C++ MIDI 解析器不支持 fork）
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from safetensors import safe_open

import midigpt._core as _core
from midigpt.augmentation.mask_bar import MaskBarConfig, MaskMode
from midigpt.inference.model import GPT2Config, GPT2LMHeadModel
from midigpt.tokenizer.tokenizer import Tokenizer
from midigpt.training.data_module import MidiGPTDataModule
from midigpt.training.lightning_module import MidiGPTLightningModule
from midigpt.training.trainer import JSONLinesLogger, TrainConfig

log = logging.getLogger("midigpt_trainer")


def load_bundle(path: str) -> tuple[dict, dict, dict]:
    """读取预训练 bundle，返回 (arch_config, encoder_config, state_dict)。

    支持官方两种格式:
      - .safetensors（format v2，元数据在文件头）
      - .pt（format v1，pickle 字典）
    """
    p = Path(path)
    if p.suffix == ".safetensors":
        meta: dict[str, str] = {}
        weights: dict[str, torch.Tensor] = {}
        with safe_open(str(p), framework="pt") as f:
            meta = f.metadata() or {}
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        arch = json.loads(meta.get("config", "{}"))
        enc = json.loads(meta.get("encoder_config", "{}"))
        return arch, enc, weights

    bundle = torch.load(p, map_location="cpu", weights_only=False)
    arch = bundle.get("config", {})
    enc = bundle.get("encoder_config", {})
    weights = bundle.get("state_dict", bundle)
    return arch, enc, weights


def build_model_from_bundle(
    path: str, max_seq_len: int
) -> tuple[GPT2LMHeadModel, Tokenizer, dict]:
    """用预训练权重构建模型（结构跟随 bundle 里的 arch 配置）。"""
    arch, enc, weights = load_bundle(path)

    tokenizer = Tokenizer(_core.EncoderConfig.from_json(json.dumps(enc)))
    n_positions = int(arch.get("n_positions", 2048))
    if max_seq_len > n_positions:
        log.warning("max_seq_len %d 超过预训练位置预算 %d，自动截断", max_seq_len, n_positions)
        max_seq_len = n_positions

    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size(),
        n_positions=n_positions,
        n_embd=int(arch.get("n_embd", 512)),
        n_layer=int(arch.get("n_layer", 6)),
        n_head=int(arch.get("n_head", 8)),
    )
    model = GPT2LMHeadModel(gpt2_cfg)
    model.encoder_config = enc

    state = weights
    if any(k.startswith("state_dict.") for k in state):
        state = {k[len("state_dict."):]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"❌ 预训练权重缺失: {missing[:10]}")
    if unexpected:
        log.warning("忽略意外权重 %d 个（如 encoder 相关键）", len(unexpected))
    log.info("✅ 已加载预训练权重: %s（%d 个张量）", path, len(state))
    return model, tokenizer, enc


def train_model(
    config: TrainConfig,
    train_path: str,
    eval_path: str | None,
    init_from: str,
    resume_from: str | None = None,
) -> None:
    L.seed_everything(config.seed, workers=True)

    model, tokenizer, _enc = build_model_from_bundle(init_from, config.max_seq_len)
    config.max_seq_len = min(config.max_seq_len, model.max_context())

    mask_cfg = (
        MaskBarConfig(
            apply_probability=config.mask_apply_probability,
            mode=MaskMode(config.mask_mode),
            bar_fraction=config.mask_bar_fraction,
            max_lookahead=config.mask_max_lookahead,
        )
        if config.mask_apply_probability > 0
        else None
    )

    data_module = MidiGPTDataModule(
        train_path=train_path,
        tokenizer=tokenizer,
        infill_probability=config.infill_probability,
        infill_bar_fraction=config.infill_bar_fraction,
        mask_bar_config=mask_cfg,
        max_seq_len=config.max_seq_len,
        max_tracks=config.max_tracks,
        min_tracks=config.min_tracks,
        min_fill_ratio=config.min_fill_ratio,
        per_device_batch_size=config.per_device_batch_size,
        num_workers=config.num_workers,
        eval_path=eval_path,
    )
    data_module.setup()

    steps_per_epoch = math.ceil(
        data_module.train_dataset_size
        / config.per_device_batch_size
        / config.gradient_accumulation_steps
    )
    total_steps = (
        config.max_steps if config.max_steps > 0 else steps_per_epoch * config.num_epochs
    )
    log.info("训练集 %d 首，每 epoch %d 步，总步数 %d", data_module.train_dataset_size, steps_per_epoch, total_steps)

    lit_module = MidiGPTLightningModule(model, config)
    lit_module.total_steps = total_steps

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    loggers = [JSONLinesLogger(config.output_dir)]
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.output_dir) / "checkpoints",
            every_n_train_steps=config.save_steps,
            save_top_k=-1,
            filename="step={step}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_steps=total_steps,
        precision={
            "fp16": "16-mixed",
            "bf16": "bf16-mixed",
            "fp32": "32",
        }[config.precision],
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm if config.max_grad_norm > 0 else None,
        val_check_interval=config.eval_steps if eval_path else None,
        limit_val_batches=config.limit_val_batches or 1.0,
        log_every_n_steps=config.logging_steps,
        default_root_dir=config.output_dir,
        callbacks=callbacks,
        logger=loggers,
    )

    trainer.fit(lit_module, data_module, ckpt_path=resume_from)

    enc_cfg = model.encoder_config
    if hasattr(enc_cfg, "to_json"):
        enc_cfg = json.loads(enc_cfg.to_json())
    final_path = Path(config.output_dir) / "model_final.safetensors"
    model.save_pretrained(str(final_path), encoder_config=enc_cfg)
    log.info("✅ 训练完成，权重: %s", final_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="MIDI-GPT 微调（从预训练权重继续训练）")
    ap.add_argument("--init-from", required=True, help="预训练权重 (.safetensors / .pt)")
    ap.add_argument("--train-data", required=True, help="训练 parquet（支持 glob）")
    ap.add_argument("--eval-data", default=None, help="验证 parquet（可选）")
    ap.add_argument("--output-dir", default="checkpoints/midi_gpt", help="输出目录")
    ap.add_argument("--config", default=None, help="TrainConfig JSON/YAML 覆盖默认值")
    ap.add_argument("--max-steps", type=int, default=2000, help="总训练步数")
    ap.add_argument("--lr", type=float, default=2e-5, help="峰值学习率")
    ap.add_argument("--batch-size", type=int, default=4, help="per-device batch")
    ap.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    ap.add_argument("--precision", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--eval-steps", type=int, default=250)
    ap.add_argument("--resume-from", default=None, help="Lightning .ckpt 断点续训")
    args = ap.parse_args()

    config = TrainConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        precision=args.precision,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    if args.config:
        base = TrainConfig.from_file(args.config)
        for field in ("n_embd", "n_layer", "n_head", "max_seq_len", "max_tracks",
                      "min_tracks", "min_fill_ratio", "infill_probability",
                      "infill_bar_fraction", "mask_apply_probability", "mask_mode",
                      "mask_bar_fraction", "mask_max_lookahead", "weight_decay",
                      "max_grad_norm", "warmup_steps", "lr_scheduler_type", "seed"):
            setattr(config, field, getattr(base, field))

    train_model(
        config,
        train_path=args.train_data,
        eval_path=args.eval_data,
        init_from=args.init_from,
        resume_from=args.resume_from,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
