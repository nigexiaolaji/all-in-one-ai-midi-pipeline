#!/usr/bin/env python3
"""
用微调后的模型（或官方预训练模型）生成 MIDI。

用法（从头生成）:
    python midi_gpt/generate.py scratch \
        --checkpoint checkpoints/midi_gpt/model_final.safetensors \
        --bars 8 --tracks 4 --out generated/song1.mid \
        --attrs '{"note_density": 5, "max_polyphony": 3}'

用法（对已有 MIDI 做局部补全 infill）:
    python midi_gpt/generate.py infill \
        --checkpoint checkpoints/midi_gpt/model_final.safetensors \
        --midi "../output/ノエル/ノエル_vocals.mid" \
        --track 0 --bars 4 5 6 7 --out generated/filled.mid

说明:
    - --checkpoint 指向微调权重；不传则用官方预训练模型（--pretrained yellow_medium）
    - 可用属性通过 --attrs JSON 传；先用 --list-attrs 查看模型支持的属性
"""

import argparse
import json
import sys
from pathlib import Path

from midigpt import Score, Track, Bar
from midigpt.inference import (
    InferenceEngine,
    GenerationRequest,
    InferenceConfig,
    TrackPrompt,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def load_engine(checkpoint: str | None, pretrained: str | None) -> InferenceEngine:
    if checkpoint:
        print(f"加载微调权重: {checkpoint}")
        return InferenceEngine.from_checkpoint(checkpoint)
    name = pretrained or "yellow_medium"
    print(f"加载官方预训练模型: {name}")
    return InferenceEngine.from_pretrained(name)


def list_attrs(engine: InferenceEngine) -> None:
    sizes = engine._analyzer.attribute_sizes()
    labels = engine._analyzer.attribute_value_labels()
    for name, size in sizes.items():
        vals = labels.get(name, [])
        print(f"  {name} (0~{size - 1}): {vals}")


def cmd_scratch(args) -> None:
    engine = load_engine(args.checkpoint, args.pretrained)
    if args.list_attrs:
        list_attrs(engine)
        return

    attrs = json.loads(args.attrs) if args.attrs else {}
    bars = args.bars
    tracks = []
    for i in range(args.tracks):
        instrument = args.instruments[i] if i < len(args.instruments) else args.instrument
        tracks.append(
            Track(
                bars=[Bar() for _ in range(bars)],
                instrument=instrument,
                track_type="melodic",
            )
        )
    score = Score(tracks=tracks)

    request = GenerationRequest(
        tracks=[
            TrackPrompt(
                id=i,
                bars=list(range(bars)),
                autoregressive=True,
                attributes=attrs,
            )
            for i in range(args.tracks)
        ],
        config=InferenceConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            model_dim=bars,
            seed=args.seed,
        ),
    )
    result = engine.session(score, request).run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_midi(str(out))
    n_notes = sum(len(b.notes) for t in result.tracks for b in t.bars)
    print(f"✅ 生成完成: {out}（{n_notes} 个音符，{len(result.tracks)} 轨）")


def cmd_infill(args) -> None:
    engine = load_engine(args.checkpoint, args.pretrained)
    if args.list_attrs:
        list_attrs(engine)
        return

    midi_bytes = Path(args.midi).read_bytes()
    score = Score.from_bytes(midi_bytes)
    print(f"已加载 {Path(args.midi).name}: {len(score.tracks)} 轨")

    attrs = json.loads(args.attrs) if args.attrs else {}
    bar_map = {t.id: t for t in score.tracks} if hasattr(score, "id") else {}
    prompts = []
    for tid in args.track:
        prompts.append(
            TrackPrompt(
                id=tid,
                bars=args.bars,
                attributes=attrs,
            )
        )
    # 未指定的轨道保持原样
    for i in range(len(score.tracks)):
        if i not in args.track:
            prompts.append(TrackPrompt(id=i, bars=[], ignore=True))

    request = GenerationRequest(
        tracks=prompts,
        config=InferenceConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            model_dim=args.bars[0] if args.bars else 4,
            seed=args.seed,
        ),
    )
    result = engine.session(score, request).run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_midi(str(out))
    print(f"✅ 补全完成: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MIDI-GPT 生成")
    ap.add_argument("--checkpoint", default=None, help="微调权重路径")
    ap.add_argument("--pretrained", default=None, help='官方模型名（如 yellow_medium）')
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--attrs", default=None, help='属性 JSON，如 \'{"note_density": 5}\'')
    ap.add_argument("--list-attrs", action="store_true", help="列出模型支持的属性")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("scratch", help="从头生成")
    s.add_argument("--bars", type=int, default=8)
    s.add_argument("--tracks", type=int, default=4)
    s.add_argument("--instrument", type=int, default=0, help="GM 音色编号")
    s.add_argument("--instruments", type=int, nargs="*", default=[], help="每轨 GM 音色")
    s.add_argument("--out", required=True)

    i = sub.add_parser("infill", help="对已有 MIDI 局部补全")
    i.add_argument("--midi", required=True)
    i.add_argument("--track", type=int, nargs="+", required=True, help="要重新生成的轨道 id")
    i.add_argument("--bars", type=int, nargs="+", required=True, help="要重新生成的小节号")
    i.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.list_attrs:
        engine = load_engine(args.checkpoint, args.pretrained)
        list_attrs(engine)
        return 0
    if args.cmd == "scratch":
        cmd_scratch(args)
    elif args.cmd == "infill":
        cmd_infill(args)
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
