#!/usr/bin/env python3
"""
将 pipeline 输出的多轨 MIDI 转换为 MIDI-GPT 训练用的 GigaMIDI 格式 parquet。

用法:
    python midi_gpt/prepare_data.py \
        --input "../output/*/*_merged.mid" \
        --output-dir data/midi_gpt

说明:
    - 每行一条 MIDI（music 列存原始字节，兼容 MIDI-GPT 训练数据格式）
    - 自动做 train/valid 划分（默认 valid 15%）
    - 使用 Score.from_bytes 校验（Windows 下 from_midi 对非 ASCII 路径会失败）
"""

import argparse
import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from midigpt import Score

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCHEMA = pa.schema(
    [
        pa.field("music", pa.binary()),
        pa.field("num_tracks", pa.int64()),
        pa.field("total_notes", pa.int64()),
    ]
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent


def collect_midis(patterns: list[str]) -> list[str]:
    """展开 glob 模式，按路径排序去重。"""
    files: set[str] = set()
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        files.update(expanded)
    return sorted(files)


def midis_to_parquet(midi_paths: list[str], output_path: str) -> int:
    """把 MIDI 文件写成一个 parquet shard，返回写入行数。"""
    rows = []
    dropped = 0
    for path in midi_paths:
        try:
            data = Path(path).read_bytes()
            score = Score.from_bytes(data)
        except Exception as exc:
            print(f"  ✗ 跳过 {Path(path).name}: {str(exc)[:80]}")
            dropped += 1
            continue
        n_notes = sum(len(bar.notes) for t in score.tracks for bar in t.bars)
        if len(score.tracks) < 1 or n_notes == 0:
            print(f"  ✗ 跳过 {Path(path).name}: 空曲")
            dropped += 1
            continue
        rows.append(
            {
                "music": data,
                "num_tracks": len(score.tracks),
                "total_notes": n_notes,
            }
        )

    if not rows:
        print("❌ 没有可用的 MIDI 文件，请检查 --input 路径")
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, output_path)
    total_notes = sum(r["total_notes"] for r in rows)
    print(f"✅ {output_path}: {len(rows)} 首 / {total_notes} 音符（跳过 {dropped} 首）")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="MIDI → MIDI-GPT parquet")
    ap.add_argument(
        "--input",
        nargs="+",
        default=[str(PIPELINE_ROOT / ".." / "output" / "*" / "*_merged.mid")],
        help='glob 模式，如 "../output/*/*_merged.mid"',
    )
    ap.add_argument(
        "--output-dir",
        default=str(PIPELINE_ROOT / "data" / "midi_gpt"),
        help="输出目录（写入 train.parquet / valid.parquet）",
    )
    ap.add_argument(
        "--valid-fraction",
        type=float,
        default=0.15,
        help="验证集比例（按歌曲数划分，默认 0.15）",
    )
    ap.add_argument("--no-valid", action="store_true", help="不划分验证集")
    args = ap.parse_args()

    midis = collect_midis(args.input)
    if not midis:
        print(f"❌ 没有匹配到任何 MIDI 文件: {args.input}")
        return 1
    print(f"共找到 {len(midis)} 个 MIDI 文件")

    if args.no_valid or args.valid_fraction <= 0:
        ok = midis_to_parquet(midis, str(Path(args.output_dir) / "train.parquet"))
        return 0 if ok else 1

    n_valid = max(1, round(len(midis) * args.valid_fraction))
    train, valid = midis[: len(midis) - n_valid], midis[-n_valid:]
    print(f"划分: train {len(train)} 首 / valid {len(valid)} 首")
    ok1 = midis_to_parquet(train, str(Path(args.output_dir) / "train.parquet"))
    ok2 = midis_to_parquet(valid, str(Path(args.output_dir) / "valid.parquet"))
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
