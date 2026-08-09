#!/usr/bin/env python3
"""
快速体检 pipeline 输出：打印每首歌的处理摘要与多轨 MIDI 结构。

用法:
    python3 check_output.py [输出目录]   # 默认 ../output（仓库同级的 output/）

说明:
    - summary 文件实际是 {song_id}_summary.txt（文本格式），见 steps/aggregate_dataset.py
    - tempo 读自 MIDI 元数据（写入值），不用 estimate_tempo()（按音符间隔估算会翻倍）

输出示例:
    【本家】即死亡 - 初音ミク    时长 145.6s  音符 1394  词 370  匹配率 100.00% 合格
    歌名                    | tempo=117.5 | [('voxlead', 1049), ('voxbg', 204), ...]
"""

import glob
import os
import re
import sys


def _parse_summary_txt(path: str) -> dict:
    """解析 {song_id}_summary.txt 文本格式，返回关键字段字典。"""
    data = {"song_id": os.path.basename(os.path.dirname(path))}
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def grab(pattern: str, cast=str):
        m = re.search(pattern, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except (ValueError, TypeError):
            return m.group(1)

    dur = grab(r"估算时长:\s+([\d.]+)", float)
    notes = grab(r"音符总数:\s+(\d+)", int)
    words = grab(r"歌词总词数:\s+(\d+)", int)
    rate = grab(r"对齐匹配率:\s+([\d.]+)%", float)
    label = grab(r"质量等级:\s+(\S+)")

    if rate is not None:
        rate = rate / 100.0  # 文本里是百分数
    data.update(
        {
            "duration_sec": dur or 0.0,
            "total_notes": notes or 0,
            "total_lyric_words": words or 0,
            "alignment_match_rate": rate or 0.0,
            "quality_label": label or "?",
        }
    )
    return data


def _midi_tempo(pm) -> float:
    """读取 MIDI 元数据中写入的 tempo（取 0 时刻值），兜底 estimate_tempo。"""
    try:
        times, tempos = pm.get_tempo_changes()
        if len(tempos):
            return float(tempos[0])
    except Exception:
        pass
    try:
        return float(pm.initial_tempo)
    except Exception:
        pass
    return pm.estimate_tempo()


def main() -> int:
    out_root = sys.argv[1] if len(sys.argv) > 1 else "../output"
    out_root = os.path.abspath(out_root)
    if not os.path.isdir(out_root):
        print(f"❌ 输出目录不存在: {out_root}")
        return 1

    # --- 1) 每首歌的处理摘要（文本格式 _summary.txt） ---
    summaries = sorted(glob.glob(os.path.join(out_root, "*", "*_summary.txt")))
    if not summaries:
        print(f"⚠️ 未找到 *_summary.txt（{out_root} 下没有已处理的歌曲）")
    for f in summaries:
        s = _parse_summary_txt(f)
        sid = str(s.get("song_id", "?"))[:20]
        print(
            f"{sid:<22} 时长{s['duration_sec']:>6.1f}s "
            f"音符{s['total_notes']:>5} 词{s['total_lyric_words']:>4} "
            f"匹配率{s['alignment_match_rate'] * 100:>6.2f}% "
            f"{s['quality_label']}"
        )

    # --- 2) 每首歌多轨 MIDI 的轨道结构 ---
    midis = sorted(glob.glob(os.path.join(out_root, "*", "*_merged.mid")))
    for mid in midis:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(mid)
        info = [(i.name, len(i.notes)) for i in pm.instruments if len(i.notes)]
        name = os.path.basename(os.path.dirname(mid))[:22]
        print(f"{name:<22} | tempo={_midi_tempo(pm):.1f} | {info}")

    print(f"\n共 {len(summaries)} 首处理报告，{len(midis)} 个多轨 MIDI。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
