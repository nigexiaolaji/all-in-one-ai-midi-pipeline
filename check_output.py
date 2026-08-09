#!/usr/bin/env python3
"""
快速体检 pipeline 输出：打印每首歌的处理摘要与多轨 MIDI 结构。

用法:
    python3 check_output.py [输出目录]   # 默认 ../output（仓库同级的 output/）

输出示例:
    【本家】即死亡 - 初音ミク    时长 145.6s  音符 1394  词 370  匹配率 100.00% 合格
    歌名                    | tempo=117.5 | [('voxlead', 1049), ('voxbg', 204), ...]
"""

import glob
import json
import os
import sys


def main() -> int:
    out_root = sys.argv[1] if len(sys.argv) > 1 else "../output"
    out_root = os.path.abspath(out_root)
    if not os.path.isdir(out_root):
        print(f"❌ 输出目录不存在: {out_root}")
        return 1

    # --- 1) 每首歌的处理摘要 ---
    summaries = sorted(glob.glob(os.path.join(out_root, "*", "*_summary.json")))
    if not summaries:
        print(f"⚠️ 未找到 *_summary.json（{out_root} 下没有已处理的歌曲）")
    for f in summaries:
        s = json.load(open(f, encoding="utf-8"))
        sid = str(s.get("song_id", "?"))[:20]
        print(
            f"{sid:<22} 时长{s.get('duration_sec', 0):>6.1f}s "
            f"音符{s.get('total_notes', 0):>5} 词{s.get('total_lyric_words', 0):>4} "
            f"匹配率{s.get('alignment_match_rate', 0) * 100:>6.2f}% "
            f"{s.get('quality_label', '?')}"
        )

    # --- 2) 每首歌多轨 MIDI 的轨道结构 ---
    midis = sorted(glob.glob(os.path.join(out_root, "*", "*_merged.mid")))
    for mid in midis:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(mid)
        info = [(i.name, len(i.notes)) for i in pm.instruments if len(i.notes)]
        name = os.path.basename(os.path.dirname(mid))[:22]
        print(f"{name:<22} | tempo={pm.estimate_tempo():.1f} | {info}")

    print(f"\n共 {len(summaries)} 首歌，{len(midis)} 个多轨 MIDI。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
