"""
词-音对齐模块 — 将歌词时间戳与MIDI音符进行时间匹配

@implSpec
    上游调用方：run_extraction.py（顶层编排）
    下游依赖：pretty_midi（MIDI解析）、json（持久化）
    在架构图中的位置：Service 层 — 负责歌词与音符的时间域对齐

对齐策略：
    1. 单词时间区间内只有一个音符 → 直接映射
    2. 单词时间区间内有多个音符 → 记录全部音符 + 平均音高
    3. 单词时间区间内无音符 → 标记为 rest（休止符）

@author AI Assistant
"""

import json
import os
import pretty_midi


# === 常量定义 ===
DEFAULT_ALIGN_TOLERANCE_SEC = 0.15   # 对齐容差：单位 秒
MIN_NOTE_DURATION_SEC = 0.01         # 最小音符时长：单位 秒
REST_NOTE_PITCH = -1                 # 休止符标记音高


def align_lyrics_to_midi(
    lyrics_structured: dict,
    vocals_midi_path: str,
    song_output_dir: str,
    song_id: str,
    tolerance_sec: float = DEFAULT_ALIGN_TOLERANCE_SEC,
) -> dict:
    """
    将歌词与MIDI人声轨进行时间对齐。

    @param lyrics_structured: 结构化歌词数据（来自 extract_lyrics.py）
    @param vocals_midi_path: 人声轨 MIDI 文件路径
    @param song_output_dir: 歌曲输出目录
    @param song_id: 歌曲标识
    @param tolerance_sec: 时间对齐容差（秒），允许歌词起始与音符起始的偏差
    @return: 对齐结果字典，包含 alignments 列表和统计信息
    @throws FileNotFoundError: 当 MIDI 文件不存在时抛出
    """
    if not os.path.exists(vocals_midi_path):
        raise FileNotFoundError(f"人声MIDI文件不存在: {vocals_midi_path}")

    notes = _load_vocal_notes(vocals_midi_path)
    words = lyrics_structured.get("words", [])

    alignments = []
    matched_count = 0
    rest_count = 0
    multi_note_count = 0

    # === 1. 逐个词对齐 ===
    for word_info in words:
        word_start = word_info["start"]
        word_end = word_info["end"]
        word_text = word_info["word"]

        matching_notes = _find_notes_in_interval(
            notes, word_start, word_end, tolerance_sec
        )

        alignment_entry = _build_alignment_entry(
            word_text, word_start, word_end, matching_notes
        )
        alignments.append(alignment_entry)

        note_count = len(alignment_entry["notes"])
        if note_count == 0:
            rest_count += 1
        elif note_count == 1:
            matched_count += 1
        else:
            matched_count += 1
            multi_note_count += 1

    # === 2. 计算统计指标 ===
    total_words = len(words) if words else 1
    match_rate = round(matched_count / total_words, 4)

    alignment_result = {
        "song_id": song_id,
        "alignments": alignments,
        "stats": {
            "total_words": total_words,
            "matched_words": matched_count,
            "rest_words": rest_count,
            "multi_note_words": multi_note_count,
            "match_rate": match_rate,
            "total_notes": len(notes),
        },
    }

    # === 3. 保存对齐文件 ===
    alignment_path = os.path.join(song_output_dir, f"{song_id}_alignment.json")
    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(alignment_result, f, ensure_ascii=False, indent=2)

    return alignment_result


def _load_vocal_notes(midi_path: str) -> list:
    """
    从MIDI文件加载所有人声音符，按起始时间排序。

    @param midi_path: MIDI 文件路径
    @return: 音符列表，每项为 {pitch, start, end, velocity}
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes = []

    for instrument in pm.instruments:
        for note in instrument.notes:
            if note.end - note.start < MIN_NOTE_DURATION_SEC:
                continue
            all_notes.append(
                {
                    "pitch": note.pitch,
                    "start": round(note.start, 3),
                    "end": round(note.end, 3),
                    "velocity": note.velocity,
                }
            )

    all_notes.sort(key=lambda n: n["start"])
    return all_notes


def _find_notes_in_interval(
    notes: list,
    word_start: float,
    word_end: float,
    tolerance: float,
) -> list:
    """
    查找时间区间内的所有音符（含容差扩展）。

    @param notes: 音符列表（已按 start 排序）
    @param word_start: 歌词起始时间
    @param word_end: 歌词结束时间
    @param tolerance: 时间容差
    @return: 匹配的音符列表
    """
    expanded_start = word_start - tolerance
    expanded_end = word_end + tolerance

    matching = []
    for note in notes:
        if note["start"] > expanded_end:
            break
        if note["end"] < expanded_start:
            continue

        overlap_start = max(note["start"], word_start)
        overlap_end = min(note["end"], word_end)
        overlap_duration = max(0.0, overlap_end - overlap_start)

        if overlap_duration > 0 or note["start"] <= word_end + tolerance:
            note_with_overlap = dict(note)
            note_with_overlap["overlap_ratio"] = round(
                overlap_duration / max(word_end - word_start, 0.001), 4
            )
            matching.append(note_with_overlap)

    return matching


def _build_alignment_entry(
    word: str,
    start_time: float,
    end_time: float,
    matching_notes: list,
) -> dict:
    """
    构建单个词的对齐条目。

    @param word: 歌词文本
    @param start_time: 起始时间
    @param end_time: 结束时间
    @param matching_notes: 匹配的音符列表
    @return: 对齐条目字典
    """
    pitches = [n["pitch"] for n in matching_notes]

    if not pitches:
        return {
            "lyric": word,
            "start_time": round(start_time, 3),
            "end_time": round(end_time, 3),
            "notes": [],
            "average_pitch": REST_NOTE_PITCH,
            "is_rest": True,
        }

    avg_pitch = round(sum(pitches) / len(pitches), 1)

    return {
        "lyric": word,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "notes": pitches,
        "average_pitch": avg_pitch,
        "is_rest": False,
        "note_details": [
            {
                "pitch": n["pitch"],
                "start": n["start"],
                "end": n["end"],
                "velocity": n["velocity"],
            }
            for n in matching_notes
        ],
    }


def is_quality_passed(alignment_result: dict, threshold: float = 0.60) -> bool:
    """
    判断对齐质量是否合格（匹配率 > 阈值）。

    @param alignment_result: 对齐结果字典
    @param threshold: 合格阈值（默认 60%）
    @return: 是否合格
    """
    return alignment_result["stats"]["match_rate"] > threshold
