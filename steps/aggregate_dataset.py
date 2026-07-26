"""
数据汇总与质检模块 — 生成处理报告、统计指标、批量数据集聚合

@implSpec
    上游调用方：run_extraction.py（顶层编排）
    下游依赖：json（持久化）、os（文件操作）
    在架构图中的位置：Service 层 — 负责结果汇总、质量评估、数据集构建

质检标准：
    - 对齐匹配率 > 60% 为合格
    - 匹配率 < 60% 标记为"低质量"

@author AI Assistant
"""

import json
import os


# === 常量定义 ===
QUALITY_THRESHOLD = 0.60           # 对齐匹配率合格阈值
QUALITY_LABEL_PASS = "合格"
QUALITY_LABEL_LOW = "低质量"


def generate_song_summary(
    song_id: str,
    source_audio_path: str,
    lyrics_structured: dict,
    alignment_result: dict,
    midi_path: str,
    song_output_dir: str,
) -> dict:
    """
    生成单首歌曲的处理报告。

    @param song_id: 歌曲标识
    @param source_audio_path: 原始音频文件路径
    @param lyrics_structured: 结构化歌词数据
    @param alignment_result: 词-音对齐结果
    @param midi_path: 多轨MIDI文件路径
    @param song_output_dir: 歌曲输出目录
    @return: 报告数据字典
    """
    file_size_mb = round(os.path.getsize(source_audio_path) / (1024 * 1024), 2)

    total_words = alignment_result["stats"]["total_words"]
    match_rate = alignment_result["stats"]["match_rate"]
    quality_label = (
        QUALITY_LABEL_PASS
        if match_rate > QUALITY_THRESHOLD
        else QUALITY_LABEL_LOW
    )

    duration_sec = _estimate_duration(lyrics_structured)

    summary = {
        "song_id": song_id,
        "source_file": source_audio_path,
        "file_size_mb": file_size_mb,
        "duration_sec": round(duration_sec, 2),
        "total_notes": alignment_result["stats"]["total_notes"],
        "total_lyric_words": total_words,
        "matched_words": alignment_result["stats"]["matched_words"],
        "rest_words": alignment_result["stats"]["rest_words"],
        "multi_note_words": alignment_result["stats"]["multi_note_words"],
        "alignment_match_rate": match_rate,
        "quality_label": quality_label,
        "is_passed": match_rate > QUALITY_THRESHOLD,
    }

    summary_path = os.path.join(song_output_dir, f"{song_id}_summary.txt")
    _write_summary_text(summary_path, summary)

    return summary


def _estimate_duration(lyrics_structured: dict) -> float:
    """
    根据歌词最后一个词的结束时间估算歌曲时长。

    @param lyrics_structured: 结构化歌词数据
    @return: 估算时长（秒）
    """
    words = lyrics_structured.get("words", [])
    if not words:
        segments = lyrics_structured.get("segments", [])
        if segments:
            return segments[-1]["end"]
        return 0.0
    return words[-1]["end"]


def _write_summary_text(path: str, summary: dict) -> None:
    """
    将处理报告写入文本文件。

    @param path: 输出文件路径
    @param summary: 报告数据字典
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  歌曲处理报告: {summary['song_id']}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【基本信息】\n")
        f.write(f"  源文件:       {summary['source_file']}\n")
        f.write(f"  文件大小:     {summary['file_size_mb']} MB\n")
        f.write(f"  估算时长:     {summary['duration_sec']} 秒\n\n")

        f.write("【MIDI提取统计】\n")
        f.write(f"  音符总数:     {summary['total_notes']}\n\n")

        f.write("【歌词提取统计】\n")
        f.write(f"  歌词总词数:   {summary['total_lyric_words']}\n\n")

        f.write("【词-音对齐统计】\n")
        f.write(f"  匹配词数:     {summary['matched_words']}\n")
        f.write(f"  休止符词数:   {summary['rest_words']}\n")
        f.write(f"  多音符词数:   {summary['multi_note_words']}\n")
        f.write(f"  对齐匹配率:   {summary['alignment_match_rate']:.2%}\n\n")

        f.write("【质量评估】\n")
        f.write(f"  质量等级:     {summary['quality_label']}\n")
        f.write(f"  是否合格:     {'是' if summary['is_passed'] else '否'}\n\n")

        if not summary["is_passed"]:
            f.write("⚠️  注意：对齐匹配率低于60%，建议人工检查修正\n")


def aggregate_dataset(all_song_results: list, output_dir: str) -> dict:
    """
    将所有歌曲的对齐数据聚合成一个总数据集。

    @param all_song_results: 每首歌的结果列表，每项包含 song_id 和对齐信息
    @param output_dir: 输出目录
    @return: 聚合后的数据集字典
    """
    os.makedirs(output_dir, exist_ok=True)

    all_alignments = []
    total_songs = len(all_song_results)
    passed_songs = 0
    total_match_rates = 0.0
    total_words = 0
    total_notes = 0

    # === 1. 汇总每首歌的数据 ===
    for song_result in all_song_results:
        alignment = song_result.get("alignment_result", {})
        all_alignments.append(
            {
                "song_id": song_result["song_id"],
                "alignments": alignment.get("alignments", []),
                "stats": alignment.get("stats", {}),
            }
        )

        stats = alignment.get("stats", {})
        total_match_rates += stats.get("match_rate", 0.0)
        total_words += stats.get("total_words", 0)
        total_notes += stats.get("total_notes", 0)

        if stats.get("match_rate", 0) > QUALITY_THRESHOLD:
            passed_songs += 1

    # === 2. 计算整体统计 ===
    avg_match_rate = (
        round(total_match_rates / total_songs, 4) if total_songs > 0 else 0.0
    )

    dataset = {
        "total_songs": total_songs,
        "passed_songs": passed_songs,
        "failed_songs": total_songs - passed_songs,
        "average_match_rate": avg_match_rate,
        "total_words_across_songs": total_words,
        "total_notes_across_songs": total_notes,
        "quality_threshold": QUALITY_THRESHOLD,
        "songs": all_alignments,
    }

    # === 3. 写入数据集文件 ===
    dataset_path = os.path.join(output_dir, "dataset.json")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return dataset


def generate_batch_report(dataset: dict, output_dir: str) -> str:
    """
    生成批量处理统计报告。

    @param dataset: 聚合数据集
    @param output_dir: 输出目录
    @return: 报告文本
    """
    report_lines = [
        "=" * 60,
        "  批量处理统计报告",
        "=" * 60,
        "",
        f"  处理歌曲总数:    {dataset['total_songs']}",
        f"  合格歌曲数:      {dataset['passed_songs']}",
        f"  低质量歌曲数:    {dataset['failed_songs']}",
        f"  平均对齐匹配率:  {dataset['average_match_rate']:.2%}",
        f"  总词数:          {dataset['total_words_across_songs']}",
        f"  总音符数:        {dataset['total_notes_across_songs']}",
        f"  质量阈值:        {dataset['quality_threshold']:.0%}",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "batch_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
