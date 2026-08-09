"""
AI 音乐数据提取 — 顶层编排入口

串联现有 all-in-one-ai-midi-pipeline 与新增的歌词提取、词-音对齐模块，
输出标准化的训练数据文件。

@implSpec
    上游调用方：命令行 / 批量处理调度
    下游依赖：
        - steps.separate / transcribe_melodic / transcribe_drums 等（现有MIDI pipeline）
        - steps.extract_lyrics（Whisper 歌词提取）
        - steps.align_lyric_midi（词-音对齐）
        - steps.aggregate_dataset（数据汇总与质检）
    在架构图中的位置：Controller 层 — 负责整体流程编排与输出组织

处理流程：
    阶段一：环境检查
    阶段二：音源分离（Demucs）
    阶段三：MIDI 转录（Basic Pitch + ADTOF）
    阶段四：歌词提取（Whisper）
    阶段五：词-音对齐
    阶段六：质量检查与报告生成
    阶段七：批量汇总（如为批量模式）

@author AI Assistant
"""

import argparse
import glob
import gc
import json
import os
import shutil
import sys

import pretty_midi
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steps.separate import separate_track
from steps.beats_meter import estimate_tempo_downbeats_meter
from steps.transcribe_melodic import transcribe_pitched_tracks
from steps.transcribe_drums import transcribe_drums_to_midi
from steps.assign_parts import assign_seven_classes
from steps.key_normalize import detect_and_normalize_key
from steps.meter_apply import insert_time_signatures
from steps.clean_quantize import gentle_cleanup
from steps.write_midi import assemble_and_write_midi
from utils.manifest import load_config, read_manifest, write_manifest, song_id_from_path

from steps.extract_lyrics import (
    extract_lyrics_with_whisper,
    save_lyrics_outputs,
    count_total_words,
)
from steps.align_lyric_midi import align_lyrics_to_midi, is_quality_passed
from steps.aggregate_dataset import (
    generate_song_summary,
    aggregate_dataset,
    generate_batch_report,
)
from steps.correct_lyrics import (
    correct_lyrics_with_original,
    find_original_lyrics_file,
)


# === 常量定义 ===
CFG_PATH = "config.yaml"
DEFAULT_OUTPUT_ROOT = "../output"
DEFAULT_LANGUAGE = "ja"
DEFAULT_WHISPER_MODEL = "large-v3"  # GPU 环境使用 large-v3，日语识别准确率最高
ALIGNMENT_TOLERANCE_SEC = 0.15


def _cleanup_memory():
    """
    强制释放内存：gc 回收 + PyTorch/TensorFlow GPU 缓存清理。
    在阶段切换时调用，降低并行处理时的显存峰值。
    """
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass


def process_single_song(
    audio_path: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    language: str = DEFAULT_LANGUAGE,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    normalize_key: bool = False,
    no_clean: bool = False,
    skip_drums: bool = False,
    tracks: list = None,
    skip_to_stage: int = 0,
) -> dict:
    """
    处理单首歌曲：音源分离 → MIDI转录 → 歌词提取 → 词音对齐 → 质检报告。

    @param audio_path: 输入音频文件路径
    @param output_root: 输出根目录
    @param language: 歌词语言（ja/en/zh）
    @param whisper_model: Whisper 模型大小
    @param normalize_key: 是否进行调性归一化
    @param no_clean: 是否跳过MIDI清理
    @param tracks: 指定输出音轨列表
    @return: 处理结果字典，包含所有输出文件路径与统计信息
    @throws ValueError: 输入文件不存在时抛出
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"音频文件不存在: {audio_path}")

    cfg = load_config(CFG_PATH)
    cfg["no_clean"] = no_clean
    if tracks:
        cfg["tracks"] = tracks

    song_id = song_id_from_path(audio_path)
    song_output_dir = os.path.join(output_root, song_id)
    os.makedirs(song_output_dir, exist_ok=True)

    manifest_path = os.path.join("manifests", f"{song_id}.json")
    manifest = read_manifest(manifest_path)
    manifest.setdefault("song_id", song_id)
    manifest.setdefault("source_audio", audio_path)
    manifest.setdefault("pipeline_flags", {})
    manifest["pipeline_flags"].update(
        {
            "normalize_key": normalize_key,
            "no_clean": no_clean,
            "language": language,
            "whisper_model": whisper_model,
        }
    )

    print(f"\n{'='*60}")
    print(f"  正在处理: {song_id}")
    print(f"{'='*60}")

    # === 快速通道：跳过已完成的阶段 ===
    if skip_to_stage >= 4:
        stems = _reconstruct_stems_from_manifest(manifest)
        midi_output_path = os.path.join(song_output_dir, f"{song_id}_merged.mid")
        meter_info = manifest.get("meter_key", {"tempo": 120.0})
        cleaned = _load_existing_midi_tracks(midi_output_path)

        print(f"\n⏭️  跳过阶段二/三（音源分离+MIDI转录），直接进入阶段四")
        print(f"  已复用音轨：{', '.join(stems.keys())}")
        print(f"  已复用MIDI：{midi_output_path}")
    else:
        # === 阶段二：音源分离 ===
        print("\n[阶段二/六] 音源分离中...")
        stems = separate_track(audio_path, cfg, manifest)
        write_manifest(manifest_path, manifest)
        print("✅ 音源分离完成，已生成音轨：" + ", ".join(stems.keys()))
        _cleanup_memory()  # 释放 Demucs 模型内存

        # === 阶段三：MIDI 转录 ===
        print("\n[阶段三/六] MIDI 转录中...")
        meter_info = estimate_tempo_downbeats_meter(stems, cfg, manifest)
        pitched = transcribe_pitched_tracks(stems, cfg, manifest)

        if skip_drums:
            print("  ⏭️  跳过鼓组转录 (--skip-drums)")
            drums = _empty_drum_track()
        else:
            drums = transcribe_drums_to_midi(stems.get("drums"), cfg, manifest)

        assigned = assign_seven_classes(pitched, drums, stems, cfg, manifest)

        if normalize_key:
            assigned = detect_and_normalize_key(assigned, cfg, manifest)
        else:
            key_info = manifest.setdefault("key", {})
            key_info["normalized"] = False
            key_info["transpose_semitones"] = 0
            key_info["target"] = None
            key_info["reason"] = "key normalization disabled"

        with_meter = insert_time_signatures(assigned, meter_info, cfg, manifest)

        if no_clean:
            cleaned = with_meter
            manifest.setdefault("cleanup", {})["enabled"] = False
        else:
            cleaned = gentle_cleanup(with_meter, cfg, manifest)

        midi_output_path = os.path.join(song_output_dir, f"{song_id}_merged.mid")
        assemble_and_write_midi(cleaned, meter_info, midi_output_path, cfg, manifest)
        write_manifest(manifest_path, manifest)
        print("✅ MIDI 转录完成，多轨MIDI已生成")
        _cleanup_memory()  # 释放 Basic Pitch 模型内存

    # === 阶段四：歌词提取 ===
    print("\n[阶段四/六] 歌词提取中 (Whisper)...")
    vocals_path = stems.get("vocals")
    if vocals_path and os.path.exists(vocals_path):
        lyrics_structured = extract_lyrics_with_whisper(
            vocals_path,
            song_output_dir,
            language=language,
            model_size=whisper_model,
        )
        lyrics_files = save_lyrics_outputs(
            lyrics_structured, song_output_dir, song_id
        )
        total_word_count = count_total_words(lyrics_structured)
        print(f"✅ 歌词提取完成，共 {total_word_count} 个词")
        _cleanup_memory()  # 释放 Whisper 模型内存
    else:
        print("⚠️  无人声音轨，跳过人声歌词提取")
        lyrics_structured = {"full_text": "", "segments": [], "words": []}
        lyrics_files = {}

    # === 阶段4.5：原始歌词校正（可选） ===
    original_lyrics_path = find_original_lyrics_file(song_id)
    if original_lyrics_path:
        print(f"\n[阶段4.5/六] 原始歌词校正中...")
        print(f"  原始歌词文件: {original_lyrics_path}")
        lyrics_structured = correct_lyrics_with_original(
            lyrics_structured,
            original_lyrics_path,
            language=language,
        )
        # 用校正后的歌词重新保存
        lyrics_files = save_lyrics_outputs(
            lyrics_structured, song_output_dir, song_id
        )
        corrected_count = count_total_words(lyrics_structured)
        print(f"✅ 歌词校正完成，校正后共 {corrected_count} 个词")
    elif lyrics_structured.get("words"):
        print("\n  💡 提示：未找到原始歌词文件，使用 Whisper 识别结果")
        print(f"     将原始歌词保存为 lyrics/{song_id}_original.txt 即可自动校正")

    # === 阶段五：词-音对齐 ===
    print("\n[阶段五/六] 词-音对齐中...")
    vocal_midi_path = _build_vocal_midi(cleaned, meter_info, song_output_dir, song_id)
    if vocal_midi_path and lyrics_structured.get("words"):
        alignment_result = align_lyrics_to_midi(
            lyrics_structured,
            vocal_midi_path,
            song_output_dir,
            song_id,
            tolerance_sec=ALIGNMENT_TOLERANCE_SEC,
        )
        match_rate = alignment_result["stats"]["match_rate"]
        quality = "合格" if is_quality_passed(alignment_result) else "低质量"
        print(f"✅ 词-音对齐完成，匹配率: {match_rate:.2%} ({quality})")
    else:
        print("⚠️  无法进行词-音对齐（缺少人声MIDI或歌词）")
        alignment_result = {
            "song_id": song_id,
            "alignments": [],
            "stats": {
                "total_words": 0,
                "matched_words": 0,
                "rest_words": 0,
                "multi_note_words": 0,
                "match_rate": 0.0,
                "total_notes": 0,
            },
        }

    # === 阶段六：质量检查与报告 ===
    print("\n[阶段六/六] 生成处理报告...")
    summary = generate_song_summary(
        song_id,
        audio_path,
        lyrics_structured,
        alignment_result,
        midi_output_path,
        song_output_dir,
    )
    print(f"✅ 处理报告已生成，质量等级: {summary['quality_label']}")

    return {
        "song_id": song_id,
        "output_dir": song_output_dir,
        "midi_path": midi_output_path,
        "lyrics_files": lyrics_files,
        "alignment_result": alignment_result,
        "summary": summary,
    }


def _empty_drum_track():
    """返回一个空鼓组 Instrument，用于跳过鼓组转录时。"""
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    return inst


def _reconstruct_stems_from_manifest(manifest: dict) -> dict:
    """
    从 manifest 中重建 stems 字典，用于快速通道跳过音源分离阶段。

    @param manifest: 已读取的 manifest 字典
    @return: {name -> wav_path} 格式的 stems 字典
    """
    sep = manifest.get("separation", {})
    stem_map = sep.get("stems", {})
    if not stem_map:
        raise ValueError("Manifest 中缺少 separation.stems 信息，无法重建音轨")

    stems = {}
    for name, rel_path in stem_map.items():
        abs_path = os.path.normpath(os.path.join(os.path.dirname(__file__), rel_path))
        if os.path.exists(abs_path):
            stems[name] = abs_path
        else:
            print(f"  ⚠️ 音轨文件不存在，跳过: {abs_path}")

    return stems


def _load_existing_midi_tracks(midi_path: str) -> dict:
    """
    从已有 MIDI 文件中加载所有轨道，用于快速通道跳过 MIDI 转录阶段。

    @param midi_path: 已生成的 merged MIDI 文件路径
    @return: {track_name -> pretty_midi.Instrument} 格式的轨道字典
    """
    if not os.path.exists(midi_path):
        print(f"  ⚠️ MIDI 文件不存在: {midi_path}")
        return {}

    pm = pretty_midi.PrettyMIDI(midi_path)
    tracks = {}
    for inst in pm.instruments:
        name = inst.name.strip() if inst.name else f"track_{inst.program}"
        tracks[name] = inst
    return tracks


def _build_vocal_midi(
    cleaned_tracks: dict,
    meter_info: dict,
    output_dir: str,
    song_id: str,
) -> str:
    """
    从清理后的轨道中提取人声轨（voxlead + voxbg），合成为单独的人声MIDI文件。

    @param cleaned_tracks: 清理后的音轨字典 {name -> pretty_midi.Instrument}
    @param meter_info: 节拍/速度信息
    @param output_dir: 输出目录
    @param song_id: 歌曲标识
    @return: 人声MIDI文件路径，没有人声轨返回 None
    """
    vocal_tracks = []
    for track_name in ["voxlead", "voxbg"]:
        if track_name in cleaned_tracks and cleaned_tracks[track_name].notes:
            vocal_tracks.append(cleaned_tracks[track_name])

    if not vocal_tracks:
        return None

    tempo = meter_info.get("tempo", 120.0) if meter_info else 120.0
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    for track in vocal_tracks:
        pm.instruments.append(track)

    os.makedirs(output_dir, exist_ok=True)
    vocal_midi_path = os.path.join(output_dir, f"{song_id}_vocals.mid")
    pm.write(vocal_midi_path)
    return vocal_midi_path


def process_batch(
    input_pattern: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    language: str = DEFAULT_LANGUAGE,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    normalize_key: bool = False,
    no_clean: bool = False,
    skip_drums: bool = False,
    tracks: list = None,
    skip_to_stage: int = 0,
    parallel: int = 1,
) -> dict:
    """
    批量处理多首歌曲，并聚合为总数据集。
    支持多进程并行处理（--parallel N）。

    @param input_pattern: 输入文件 glob 模式
    @param output_root: 输出根目录
    @param language: 歌词语言
    @param whisper_model: Whisper 模型大小
    @param normalize_key: 是否调性归一化
    @param no_clean: 是否跳过清理
    @param tracks: 指定输出音轨
    @param skip_to_stage: 跳过到指定阶段
    @param parallel: 并行处理歌曲数
    @return: 批量处理结果 {results, dataset, report}
    """
    files = sorted(glob.glob(input_pattern, recursive=True))
    if not files:
        print(f"未找到匹配的文件: {input_pattern}")
        return {"results": [], "dataset": None, "report": ""}

    print(f"\n共找到 {len(files)} 首歌曲，开始批量处理...")
    if parallel > 1:
        print(f"  并行模式：{parallel} 个 worker 同时处理")

    all_results = []
    failed_songs = []

    if parallel > 1:
        # === 多进程并行处理（实时进度） ===
        # 必须用 spawn 而非 fork：fork 会继承主进程已初始化的 TensorFlow 状态，
        # 子进程预测时死锁（CPU 0% 卡死）。spawn 下每个 worker 干净重启、懒加载模型。
        from multiprocessing import get_context
        pool_ctx = get_context("spawn")

        task_args = [
            (
                audio_path,
                output_root,
                language,
                whisper_model,
                normalize_key,
                no_clean,
                skip_drums,
                tracks,
                skip_to_stage,
            )
            for audio_path in files
        ]

        ok_count = 0
        fail_count = 0
        with pool_ctx.Pool(processes=min(parallel, len(files))) as pool:
            for i, result in enumerate(
                pool.imap(_process_single_song_safe_star, task_args, chunksize=1), 1
            ):
                song_id = song_id_from_path(files[i - 1])
                if isinstance(result, Exception):
                    fail_count += 1
                    print(
                        f"❌ [{i}/{len(files)}] 失败({fail_count}): {song_id} - {result}",
                        flush=True,
                    )
                    failed_songs.append({"song_id": song_id, "error": str(result)})
                else:
                    ok_count += 1
                    all_results.append(result)
                    print(
                        f"✅ [{i}/{len(files)}] 完成({ok_count}): {song_id}",
                        flush=True,
                    )
                # 实时汇总行（覆盖式显示，便于一眼看进度）
                print(
                    f"    └ 进度: {i}/{len(files)}  ✅{ok_count} ❌{fail_count}",
                    flush=True,
                )
    else:
        # === 顺序处理 ===
        for idx, audio_path in enumerate(files, 1):
            song_id = song_id_from_path(audio_path)
            print(f"\n[{idx}/{len(files)}] 处理: {song_id}")
            try:
                result = process_single_song(
                    audio_path,
                    output_root=output_root,
                    language=language,
                    whisper_model=whisper_model,
                    normalize_key=normalize_key,
                    no_clean=no_clean,
                    skip_drums=skip_drums,
                    tracks=tracks,
                    skip_to_stage=skip_to_stage,
                )
                all_results.append(result)
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                failed_songs.append({"song_id": song_id, "error": str(e)})

    print(f"\n{'='*60}")
    print(f"  批量处理完成")
    print(f"  成功: {len(all_results)} / 失败: {len(failed_songs)}")
    print(f"{'='*60}")

    # === 阶段七：数据集聚合 ===
    print("\n[阶段七] 聚合总数据集...")
    dataset = aggregate_dataset(all_results, output_root)
    report = generate_batch_report(dataset, output_root)
    print(report)

    if failed_songs:
        failed_path = os.path.join(output_root, "failed_songs.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_songs, f, ensure_ascii=False, indent=2)
        print(f"\n失败列表已保存至: {failed_path}")

    return {
        "results": all_results,
        "dataset": dataset,
        "report": report,
        "failed": failed_songs,
    }


def _process_single_song_safe_star(args):
    """pool.imap 用的模块级包装：解包参数元组后调用（局部函数不可 pickle）。"""
    return _process_single_song_safe(*args)


def _process_single_song_safe(
    audio_path: str,
    output_root: str,
    language: str,
    whisper_model: str,
    normalize_key: bool,
    no_clean: bool,
    skip_drums: bool,
    tracks: list,
    skip_to_stage: int,
):
    """
    多进程安全包装器：捕获异常并返回，避免子进程崩溃影响主进程。

    @return: 处理结果 dict 或 Exception 对象
    """
    try:
        return process_single_song(
            audio_path,
            output_root=output_root,
            language=language,
            whisper_model=whisper_model,
            normalize_key=normalize_key,
            no_clean=no_clean,
            skip_drums=skip_drums,
            tracks=tracks,
            skip_to_stage=skip_to_stage,
        )
    except Exception as e:
        return e


def main():
    ap = argparse.ArgumentParser(description="AI 音乐数据提取流水线")
    sub = ap.add_subparsers(dest="cmd")

    single = sub.add_parser("single", help="处理单首歌曲")
    single.add_argument("audio_path", help="输入音频文件路径")
    single.add_argument(
        "--output", default=DEFAULT_OUTPUT_ROOT, help="输出根目录"
    )
    single.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="歌词语言 (ja/en/zh)"
    )
    single.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper 模型大小"
    )
    single.add_argument("--normalize-key", action="store_true", help="调性归一化")
    single.add_argument("--no-clean", action="store_true", help="跳过MIDI清理")
    single.add_argument("--skip-drums", action="store_true", help="跳过鼓组转录（CPU上极慢）")
    single.add_argument("--tracks", type=str, default=None, help="输出音轨")
    single.add_argument(
        "--skip-to-stage",
        type=int,
        default=0,
        choices=[4],
        help="跳过到指定阶段（4=直接从歌词提取开始，跳过音源分离和MIDI转录）",
    )

    batch = sub.add_parser("batch", help="批量处理")
    batch.add_argument("pattern", help='输入文件 glob 模式，如 "input/*.mp3"')
    batch.add_argument(
        "--output", default=DEFAULT_OUTPUT_ROOT, help="输出根目录"
    )
    batch.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="歌词语言 (ja/en/zh)"
    )
    batch.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper 模型大小"
    )
    batch.add_argument("--normalize-key", action="store_true", help="调性归一化")
    batch.add_argument("--no-clean", action="store_true", help="跳过MIDI清理")
    batch.add_argument("--skip-drums", action="store_true", help="跳过鼓组转录（CPU上极慢）")
    batch.add_argument("--tracks", type=str, default=None, help="输出音轨")
    batch.add_argument(
        "--skip-to-stage",
        type=int,
        default=0,
        choices=[4],
        help="跳过到指定阶段（4=直接从歌词提取开始，跳过音源分离和MIDI转录）",
    )
    batch.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行处理歌曲数（默认1=顺序处理，GPU 24GB 建议 2-3，CPU 可设 10+）",
    )

    args = ap.parse_args()

    if not args.cmd:
        ap.print_help()
        return 2

    tracks = None
    if hasattr(args, "tracks") and args.tracks:
        tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]

    if args.cmd == "single":
        process_single_song(
            args.audio_path,
            output_root=args.output,
            language=args.language,
            whisper_model=args.whisper_model,
            normalize_key=args.normalize_key,
            no_clean=args.no_clean,
            skip_drums=args.skip_drums,
            tracks=tracks,
            skip_to_stage=args.skip_to_stage,
        )
    elif args.cmd == "batch":
        process_batch(
            args.pattern,
            output_root=args.output,
            language=args.language,
            whisper_model=args.whisper_model,
            normalize_key=args.normalize_key,
            no_clean=args.no_clean,
            skip_drums=args.skip_drums,
            tracks=tracks,
            skip_to_stage=args.skip_to_stage,
            parallel=args.parallel,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C：多进程池会自行清理，这里只做友好提示（不打印一屏 traceback）
        print("\n[中断] 收到 Ctrl+C，已停止处理。已完成歌曲的结果已保存，重跑会从断点继续。", flush=True)
        raise SystemExit(130)
