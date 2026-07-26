"""
歌词提取模块 — 使用 faster-whisper 从人声轨提取带时间戳的歌词

@implSpec
    上游调用方：run_extraction.py（顶层编排）
    下游依赖：faster_whisper（语音识别，走 HuggingFace Hub，支持 HF_ENDPOINT 镜像）
    在架构图中的位置：Service 层 — 负责从音频到结构化歌词的转换

@author AI Assistant
"""

import json
import os


# === 常量定义 ===
DEFAULT_MODEL_SIZE = "large-v3"  # GPU 环境使用 large-v3 模型，日语识别准确率最高
DEFAULT_LANGUAGE = "ja"
DEFAULT_COMPUTE_TYPE = "float16"  # GPU 推理：半精度，速度与精度最佳平衡
DEFAULT_BEAM_SIZE = 5  # 恢复全精度 beam search，GPU 上几乎无性能损失
DEFAULT_CPU_THREADS = 0  # 0 = 自动检测 CPU 核心数
DEFAULT_DOWNLOAD_ROOT = "/root/.cache/huggingface"  # Linux 模型缓存目录
MIN_WORD_DURATION_SEC = 0.05


def extract_lyrics_with_whisper(
    vocals_path: str,
    output_dir: str,
    language: str = DEFAULT_LANGUAGE,
    model_size: str = DEFAULT_MODEL_SIZE,
) -> dict:
    """
    使用 faster-whisper 从人声 WAV 提取带时间戳的歌词。
    模型通过 HuggingFace Hub 下载，支持 HF_ENDPOINT 镜像。

    @param vocals_path: 人声轨 WAV 文件绝对路径
    @param output_dir: 输出目录，用于存放 Whisper 原始结果
    @param language: 语言代码（ja/en/zh 等）
    @param model_size: 模型大小（tiny/base/small/medium/large-v3 等）
    @return: 结构化字典，包含 full_text、segments、words 三级时间戳
    @throws RuntimeError: 当转录失败时抛出
    """
    from faster_whisper import WhisperModel

    os.makedirs(output_dir, exist_ok=True)

    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
    cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", DEFAULT_CPU_THREADS))

    # === 自动检测 GPU ===
    device = "cuda" if _has_gpu() else "cpu"
    if device == "cpu":
        compute_type = "int8"  # CPU 降级为 int8

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
        download_root=DEFAULT_DOWNLOAD_ROOT,
    )

    segments_iter, info = model.transcribe(
        vocals_path,
        language=language,
        beam_size=DEFAULT_BEAM_SIZE,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,  # 关闭上下文依赖，加速推理
        no_speech_threshold=0.6,  # 提高无声阈值，跳过非人声段
        compression_ratio_threshold=2.4,  # 过滤压缩比异常的片段
        log_prob_threshold=-1.0,  # 过滤低置信度片段
    )

    # === 收集所有片段 ===
    raw_segments = []
    for seg in segments_iter:
        raw_segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4),
                    }
                    for w in (seg.words or [])
                ],
            }
        )

    # === 保存原始结果 ===
    raw_output_path = os.path.join(output_dir, "whisper_raw.json")
    raw_result = {
        "text": " ".join(s["text"] for s in raw_segments),
        "segments": raw_segments,
        "language": info.language,
        "duration": info.duration,
    }
    with open(raw_output_path, "w", encoding="utf-8") as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)

    return _structure_whisper_result(raw_result)


def _structure_whisper_result(raw_result: dict) -> dict:
    """
    将 Whisper 原始输出整理为标准化结构。

    @param raw_result: Whisper transcribe 返回的原始字典
    @return: 标准化结构 {full_text, segments, words}
    """
    segments = []
    all_words = []

    for seg in raw_result.get("segments", []):
        segment_info = {
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "text": seg["text"].strip(),
        }

        words_in_seg = []
        for w in seg.get("words", []):
            word_info = {
                "word": w["word"].strip(),
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "probability": round(float(w.get("probability", 0.0)), 4),
            }
            words_in_seg.append(word_info)
            all_words.append(word_info)

        segment_info["words"] = words_in_seg
        segments.append(segment_info)

    return {
        "full_text": raw_result.get("text", "").strip(),
        "segments": segments,
        "words": all_words,
    }


def save_lyrics_outputs(structured: dict, song_output_dir: str, song_id: str) -> dict:
    """
    保存歌词相关的所有输出文件。

    @param structured: _structure_whisper_result 返回的结构化数据
    @param song_output_dir: 歌曲输出目录
    @param song_id: 歌曲标识
    @return: 各输出文件路径字典
    """
    os.makedirs(song_output_dir, exist_ok=True)

    # === 1. 纯歌词文本 ===
    lyrics_txt_path = os.path.join(song_output_dir, f"{song_id}_lyrics.txt")
    with open(lyrics_txt_path, "w", encoding="utf-8") as f:
        for seg in structured["segments"]:
            f.write(seg["text"] + "\n")

    # === 2. 带时间戳的歌词 JSON ===
    timed_json_path = os.path.join(song_output_dir, f"{song_id}_lyrics_timed.json")
    timed_data = {
        "song_id": song_id,
        "full_text": structured["full_text"],
        "segments": structured["segments"],
    }
    with open(timed_json_path, "w", encoding="utf-8") as f:
        json.dump(timed_data, f, ensure_ascii=False, indent=2)

    return {
        "lyrics_txt": lyrics_txt_path,
        "lyrics_timed_json": timed_json_path,
    }


def count_total_words(structured: dict) -> int:
    """
    统计歌词总字数（词数）。

    @param structured: 结构化歌词数据
    @return: 总词数
    """
    return len(structured.get("words", []))


def _has_gpu() -> bool:
    """
    检测是否有可用的 CUDA GPU。

    @return: 是否有 GPU 可用
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False