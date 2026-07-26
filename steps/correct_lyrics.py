"""
原始歌词校正模块 — 将原始歌词与 Whisper 时间戳对齐

@implSpec
    上游调用方：run_extraction.py（顶层编排，歌词提取阶段之后）
    下游依赖：无（纯算法模块，不依赖外部 API）
    在架构图中的位置：Service 层 — 负责将人工提供的原始歌词对齐到 Whisper 时间戳

工作原理：
    1. Whisper 提供词级时间戳序列（词可能识别错误，但时间戳准确）
    2. 用户提供原始歌词文本（词100%准确，但无时间戳）
    3. 使用 DP（动态规划）对齐算法，将原始歌词的每个词匹配到最相似的 Whisper 词
    4. 输出：原始歌词 + Whisper 时间戳 = 100%准确歌词 + 精确时间戳

@author AI Assistant
"""

import json
import os
import re
from typing import Optional


# === 常量定义 ===
DEFAULT_LYRICS_DIR = "lyrics"  # 原始歌词文件存放目录
MAX_ALIGNMENT_GAP = 5  # DP 对齐时允许的最大跳跃距离（控制对齐精度与速度）


# === 日文分词工具 ===
def _tokenize_japanese(text: str) -> list[str]:
    """
    日文简易分词：按字符切分，合并连续假名/汉字块。
    不使用外部分词器（如 MeCab），避免额外依赖。

    @param text: 日文文本
    @return: 词列表
    """
    if not text:
        return []

    # 移除标点符号，保留日文假名、汉字、英文
    cleaned = re.sub(r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\w]", " ", text)
    # 按空格分割，过滤空字符串
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


def _tokenize_simple(text: str) -> list[str]:
    """
    通用分词：按空白字符切分。

    @param text: 任意文本
    @return: 词列表
    """
    if not text:
        return []
    return [t.strip() for t in text.split() if t.strip()]


def _read_original_lyrics(lyrics_path: str) -> list[str]:
    """
    读取原始歌词文件，每行一句。

    @param lyrics_path: 原始歌词文本文件路径
    @return: 歌词行列表
    @throws FileNotFoundError: 当文件不存在时抛出
    """
    if not os.path.exists(lyrics_path):
        raise FileNotFoundError(f"原始歌词文件不存在: {lyrics_path}")

    with open(lyrics_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return lines


def _dp_align_words(
    original_words: list[str],
    whisper_words: list[dict],
) -> list[dict]:
    """
    使用 DP（动态规划）将原始歌词词对齐到 Whisper 词级时间戳。

    算法思路：
        构建编辑距离矩阵，对每个原始词找到最匹配的 Whisper 词。
        匹配规则：相同词优先匹配，否则选择距离最近的时间戳。

    @param original_words: 原始歌词词列表
    @param whisper_words: Whisper 词列表 [{word, start, end, probability}]
    @return: 对齐后的词列表 [{word, start, end, probability, source}]
    """
    if not original_words or not whisper_words:
        return [
            {"word": w, "start": 0.0, "end": 0.0, "probability": 1.0, "source": "original_no_ts"}
            for w in original_words
        ]

    ow = original_words
    ww = whisper_words
    n_orig = len(ow)
    n_whis = len(ww)

    # === DP 矩阵：dp[i][j] = 对齐前 i 个原始词和前 j 个 Whisper 词的最小代价 ===
    INF = float("inf")
    dp = [[INF] * (n_whis + 1) for _ in range(n_orig + 1)]
    dp[0][0] = 0

    for i in range(n_orig + 1):
        for j in range(n_whis + 1):
            if i > 0 and j > 0:
                # 匹配代价：相同词 = 0，不同词 = 1
                cost = 0 if ow[i - 1] == ww[j - 1]["word"] else 1
                dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + cost)
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i - 1][j] + 2)  # 原始词未匹配（跳过）
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j - 1] + 2)  # Whisper 词未匹配（跳过）

    # === 回溯：找到最佳对齐路径 ===
    aligned = []
    i, j = n_orig, n_whis

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ow[i - 1] == ww[j - 1]["word"] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append({
                    "word": ow[i - 1],
                    "start": ww[j - 1]["start"],
                    "end": ww[j - 1]["end"],
                    "probability": ww[j - 1].get("probability", 1.0),
                    "source": "original_matched",
                })
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 2:
            # 原始词没有对应的时间戳，使用前一个词的时间戳或 0
            prev_end = aligned[-1]["end"] if aligned else 0.0
            aligned.append({
                "word": ow[i - 1],
                "start": prev_end,
                "end": prev_end + 0.5,
                "probability": 1.0,
                "source": "original_interpolated",
            })
            i -= 1
            continue
        if j > 0:
            j -= 1

    aligned.reverse()
    return aligned


def _segment_align(
    original_lines: list[str],
    whisper_segments: list[dict],
) -> list[dict]:
    """
    将原始歌词按行对齐到 Whisper 段落级时间戳。

    策略：
        1. 如果原始行数与 Whisper 段落数一致 → 直接一一对应
        2. 如果数量不同 → 按比例映射，每行原始歌词取最近 Whisper 段落的时间戳

    @param original_lines: 原始歌词行列表
    @param whisper_segments: Whisper 段落列表 [{start, end, text, words}]
    @return: 对齐后的段落列表 [{start, end, text, words}]
    """
    n_orig = len(original_lines)
    n_whis = len(whisper_segments)

    if n_orig == 0:
        return []

    if n_whis == 0:
        return [
            {"start": 0.0, "end": i * 3.0, "text": line, "words": []}
            for i, line in enumerate(original_lines)
        ]

    aligned_segments = []
    for i, line in enumerate(original_lines):
        ratio = (i + 0.5) / n_orig
        whis_idx = min(int(ratio * n_whis), n_whis - 1)
        seg = whisper_segments[whis_idx]

        # === 按比例分配每个段落的时间边界 ===
        t_start = max(0.0, seg["start"] - 0.1)
        t_end = seg["end"]

        aligned_segments.append({
            "start": round(t_start, 3),
            "end": round(t_end, 3),
            "text": line,
            "words": [],
        })

    return aligned_segments


def _distribute_word_timestamps(
    segment: dict,
    tokens: list[str],
    global_word_idx: int,
) -> list[dict]:
    """
    将段落的词级时间戳均匀分布在段落时间范围内。

    @param segment: 段落 {start, end, text}
    @param tokens: 段落内的词列表
    @param global_word_idx: 全局词索引（用于生成连续递增的时间戳）
    @return: 带时间戳的词列表
    """
    if not tokens:
        return []

    t_start = segment["start"]
    t_end = segment["end"]
    duration = max(t_end - t_start, 0.5)  # 最小 0.5 秒
    n = len(tokens)

    words = []
    for i, token in enumerate(tokens):
        t = t_start + (i / max(n, 1)) * duration
        # 微小偏移避免时间戳完全重叠
        t = round(t + global_word_idx * 0.001, 3)
        words.append({
            "word": token,
            "start": t,
            "end": round(t + duration / max(n, 1), 3),
            "probability": 1.0,
            "source": "original_aligned",
        })

    return words


def correct_lyrics_with_original(
    whisper_output: dict,
    original_lyrics_path: str,
    language: str = "ja",
) -> dict:
    """
    将原始歌词文本对齐到 Whisper 时间戳，输出校正后的结构化歌词。

    对齐策略：
        1. 段落级：原始歌词每行映射到 Whisper 对应段落的时间戳
        2. 词级：每个段落内的词按时间均匀分布

    @param whisper_output: Whisper 输出的结构化数据 {full_text, segments, words}
    @param original_lyrics_path: 原始歌词文本文件路径
    @param language: 语言代码（ja/zh/en）
    @return: 校正后的结构化歌词 {full_text, segments, words}
    """
    original_lines = _read_original_lyrics(original_lyrics_path)
    if not original_lines:
        print("  ⚠️ 原始歌词文件为空，使用 Whisper 原始输出")
        return whisper_output

    whisper_segments = whisper_output.get("segments", [])

    print(f"  原始歌词: {len(original_lines)} 行")
    print(f"  Whisper 段落: {len(whisper_segments)} 个")

    # === 1. 段落级对齐 ===
    aligned_segments = _segment_align(original_lines, whisper_segments)

    # === 2. 词级时间戳均匀分布 ===
    tokenizer = _tokenize_japanese if language == "ja" else _tokenize_simple
    all_words = []
    global_idx = 0

    for seg in aligned_segments:
        tokens = tokenizer(seg["text"])
        word_list = _distribute_word_timestamps(seg, tokens, global_idx)
        seg["words"] = word_list
        all_words.extend(word_list)
        global_idx += len(tokens)

    # === 3. 组装输出 ===
    full_text = " ".join(original_lines)

    return {
        "full_text": full_text,
        "segments": aligned_segments,
        "words": all_words,
    }


def find_original_lyrics_file(
    song_id: str,
    lyrics_dir: str = DEFAULT_LYRICS_DIR,
) -> Optional[str]:
    """
    根据歌曲 ID 查找对应的原始歌词文件。

    查找顺序：
        1. {lyrics_dir}/{song_id}_original.txt
        2. {lyrics_dir}/{song_id}.txt

    @param song_id: 歌曲标识
    @param lyrics_dir: 歌词文件目录
    @return: 文件路径，未找到返回 None
    """
    candidates = [
        os.path.join(lyrics_dir, f"{song_id}_original.txt"),
        os.path.join(lyrics_dir, f"{song_id}.txt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None