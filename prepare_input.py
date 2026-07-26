"""
输入文件去重整理脚本

从原始目录中筛选唯一歌曲，排除重复版本（CC字幕版/中文字幕版/唱见版等），
优先选择 CC 字幕版或完整版本，复制到 input 目录供后续处理。

@implSpec
    上游调用方：人工执行 / 初始化阶段
    下游依赖：os, shutil, re
    在架构图中的位置：Utility 层 — 输入数据预处理

筛选规则：
    1. 同一首歌有多个版本时，优先选择 CC字幕版（字幕信息更准确）
    2. 排除唱见演唱版本（非 VOCALOID 原唱）
    3. 排除「不全」版本
    4. 文件名中的歌手信息用于去重判断

@author AI Assistant
"""

import os
import re
import shutil


# === 常量定义 ===
PRIORITY_CC = 3           # CC字幕版优先级最高
PRIORITY_COMPLETE = 2     # 完整版本
PRIORITY_STANDARD = 1     # 普通中文字幕版
PRIORITY_LOW = 0          # 低优先级（不全版本等）
SKIP_SINGERS = ["兎迷夢々", "つぐ", "SHIKI", "フリモメン", "KAITO"]  # 跳过唱见/非初音版本
SKIP_DIR_KEYWORDS = ["专辑", "单曲", "唱见"]  # 跳过分类目录名


def deduplicate_songs(source_dir: str, output_dir: str) -> list:
    """
    从源目录去重并复制歌曲到输出目录。

    @param source_dir: 源目录路径
    @param output_dir: 输出目录路径
    @return: 复制的文件列表 [(song_name, file_path), ...]
    """
    all_mp3_files = _collect_all_mp3s(source_dir)
    song_groups = _group_by_song(all_mp3_files)
    selected = _select_best_versions(song_groups)

    os.makedirs(output_dir, exist_ok=True)

    copied = []
    for song_name, file_path in selected.items():
        safe_name = _sanitize_filename(song_name)
        dest_path = os.path.join(output_dir, f"{safe_name}.mp3")
        shutil.copy2(file_path, dest_path)
        copied.append((safe_name, dest_path))
        print(f"  ✅ {safe_name}")

    print(f"\n共筛选出 {len(copied)} 首唯一歌曲，已复制到: {output_dir}")
    return copied


def _collect_all_mp3s(source_dir: str) -> list:
    """
    递归收集目录下所有 MP3 文件。

    @param source_dir: 源目录
    @return: MP3 文件路径列表
    """
    mp3_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".mp3"):
                mp3_files.append(os.path.join(root, f))
    return mp3_files


def _group_by_song(mp3_files: list) -> dict:
    """
    按歌曲名分组。

    @param mp3_files: MP3 文件路径列表
    @return: 分组字典 {song_key: [(file_path, priority), ...]}
    """
    groups = {}

    for file_path in mp3_files:
        filename = os.path.basename(file_path)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        grandparent_dir = os.path.basename(
            os.path.dirname(os.path.dirname(file_path))
        )

        if _should_skip(filename, parent_dir):
            continue

        song_dir = _resolve_song_dir(file_path, filename, parent_dir, grandparent_dir)
        song_key = _extract_song_key(filename, song_dir)
        if not song_key:
            continue

        priority = _calc_priority(filename, file_path)

        if song_key not in groups:
            groups[song_key] = []
        groups[song_key].append((file_path, priority))

    return groups


def _resolve_song_dir(
    file_path: str,
    filename: str,
    parent_dir: str,
    grandparent_dir: str,
) -> str:
    """
    确定歌曲所在的目录名（向上跳过分类目录）。

    @param file_path: 文件完整路径
    @param filename: 文件名
    @param parent_dir: 父目录名
    @param grandparent_dir: 祖父目录名
    @return: 用于提取歌曲名的源文本
    """
    if _is_category_dir(parent_dir):
        if _is_version_only_name(filename):
            return grandparent_dir
        return filename

    if _is_version_only_name(filename):
        return parent_dir

    return filename if _is_filename_has_songname(filename) else parent_dir


def _is_filename_has_songname(filename: str) -> bool:
    """
    判断文件名本身是否包含歌曲名（非单纯版本名）。

    @param filename: 文件名
    @return: 是否包含歌曲名
    """
    name = os.path.splitext(filename)[0]
    if _is_version_only_name(filename):
        return False
    has_japanese = bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF]", name))
    has_chinese = bool(re.search(r"[\u4E00-\u9FFF]", name))
    return has_japanese or has_chinese


def _is_category_dir(dir_name: str) -> bool:
    """
    判断目录名是否为分类目录（专辑/单曲等）。

    @param dir_name: 目录名
    @return: 是否为分类目录
    """
    for kw in SKIP_DIR_KEYWORDS:
        if kw in dir_name:
            return True
    return False


def _should_skip(filename: str, parent_dir: str) -> bool:
    """
    判断是否应跳过该文件（唱见版本、不完整版本等）。

    @param filename: 文件名
    @param parent_dir: 父目录名
    @return: 是否跳过
    """
    for singer in SKIP_SINGERS:
        if singer in filename or singer in parent_dir:
            return True

    if "唱见" in parent_dir:
        return True

    if "不全" in filename:
        return True

    return False


def _extract_song_key(filename: str, song_dir: str) -> str:
    """
    从歌曲目录名或文件名提取歌曲标识键。

    优先提取日文原名（下划线后的部分），其次提取中文名。

    @param filename: 文件名（用于辅助判断）
    @param song_dir: 歌曲所在目录名或文件名（提取用源文本）
    @return: 歌曲标识键
    """
    source_text = os.path.splitext(song_dir)[0]

    raw_name = _extract_japanese_title(source_text)
    if raw_name:
        return raw_name

    raw_name = _extract_chinese_title(source_text)
    if raw_name:
        return raw_name

    cleaned = re.sub(r"\[.*?\]|【.*?】|（.*?）|\(.*?\)", "", source_text).strip()
    cleaned = cleaned.strip("_- ")
    return cleaned if cleaned else song_dir


def _is_version_only_name(filename: str) -> bool:
    """
    判断文件名是否仅包含版本信息（不含歌曲名）。

    @param filename: 文件名
    @return: 是否仅为版本名
    """
    name = os.path.splitext(filename)[0]
    version_keywords = ["CC字幕版", "中文字幕版", "字幕版", "完整版", "无遮挡"]
    for kw in version_keywords:
        if kw in name:
            return True
    return False


def _extract_japanese_title(text: str) -> str:
    """
    从文本中提取日文歌曲名（下划线后的日文部分）。

    @param text: 源文本
    @return: 日文歌名，提取失败返回空串
    """
    pattern = r"[_＿]\s*([\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u00C0-\u017F]+\s*[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u00C0-\u017F]*)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_chinese_title(text: str) -> str:
    """
    从文本中提取中文歌曲名。

    @param text: 源文本
    @return: 中文歌名，提取失败返回空串
    """
    pattern = r"\[中文字幕[_\s]*(.+?)(?:\]|【)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


def _calc_priority(filename: str, file_path: str) -> int:
    """
    计算文件版本的优先级。

    @param filename: 文件名
    @param file_path: 文件完整路径
    @return: 优先级分数（越高越优）
    """
    priority = PRIORITY_STANDARD

    if "CC" in filename or "cc" in filename.lower():
        priority = max(priority, PRIORITY_CC)

    if "中文字幕" in filename:
        priority = max(priority, PRIORITY_STANDARD)

    if "无遮挡" in filename or "完整" in filename:
        priority = max(priority, PRIORITY_COMPLETE)

    if "不全" in filename:
        priority = PRIORITY_LOW

    return priority


def _select_best_versions(song_groups: dict) -> dict:
    """
    从每组中选择优先级最高的版本。

    @param song_groups: 分组字典
    @return: {song_name: best_file_path}
    """
    selected = {}
    for song_key, versions in song_groups.items():
        versions.sort(key=lambda x: x[1], reverse=True)
        best_path = versions[0][0]
        selected[song_key] = best_path
    return selected


def _sanitize_filename(name: str) -> str:
    """
    清理文件名，移除不合法字符。

    @param name: 原始名称
    @return: 清理后的名称
    """
    unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
    safe = re.sub(unsafe_chars, "_", name)
    safe = safe.strip().strip(".")
    return safe if safe else "untitled"


if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else r"D:\MIDI\programs"
    output = sys.argv[2] if len(sys.argv) > 2 else r"D:\MIDI\all-in-one-ai-midi-pipeline\input"

    print(f"源目录: {source}")
    print(f"输出目录: {output}")
    print("\n正在筛选去重...\n")

    results = deduplicate_songs(source, output)
