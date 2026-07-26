#!/bin/bash
# ============================================================
#  AI MIDI Pipeline — 全自动一键运行脚本
#  目标环境: ModelScope Notebook (Ubuntu 22.04 + CUDA 12.8.1)
#            Python 3.12 + PyTorch 2.10.0 + 24GB 显存 GPU
#
#  用法:
#      1. 把整个 all-in-one-ai-midi-pipeline 目录上传到 /mnt/workspace/
#      2. 把 MP3 文件放到 /mnt/workspace/input/ （保持原目录结构即可）
#      3. cd /mnt/workspace/all-in-one-ai-midi-pipeline
#      4. bash auto_run.sh
#
#  所有下载均走国内镜像，无需科学上网
#  自动检测 MP3、过滤 KAITO/唱见、并行 10 首处理
# ============================================================
set -e

# ============================================================
# === 可配置参数（也可通过环境变量覆盖） ===
# ============================================================
INPUT_DIR="${INPUT_DIR:-/mnt/workspace/input}"          # MP3 输入目录
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/workspace/output}"        # 输出目录
PARALLEL="${PARALLEL:-10}"                               # 并行处理歌曲数
LANGUAGE="${LANGUAGE:-ja}"                               # 歌词语言
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"               # Whisper 模型
SKIP_DRUMS="${SKIP_DRUMS:-true}"                         # 跳过鼓组转录（加速）
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"                      # 从第几阶段开始（0=从头开始）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=========================================="
echo "  🎵 AI MIDI Pipeline — 全自动运行"
echo "  ModelScope Notebook 环境"
echo "=========================================="
echo -e "${NC}"

# ============================================================
# === 阶段零：环境检查 & 镜像配置 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段零] 环境检查 & 镜像配置${NC}"
echo "----------------------------------------"

# --- 检查当前目录 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "  工作目录: $SCRIPT_DIR"

# --- 配置所有镜像源 ---
echo "  配置镜像源..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true
pip config set global.trusted-host mirrors.aliyun.com 2>/dev/null || true

# HuggingFace 镜像（Whisper 模型下载）
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
fi
echo -e "  ${GREEN}✅ 镜像源: aliyun(pip) + hf-mirror(HuggingFace)${NC}"

# --- 检查 GPU ---
echo "  检查 GPU 环境..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'GPU 不可用！'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'  VRAM: {vram:.1f} GB')
" 2>&1 | while read line; do echo "  $line"; done
echo -e "  ${GREEN}✅ GPU 环境正常${NC}"

# --- 安装依赖（幂等：已安装则跳过） ---
echo ""
echo "  检查/安装 Python 依赖..."
pip install --upgrade pip --quiet 2>/dev/null || true

# 先装 numpy（basic-pitch 编译依赖它）
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "    numpy — 安装中..."
    pip install numpy --quiet || echo "    ⚠️ numpy 安装失败"
else
    echo "    numpy — 已安装"
fi

# 核心依赖（basic-pitch 需要 --no-build-isolation 避免编译 numpy）
for pkg in pyyaml tqdm soundfile faster-whisper demucs pretty_midi; do
    if python3 -c "import ${pkg//-/_}" 2>/dev/null; then
        echo "    ${pkg} — 已安装"
    else
        echo "    ${pkg} — 安装中..."
        pip install "$pkg" --quiet || echo "    ⚠️ ${pkg} 安装失败"
    fi
done

# basic-pitch 单独处理：必须用 --no-build-isolation 复用已安装的 numpy
if python3 -c "import basic_pitch" 2>/dev/null; then
    echo "    basic-pitch — 已安装"
else
    echo "    basic-pitch — 安装中（编译较慢，约 1-2 分钟）..."
    pip install basic-pitch --no-build-isolation 2>&1 | tail -3 || echo "    ⚠️ basic-pitch 安装失败，请检查网络"
fi
echo -e "  ${GREEN}✅ 依赖检查完成${NC}"


# ============================================================
# === 阶段一：预下载模型（走镜像，幂等） ===
#  直接尝试加载模型——若已缓存则秒过，未缓存才下载
# ============================================================
echo ""
echo -e "${GREEN}[阶段一] 预下载 AI 模型${NC}"
echo "----------------------------------------"

# --- Whisper large-v3（走 HF Mirror） ---
echo "  [1/3] Whisper large-v3 模型（约 3GB，走 hf-mirror）..."
python3 -c "
import os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
t0 = time.time()
from faster_whisper import WhisperModel
model = WhisperModel(
    'large-v3',
    device='cuda',
    compute_type='float16',
    download_root='/root/.cache/huggingface',
)
elapsed = time.time() - t0
if elapsed < 3:
    print('    ✅ 已缓存，加载耗时 {:.1f}s'.format(elapsed))
else:
    print('    ✅ 下载完成，耗时 {:.1f}s'.format(elapsed))
" || echo "    ⚠️ Whisper 模型加载失败，请检查 HF_ENDPOINT 网络"

# --- Demucs htdemucs_6s（走 Facebook CDN） ---
echo "  [2/3] Demucs htdemucs_6s 模型（约 320MB）..."
python3 -c "
import time, torch
t0 = time.time()
from demucs import pretrained
model = pretrained.get_model('htdemucs_6s')
if torch.cuda.is_available():
    model.cuda()
elapsed = time.time() - t0
if elapsed < 3:
    print('    ✅ 已缓存，加载耗时 {:.1f}s'.format(elapsed))
else:
    print('    ✅ 下载完成，耗时 {:.1f}s'.format(elapsed))
" || echo "    ⚠️ Demucs 模型加载失败，后续运行时会自动重试"

# --- Basic Pitch（走 tfhub.dev） ---
echo "  [3/3] Basic Pitch 模型（约 60MB，走 tfhub.dev）..."
python3 -c "
import numpy as np, soundfile as sf, tempfile, os, time
dummy = np.zeros((44100 * 2,), dtype=np.float32)
tmp = os.path.join(tempfile.gettempdir(), '_bp_dummy.wav')
sf.write(tmp, dummy, 44100)
t0 = time.time()
from basic_pitch.inference import predict
predict(tmp)
os.remove(tmp)
elapsed = time.time() - t0
if elapsed < 3:
    print('    ✅ 已缓存，加载耗时 {:.1f}s'.format(elapsed))
else:
    print('    ✅ 下载完成，耗时 {:.1f}s'.format(elapsed))
" || echo "    ⚠️ Basic Pitch 模型加载失败，后续运行时会自动重试"

echo -e "  ${GREEN}✅ 模型准备完成${NC}"


# ============================================================
# === 阶段二：输入文件检测 & 过滤 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段二] 输入文件检测 & 过滤${NC}"
echo "----------------------------------------"

# --- 创建扁平化输入目录 ---
FLAT_INPUT="$SCRIPT_DIR/input"
rm -rf "$FLAT_INPUT"
mkdir -p "$FLAT_INPUT"

# --- 递归查找所有 MP3 ---
echo "  搜索目录: $INPUT_DIR"
MP3_COUNT=$(find "$INPUT_DIR" -name "*.mp3" 2>/dev/null | wc -l)
echo "  找到 $MP3_COUNT 个 MP3 文件"

# --- 过滤规则：排除 KAITO 版本、唱见演唱、CC字幕版（优先用中文字幕版） ---
# 使用临时文件避免管道子 shell 导致的计数丢失
FILTER_TMP=$(mktemp)
find "$INPUT_DIR" -name "*.mp3" 2>/dev/null > "$FILTER_TMP"

COPIED=0
SKIPPED_KAITO=0
SKIPPED_COVER=0
SKIPPED_CC=0

while read -r f; do
    [ -z "$f" ] && continue
    basename_f=$(basename "$f")

    # 规则1：排除 KAITO 版本
    if echo "$f" | grep -qi "KAITO"; then
        echo "    ❌ 跳过(KAITO): $basename_f"
        SKIPPED_KAITO=$((SKIPPED_KAITO + 1))
        continue
    fi

    # 规则2：排除唱见演唱
    if echo "$f" | grep -q "唱见演唱"; then
        echo "    ❌ 跳过(唱见): $basename_f"
        SKIPPED_COVER=$((SKIPPED_COVER + 1))
        continue
    fi

    # 规则3：同一目录下，有中文字幕版时跳过 CC字幕版
    if echo "$basename_f" | grep -q "CC字幕"; then
        parent_dir=$(dirname "$f")
        if ls "$parent_dir"/*中文字幕* 2>/dev/null | grep -v "CC字幕" > /dev/null; then
            echo "    ❌ 跳过(CC字幕，已有中文字幕版): $basename_f"
            SKIPPED_CC=$((SKIPPED_CC + 1))
            continue
        fi
    fi

    # 复制到扁平输入目录（用子目录名+文件名避免冲突）
    rel_path="${f#$INPUT_DIR/}"
    safe_name=$(echo "$rel_path" | tr '/' '_' | tr ' ' '_')
    cp "$f" "$FLAT_INPUT/$safe_name"
    echo "    ✅ $safe_name"
    COPIED=$((COPIED + 1))
done < "$FILTER_TMP"
rm -f "$FILTER_TMP"

echo ""
echo "  ┌───────────────────────────────┐"
echo "  │  复制:        ${COPIED} 首"
echo "  │  跳过(KAITO): ${SKIPPED_KAITO} 首"
echo "  │  跳过(唱见):  ${SKIPPED_COVER} 首"
echo "  │  跳过(CC字幕): ${SKIPPED_CC} 首"
echo "  └───────────────────────────────┘"

# 重新统计实际数量
FINAL_COUNT=$(ls "$FLAT_INPUT"/*.mp3 2>/dev/null | wc -l)
echo -e "  ${GREEN}✅ 最终待处理: $FINAL_COUNT 首歌曲${NC}"

if [ "$FINAL_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ 没有找到有效的 MP3 文件！请检查 INPUT_DIR 路径。${NC}"
    exit 1
fi


# ============================================================
# === 阶段三：运行处理流水线 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段三] 启动处理流水线${NC}"
echo "----------------------------------------"
echo "  输入目录: $FLAT_INPUT"
echo "  输出目录: $OUTPUT_DIR"
echo "  并行数:   $PARALLEL"
echo "  语言:     $LANGUAGE"
echo "  Whisper:  $WHISPER_MODEL"
echo "  跳过鼓组: $SKIP_DRUMS"
echo ""

# 构建命令行参数
CMD_ARGS=(
    "batch"
    "$FLAT_INPUT/*.mp3"
    "--output" "$OUTPUT_DIR"
    "--language" "$LANGUAGE"
    "--whisper-model" "$WHISPER_MODEL"
    "--parallel" "$PARALLEL"
)

[ "$SKIP_DRUMS" = "true" ] && CMD_ARGS+=("--skip-drums")
[ "$SKIP_TO_STAGE" != "0" ] && CMD_ARGS+=("--skip-to-stage" "$SKIP_TO_STAGE")

echo "  执行命令:"
echo "  python3 run_extraction.py ${CMD_ARGS[*]}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行
python3 run_extraction.py "${CMD_ARGS[@]}"
EXIT_CODE=$?

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))


# ============================================================
# === 阶段四：结果汇总 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段四] 结果汇总${NC}"
echo "----------------------------------------"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "  ${GREEN}✅ 流水线执行成功${NC}"
else
    echo -e "  ${RED}❌ 流水线执行异常 (exit code: $EXIT_CODE)${NC}"
fi

echo "  总耗时: ${MINUTES} 分 ${SECONDS} 秒"
echo ""

# 统计输出文件
if [ -d "$OUTPUT_DIR" ]; then
    echo "  📂 输出目录: $OUTPUT_DIR"
    echo ""
    echo "  各歌曲输出文件:"
    for song_dir in "$OUTPUT_DIR"/*/; do
        song_name=$(basename "$song_dir")
        file_count=$(ls "$song_dir" 2>/dev/null | wc -l)
        echo "    📁 $song_name ($file_count 个文件)"
        # 列出关键文件类型
        for ext in "_lyrics.txt" "_lyrics_timed.json" "_alignment.json" "_merged.mid" "_vocals.mid" "_summary.json"; do
            if ls "$song_dir"/*"$ext" 2>/dev/null; then
                echo "       ✅ *$ext"
            fi
        done
    done

    echo ""
    echo "  📊 数据集文件:"
    [ -f "$OUTPUT_DIR/dataset.json" ] && echo "    ✅ dataset.json — 汇总数据集"
    [ -f "$OUTPUT_DIR/batch_report.txt" ] && echo "    ✅ batch_report.txt — 批量处理报告"
    [ -f "$OUTPUT_DIR/failed_songs.json" ] && echo "    ⚠️ failed_songs.json — 失败歌曲列表"
fi


# ============================================================
# === 完成 ===
# ============================================================
echo ""
echo -e "${BLUE}=========================================="
echo "  🎉 全自动处理完成！"
echo "=========================================="
echo -e "${NC}"
echo "  输出目录: $OUTPUT_DIR"
echo "  总耗时:   ${MINUTES} 分 ${SECONDS} 秒"
echo ""
echo "  每首歌曲包含:"
echo "    - _lyrics.txt        歌词纯文本"
echo "    - _lyrics_timed.json 带时间戳的歌词"
echo "    - _alignment.json    词-音对齐映射表"
echo "    - _merged.mid        多轨 MIDI 文件"
echo "    - _summary.json      处理报告"
echo ""
echo "  如需重新处理（例如调整参数）："
echo "    PARALLEL=5 bash auto_run.sh"
echo ""