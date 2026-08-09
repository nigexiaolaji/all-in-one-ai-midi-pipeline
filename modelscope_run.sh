#!/bin/bash
# ============================================================
#  ModelScope 一键处理脚本（魔塔 CPU 版：8核32GB，py312 镜像，预装 torch 2.3.1）
#  用法（在魔塔 Notebook 终端执行）:
#      bash modelscope_run.sh                                 # 去重模式（每首只留最佳版本）
#      SKIP_PREPARE=true bash modelscope_run.sh               # 全量模式（所有 MP3 全处理，不过滤）
#      RAW_DIR=/mnt/workspace/programs bash modelscope_run.sh # 指定原始 MP3 目录
#      PARALLEL=3 SKIP_PREPARE=true bash modelscope_run.sh    # 覆盖并行数
#      PYTHON_BIN=python3.11 bash modelscope_run.sh           # 指定 Python 解释器
#
#  流程: 原始 MP3（默认 programs/）→ input/
#        → auto_run.sh（装依赖/下模型/跑流水线/汇总）
#
#  默认（去重模式）: prepare_input.py 按歌曲名分组，每首只保留优先级最高的版本
#        （CC字幕/完整版优先，排除唱见/不全）
#  SKIP_PREPARE=true（全量模式）: 递归复制 RAW_DIR 下所有 MP3，一首不漏
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAW_DIR="${RAW_DIR:-/mnt/workspace/programs}"     # 原始 MP3 目录（魔塔默认挂载点）
SKIP_PREPARE="${SKIP_PREPARE:-false}"             # true=全量复制，不筛选版本
PARALLEL="${PARALLEL:-8}"                          # 并行歌曲数（8 核 CPU 拉满 = 8；内存不足可降到 4-6）
FORCE_CPU="${FORCE_CPU:-1}"                        # 默认全程 CPU 模式（1=禁用所有 GPU 加速）
export FORCE_CPU
# Python 解释器：默认 python3（py312 镜像）；预装 python3.11 时可 PYTHON_BIN=python3.11 覆盖
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ 未找到 Python 解释器: $PYTHON_BIN（可用 PYTHON_BIN=python3.11 指定）"
    exit 1
fi
# CPU 模式线程配置：8 个 worker 进程各 1 线程 = 8 核满载且不互相超订
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export WHISPER_CPU_THREADS="${WHISPER_CPU_THREADS:-1}"
INPUT_DIR="$SCRIPT_DIR/input"

echo "=========================================="
echo "  🎵 ModelScope 一键处理（CPU 版）"
echo "=========================================="
echo "  原始目录: $RAW_DIR"
echo "  模式:     $([ "$SKIP_PREPARE" = "true" ] && echo '全量（不筛选）' || echo '去重（每首最佳版本）')"
echo "  并行数:   $PARALLEL（8 核拉满）"
echo "  设备:     $([ "$FORCE_CPU" = "1" ] && echo 'CPU 模式（已禁用 GPU）' || echo '自动检测 GPU/CPU')"
echo "  Python:   $("$PYTHON_BIN" --version 2>&1)"
echo "  Whisper:  large-v3（默认）"

# --- 步骤一：准备 input/ ---
if [ "$SKIP_PREPARE" = "true" ]; then
    echo -e "\n[1/2] 全量模式：复制 $RAW_DIR 下所有 MP3 → input/（不筛选）"
    if [ -d "$RAW_DIR" ]; then
        mkdir -p "$INPUT_DIR"
        COPIED=0
        while IFS= read -r -d '' f; do
            rel="${f#"$RAW_DIR"/}"
            safe_name=$(echo "$rel" | tr '/' '_' | tr ' ' '_')
            cp "$f" "$INPUT_DIR/$safe_name"
            COPIED=$((COPIED + 1))
        done < <(find "$RAW_DIR" -iname "*.mp3" -print0 2>/dev/null)
        echo "  ✅ 已复制 $COPIED 首 MP3"
    else
        echo "  ⚠️ 未找到 $RAW_DIR，直接使用已有 input/（请确认已放入 MP3）"
    fi
else
    echo -e "\n[1/2] 去重模式：prepare_input.py 筛选"
    if [ -d "$RAW_DIR" ]; then
        "$PYTHON_BIN" prepare_input.py "$RAW_DIR" "$INPUT_DIR"
    else
        echo "  ⚠️ 未找到 $RAW_DIR，改用已有 input/（请自行放入 MP3）"
    fi
fi

COUNT=$(find "$INPUT_DIR" -maxdepth 1 -iname "*.mp3" 2>/dev/null | wc -l)
if [ "$COUNT" -eq 0 ]; then
    echo "  ❌ input/ 下没有 MP3，请先把歌曲放进去再运行。"
    exit 1
fi
echo -e "  ✅ input/ 共 $COUNT 首 MP3\n"

# --- 步骤二：交给 auto_run.sh（依赖/模型/流水线/汇总） ---
echo "[2/2] 启动 auto_run.sh ..."
export INPUT_DIR="$INPUT_DIR"
export PARALLEL="$PARALLEL"
exec bash auto_run.sh
