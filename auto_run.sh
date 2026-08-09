#!/bin/bash
# ============================================================
#  AI MIDI Pipeline — 全自动一键运行脚本
#  目标环境: ModelScope Notebook CPU 版（8核32GB，长期使用实例）
#            预装镜像 ubuntu22.04-py312-torch2.3.1-1.39.0
#            （Python 3.12 + PyTorch 2.3.1 CPU 版 + ModelScope Library）
#  默认全程 CPU 模式；如需 GPU 可 FORCE_CPU=0 覆盖
#
#  用法:
#      1. 把 all-in-one-ai-midi-pipeline 目录上传到 /mnt/workspace/
#      2. 把 model/ 目录上传到 /mnt/workspace/ （内含 large/ 与 small/ 两个 Whisper 模型，
#         上传后脚本直接本地加载，无需再下载）
#      3. 把 MP3 文件放到 /mnt/workspace/all-in-one-ai-midi-pipeline/input/
#         （建议先用 prepare_input.py 去重筛选）
#      4. cd /mnt/workspace/all-in-one-ai-midi-pipeline && bash auto_run.sh
#
#  所有下载均走国内镜像，无需科学上网
# ============================================================
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# === 基础路径 & 可配置参数（可通过环境变量覆盖） ===
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_DIR="${INPUT_DIR:-$SCRIPT_DIR/input}"              # MP3 输入目录（默认 pipeline 内 input/）
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/workspace/output}"        # 输出目录
PARALLEL="${PARALLEL:-3}"                                # 并行处理歌曲数（spawn 下每 worker 独立加载 TF 模型，CPU 建议 2-3）
LYRICS_LANG="${LYRICS_LANG:-ja}"                         # 歌词语言（注意：不能叫 LANGUAGE，会被系统 locale 环境变量覆盖）
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"               # Whisper 模型 (large-v3 / small)
SKIP_DRUMS="${SKIP_DRUMS:-true}"                         # 跳过鼓组转录（加速，CPU 上极慢）
SKIP_TO_STAGE="${SKIP_TO_STAGE:-0}"                      # 从第几阶段开始（0=从头开始）
MODEL_DIR="${MODEL_DIR:-$(dirname "$SCRIPT_DIR")/model}" # 本地模型根目录（large/ 与 small/）
FORCE_CPU="${FORCE_CPU:-1}"                              # 默认 CPU 模式（1=禁用 GPU 加速；GPU 环境可 FORCE_CPU=0）
export FORCE_CPU                                         # 必须 export：Whisper/demucs/歌词模块靠环境变量判断 CPU 模式
# Python 解释器：默认 python3（py312 镜像）；预装 python3.11 时可 PYTHON_BIN=python3.11 覆盖
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo -e "${RED}❌ 未找到 Python 解释器: $PYTHON_BIN（可用 PYTHON_BIN=python3.11 指定）${NC}"
    exit 1
fi

echo -e "${BLUE}"
echo "=========================================="
echo "  🎵 AI MIDI Pipeline — 全自动运行"
echo "  ModelScope Notebook CPU 环境"
echo "=========================================="
echo -e "${NC}"

# ============================================================
# === 阶段零：环境检查 & 镜像配置 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段零] 环境检查 & 镜像配置${NC}"
echo "----------------------------------------"
echo "  工作目录: $SCRIPT_DIR"
echo "  输入目录: $INPUT_DIR"
echo "  模型目录: $MODEL_DIR"

# --- 配置镜像源 ---
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true
pip config set global.trusted-host mirrors.aliyun.com 2>/dev/null || true

export HF_ENDPOINT=https://hf-mirror.com
# 禁用 huggingface_hub 的 xet 下载后端（魔塔访问 cas-server.xethub.hf.co 会 401）
export HF_HUB_DISABLE_XET=1
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
fi
echo -e "  ${GREEN}✅ 镜像源: aliyun(pip) + hf-mirror(HuggingFace)${NC}"

# --- 打印运行环境版本 ---
echo -e "  Python:   $("$PYTHON_BIN" --version 2>&1)"
"$PYTHON_BIN" -c "import torch; print('  PyTorch: ', torch.__version__, '| CUDA 可用:', torch.cuda.is_available())" 2>/dev/null \
    || echo -e "  ${YELLOW}⚠️ torch 未安装或不可导入（依赖检查阶段会自动安装）${NC}"

# --- 检查 GPU（CPU 默认直接跳过；SKIP_GPU_CHECK=1 或 FORCE_CPU=1 时跳过） ---
if [ "${FORCE_CPU:-1}" = "1" ] || [ "${SKIP_GPU_CHECK:-false}" = "true" ]; then
    echo -e "  ${GREEN}✅ 运行模式: CPU（FORCE_CPU=1，8核并行）${NC}"
elif command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 | while read -r name mem; do
        echo "  GPU: $name ($mem)"
    done; then
        echo -e "  ${GREEN}✅ GPU 环境正常${NC}"
    else
        echo -e "  ${RED}⚠️ nvidia-smi 查询失败，GPU 可能不可用${NC}"
    fi
else
    echo "  nvidia-smi 不可用，改用 torch 检测（首次加载较慢）..."
    "$PYTHON_BIN" -c "
import torch
assert torch.cuda.is_available(), 'GPU 不可用！'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'  VRAM: {vram:.1f} GB')
" 2>&1 | while read -r line; do echo "  $line"; done
    echo -e "  ${GREEN}✅ GPU 环境正常${NC}"
fi

# --- 检查/安装依赖（幂等，Python 3.12 不锁定版本） ---
echo ""
echo "  检查/安装 Python 依赖..."
pip install --upgrade pip --quiet 2>/dev/null || true

# 软件包名 -> 导入名的映射（pyyaml 导入名为 yaml，不能直接替换连字符）
declare -A PKG_IMPORT=(
    [pyyaml]=yaml
    [tqdm]=tqdm
    [soundfile]=soundfile
    [faster-whisper]=faster_whisper
    [demucs]=demucs
    [pretty_midi]=pretty_midi
    [music21]=music21
)

for pkg in "${!PKG_IMPORT[@]}"; do
    import_name="${PKG_IMPORT[$pkg]}"
    if "$PYTHON_BIN" -c "import ${import_name}" 2>/dev/null; then
        echo "    ${pkg} — 已安装"
    else
        echo "    ${pkg} — 安装中..."
        pip install "$pkg" --quiet || echo "    ⚠️ ${pkg} 安装失败"
    fi
done

# --- 版本兼容修复：onnx / ml-dtypes / protobuf 与 tensorflow 2.15 互斥 ---
# demucs 经 torch.onnx 会导入 onnx 包，新版 onnx（>=1.17）引用 ml_dtypes.float4_e2m1fn（需 0.4+），
# 而 tensorflow 2.15 要求 ml-dtypes~=0.2.0 且 protobuf<5 —— 两者互斥。
# 解法：onnx 降到 <1.17（不再引用 float4）+ ml-dtypes 固定 0.2.x + protobuf <5，
# 使 demucs 与 basic_pitch(tensorflow) 同时可运行。
echo "    检查 tensorflow 兼容版本（onnx/ml-dtypes/protobuf）..."
"$PYTHON_BIN" -c "
import importlib.metadata as m

def ver(p):
    try:
        return tuple(map(int, m.version(p).split('.')[:3]))
    except Exception:
        return None

fix = []
ov = ver('onnx')
if ov is not None and ov >= (1, 17, 0):
    print('    onnx 版本过高 (%s)，降级 <1.17 ...' % m.version('onnx'))
    fix.append('onnx<1.17')
mv = ver('ml_dtypes')
if mv is not None and not (0, 2, 0) <= mv < (0, 3, 0):
    print('    ml-dtypes 需 0.2.x（当前 %s），修复 ...' % m.version('ml_dtypes'))
    fix.append('ml-dtypes~=0.2.0')
pv = ver('protobuf')
if pv is not None and pv >= (5, 0, 0):
    print('    protobuf 版本过高 (%s)，降级 <5 ...' % m.version('protobuf'))
    fix.append('protobuf>=3.20.3,<5.0.0dev')
if fix:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-input'] + fix)
    print('    ✅ 版本修复完成')
else:
    print('    ✅ 版本组合 OK')
" 2>/dev/null || echo "    ⚠️ 版本检查失败（onnx 未安装时无需处理，demucs 分离时会自动跳过）"

# midigpt 不再自动安装：其依赖（datasets/pyarrow 等）会把 protobuf 拉高到 7.x，
# 与 tensorflow 2.15（basic_pitch 依赖，要求 protobuf<5）冲突，破坏提取流水线。
# 训练时请用独立 venv 安装：pip install -r requirements-midigpt.txt
echo "    midigpt — 跳过自动安装（训练时用独立 venv 装 requirements-midigpt.txt，避免 protobuf 与 tensorflow 冲突）"

# adtof_pytorch 单独处理：鼓转录用（git 安装，走 gh-proxy 镜像）
# 注意：steps/transcribe_drums.py 已改为函数内懒加载，装不上时 SKIP_DRUMS=true 不受影响
if "$PYTHON_BIN" -c "import adtof_pytorch" 2>/dev/null; then
    echo "    adtof_pytorch — 已安装"
else
    echo "    adtof_pytorch — 安装中（git+gh-proxy，约 1-2 分钟）..."
    if pip install "adtof_pytorch @ git+https://v4.gh-proxy.org/https://github.com/xavriley/ADTOF-pytorch.git@main#egg=adtof_pytorch" --quiet 2>/dev/null; then
        echo "    adtof_pytorch — 安装完成"
    else
        echo "    ⚠️ adtof_pytorch 安装失败（SKIP_DRUMS=true 时不影响运行）"
    fi
fi

# basic-pitch 单独处理
#  - Python <=3.11：标准安装（可能需编译，先装构建依赖）
#  - Python 3.12+：官方依赖 tensorflow>=2.4.1,<2.15.1 在 3.12 无 wheel，
#    且旧 numpy 源码（numpy.distutils 引用 distutils.msvccompiler）在 3.12 编译失败
#    → 兼容方案：numpy==1.26.4 wheel + pip install --no-deps 绕开依赖解析 +
#      tensorflow-cpu==2.16.1 + 手动补齐运行时依赖
if "$PYTHON_BIN" -c "import basic_pitch" 2>/dev/null; then
    echo "    basic-pitch — 已安装"
else
    PY_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[0])')
    PY_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info[1])')
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 12 ]; then
        echo "    basic-pitch — 安装中（Python 3.12 兼容方案：numpy 1.26 + tensorflow-cpu 2.16）..."
        pip install "numpy==1.26.4" --only-binary=:all: --quiet 2>/dev/null || true
        if pip install basic-pitch --no-deps > /tmp/basic_pitch_install.log 2>&1 \
           && pip install "tensorflow-cpu==2.16.1" librosa mir_eval pretty_midi "resampy<0.4.3" scikit-learn scipy typing_extensions --quiet >> /tmp/basic_pitch_install.log 2>&1 \
           && "$PYTHON_BIN" -c "import basic_pitch" 2>/dev/null; then
            echo "    basic-pitch — 安装完成（Python 3.12 兼容方案）"
        else
            echo "    ⚠️ basic-pitch 安装失败（日志: /tmp/basic_pitch_install.log）"
        fi
    else
        echo "    basic-pitch — 安装中（编译较慢，约 1-2 分钟）..."
        pip install Cython wheel --quiet 2>/dev/null || true
        if pip install basic-pitch > /tmp/basic_pitch_install.log 2>&1; then
            echo "    basic-pitch — 安装完成"
        else
            echo "    ⚠️ basic-pitch 安装失败（日志: /tmp/basic_pitch_install.log）"
        fi
    fi
fi
echo -e "  ${GREEN}✅ 依赖检查完成${NC}"


# ============================================================
# === 阶段一：Whisper 模型准备（本地优先，缺失才下载） ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段一] Whisper 模型准备${NC}"
echo "----------------------------------------"

case "$WHISPER_MODEL" in
    large|large-v3)
        LOCAL_W_DIR="$MODEL_DIR/large"
        MODELSCOPE_ID="Systran/faster-whisper-large-v3"
        ;;
    small)
        LOCAL_W_DIR="$MODEL_DIR/small"
        MODELSCOPE_ID="Systran/faster-whisper-small"
        ;;
    *)
        echo -e "  ${RED}❌ 未知的 WHISPER_MODEL: $WHISPER_MODEL（可选: large-v3 / small）${NC}"
        exit 1
        ;;
esac

if [ -f "$LOCAL_W_DIR/model.bin" ]; then
    echo -e "  ${GREEN}✅ 使用本地模型: $LOCAL_W_DIR${NC}"
else
    echo "  本地模型不存在（$LOCAL_W_DIR），从 ModelScope 内网下载 $MODELSCOPE_ID..."
    "$PYTHON_BIN" -c "import modelscope" 2>/dev/null || pip install modelscope --quiet

    "$PYTHON_BIN" -c "
import os, time, shutil
from modelscope import snapshot_download
t0 = time.time()
model_dir = snapshot_download(
    '$MODELSCOPE_ID',
    cache_dir='/root/.cache/modelscope_whisper',
)
print('    ModelScope 下载耗时 {:.0f}s'.format(time.time() - t0))

# 把下载的模型复制到本地模型目录（extract_lyrics.py 的 LOCAL_MODEL_DIRS 直接命中，
# 加载走本地路径，彻底绕开 huggingface_hub 的 xet 下载后端）
os.makedirs('$LOCAL_W_DIR', exist_ok=True)
copied = 0
for fname in os.listdir(model_dir):
    src = os.path.join(model_dir, fname)
    dst = os.path.join('$LOCAL_W_DIR', fname)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        copied += 1
print('    已复制 %d 个文件到 %s' % (copied, '$LOCAL_W_DIR'))

from faster_whisper import WhisperModel
import os as _os
t0 = time.time()
if _os.environ.get('FORCE_CPU') == '1':
    model = WhisperModel('$LOCAL_W_DIR', device='cpu', compute_type='int8', cpu_threads=8)
    print('    ✅ 加载成功（CPU 模式），耗时 {:.1f}s'.format(time.time() - t0))
else:
    model = WhisperModel('$LOCAL_W_DIR', device='cuda', compute_type='float16')
    print('    ✅ 加载成功，耗时 {:.1f}s'.format(time.time() - t0))
" || echo "    ⚠️ Whisper 模型下载/加载失败，可在重试前检查模型缓存目录"
fi

# --- Demucs htdemucs_6s（走 Facebook CDN） ---
echo "  [2/3] Demucs htdemucs_6s 模型（约 320MB）..."
"$PYTHON_BIN" -c "
import time, torch, os
t0 = time.time()
from demucs import pretrained
model = pretrained.get_model('htdemucs_6s')
if torch.cuda.is_available() and os.environ.get('FORCE_CPU') != '1':
    model.cuda()
elapsed = time.time() - t0
if elapsed < 3:
    print('    ✅ 已缓存，加载耗时 {:.1f}s'.format(elapsed))
else:
    print('    ✅ 下载完成，耗时 {:.1f}s'.format(elapsed))
" || echo "    ⚠️ Demucs 模型加载失败，后续运行时会自动重试"

# --- Basic Pitch（走 tfhub.dev） ---
echo "  [3/3] Basic Pitch 模型（约 60MB，走 tfhub.dev）..."
"$PYTHON_BIN" -c "
import numpy as np, soundfile as sf, tempfile, os, time
dummy = np.zeros((44100 * 2,), dtype=np.float32)
tmp = os.path.join(tempfile.gettempdir(), '_bp_dummy.wav')
sf.write(tmp, dummy, 44100)
t0 = time.time()
from basic_pitch.inference import predict, Model
from basic_pitch import ICASSP_2022_MODEL_PATH
model = Model(ICASSP_2022_MODEL_PATH)
predict(tmp, model)
os.remove(tmp)
elapsed = time.time() - t0
if elapsed < 3:
    print('    ✅ 已缓存，加载耗时 {:.1f}s'.format(elapsed))
else:
    print('    ✅ 下载完成，耗时 {:.1f}s'.format(elapsed))
" || echo "    ⚠️ Basic Pitch 模型加载失败，后续运行时会自动重试"

# --- MIDI-GPT 基础模型（yellow_medium，微调用） ---
MIDIGPT_DIR="$SCRIPT_DIR/models/midigpt"
mkdir -p "$MIDIGPT_DIR"
echo "  [4/4] MIDI-GPT 基础模型 yellow_medium（约 57MB，走 hf-mirror）..."
if [ -f "$MIDIGPT_DIR/yellow_medium-final.safetensors" ]; then
    echo "    ✅ 已存在: $MIDIGPT_DIR/yellow_medium-final.safetensors"
else
    HF_ENDPOINT=https://hf-mirror.com "$PYTHON_BIN" -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='Metacreation/MIDI-GPT',
    filename='yellow_medium-final.safetensors',
    local_dir='$MIDIGPT_DIR',
)
print('    ✅ MIDI-GPT 下载完成:', p)
" || echo "    ⚠️ MIDI-GPT 模型下载失败，微调前需手动下载"
fi

echo -e "  ${GREEN}✅ 模型准备完成${NC}"


# ============================================================
# === 阶段二：输入文件检测 ===
# ============================================================
echo ""
echo -e "${GREEN}[阶段二] 输入文件检测${NC}"
echo "----------------------------------------"
echo "  搜索目录: $INPUT_DIR"

# 安全处理：绝不删除 input/ 目录内容
if [ "$INPUT_DIR" = "$SCRIPT_DIR/input" ]; then
    # 默认情况：歌曲已放入 pipeline 的 input/（建议先用 prepare_input.py 去重）
    FLAT_INPUT="$SCRIPT_DIR/input"
    MP3_COUNT=$(find "$FLAT_INPUT" -maxdepth 1 -name "*.mp3" 2>/dev/null | wc -l)
    echo "  使用 input/ 目录，已找到 $MP3_COUNT 首 MP3（跳过过滤/复制）"
    if [ "$MP3_COUNT" -eq 0 ]; then
        echo -e "${RED}❌ input/ 目录下没有 MP3 文件！${NC}"
        echo "  提示: 1) 把 MP3 放入 input/；2) 或设置 INPUT_DIR 指向外部目录（脚本会自动筛选复制）"
        exit 1
    fi
else
    # 外部目录：递归查找 + 过滤（KAITO/唱见/CC字幕版），暂存到临时目录，不动 input/
    STAGING="$(mktemp -d)"
    FLAT_INPUT="$STAGING"
    trap 'rm -rf "$STAGING"' EXIT

    MP3_COUNT=$(find "$INPUT_DIR" -name "*.mp3" 2>/dev/null | wc -l)
    echo "  找到 $MP3_COUNT 个 MP3 文件（外部目录模式，将筛选后复制到临时目录）"

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

        # 复制到临时目录（用子目录名+文件名避免冲突）
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
fi

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
echo "  语言:     $LYRICS_LANG"
echo "  Whisper:  $WHISPER_MODEL (本地: $LOCAL_W_DIR)"
echo "  跳过鼓组: $SKIP_DRUMS"
echo ""

# 构建命令行参数
CMD_ARGS=(
    "batch"
    "$FLAT_INPUT/*.mp3"
    "--output" "$OUTPUT_DIR"
    "--language" "$LYRICS_LANG"
    "--whisper-model" "$WHISPER_MODEL"
    "--parallel" "$PARALLEL"
)

if [ "$SKIP_DRUMS" = "true" ]; then
    CMD_ARGS+=("--skip-drums")
fi
if [ "$SKIP_TO_STAGE" != "0" ]; then
    CMD_ARGS+=("--skip-to-stage" "$SKIP_TO_STAGE")
fi

echo "  执行命令:"
echo "  $PYTHON_BIN run_extraction.py ${CMD_ARGS[*]}"
echo ""

START_TIME=$(date +%s)

# 关闭 set -e，确保流水线失败后仍能执行阶段四的汇总
set +e
"$PYTHON_BIN" run_extraction.py "${CMD_ARGS[@]}"
EXIT_CODE=$?
set -e

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
        [ -d "$song_dir" ] || continue
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
