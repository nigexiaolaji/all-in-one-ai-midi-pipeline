#!/bin/bash
# ============================================================
#  MIDI-GPT 微调一键脚本（魔塔 GPU 实例专用）
#  目标环境: PAI-DSW GPU（8核32GB 内存 / 24GB 显存）
#            预装镜像 ubuntu22.04-cuda12.8.1-py312-torch2.10.0-1.39.0
#
#  前置准备（魔塔 GPU 实例上）:
#    1. 代码:      /mnt/workspace/all-in-one-ai-midi-pipeline （clone 或上传）
#    2. 数据:      data/midi_gpt/train.parquet + valid.parquet
#                  （从本机 data/midi_gpt/ 上传，或用 modelscope 数据集）
#    3. 基础权重:  脚本会自动从 hf-mirror 下载 yellow_medium（若不存在）
#
#  用法:
#    bash auto_train_midigpt.sh                    # 默认 4000 步
#    MAX_STEPS=6000 bash auto_train_midigpt.sh     # 覆盖步数
#    BATCH_SIZE=2 bash auto_train_midigpt.sh       # 显存不足时降 batch
#    nohup bash auto_train_midigpt.sh > train.log 2>&1 &   # 后台训练
#
#  输出: checkpoints/midigpt/run_001/model_final.safetensors
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# ============ 可配置参数（环境变量覆盖） ============
MAX_STEPS="${MAX_STEPS:-2000}"        # 69 首数据 2000 步足够，且防过拟合；可覆盖
BATCH_SIZE="${BATCH_SIZE:-8}"         # A10 24G 显存充足；有效 batch = 8 x 4 = 32
GRAD_ACCUM="${GRAD_ACCUM:-4}"          # 有效 batch = 8 x 4 = 32
PRECISION="${PRECISION:-bf16}"
LR="${LR:-2e-5}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/midigpt/run_001}"
INIT_FROM="${INIT_FROM:-models/midigpt/yellow_medium-final.safetensors}"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}  🎵 MIDI-GPT 微调（魔塔 GPU 24G）${NC}"
echo -e "${BLUE}==========================================${NC}"

# ============ 1. 镜像配置 ============
echo ""
echo -e "${GREEN}[1/5] 镜像配置${NC}"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true
pip config set global.trusted-host mirrors.aliyun.com 2>/dev/null || true
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
echo -e "  ${GREEN}✅ aliyun(pip) + hf-mirror(HF)${NC}"

# ============ 2. GPU 检查 ============
echo ""
echo -e "${GREEN}[2/5] GPU 环境检查${NC}"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'GPU 不可用！'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# ============ 3. 依赖安装 ============
echo ""
echo -e "${GREEN}[3/5] 安装 midigpt 训练依赖${NC}"
if python3 -c "import midigpt" 2>/dev/null; then
    echo "  ✅ midigpt 已安装"
else
    echo "  安装 midigpt[train]（py312 wheel，走阿里云镜像）..."
    pip install -r requirements-midigpt.txt || pip install "midigpt[train]>=0.3.3"
fi

# ============ 4. 数据 & 基础权重检查 ============
echo ""
echo -e "${GREEN}[4/5] 数据与基础权重${NC}"
if [ ! -f data/midi_gpt/train.parquet ]; then
    echo -e "${RED}❌ 未找到 data/midi_gpt/train.parquet${NC}"
    echo "  请先把本机 data/midi_gpt/*.parquet 上传到:"
    echo "  /mnt/workspace/all-in-one-ai-midi-pipeline/data/midi_gpt/"
    exit 1
fi
python3 -c "
import pyarrow.parquet as pq
for n in ['train.parquet', 'valid.parquet']:
    t = pq.read_table(f'data/midi_gpt/{n}')
    print(f'  {n}: {t.num_rows} 首')
"

MIDIGPT_DIR="$(dirname "$INIT_FROM")"
mkdir -p "$MIDIGPT_DIR"
if [ -f "$INIT_FROM" ]; then
    echo "  ✅ 基础权重已存在: $INIT_FROM"
else
    echo "  下载 yellow_medium 基础权重（约 57MB，走 hf-mirror）..."
    python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id='Metacreation/MIDI-GPT',
    filename='yellow_medium-final.safetensors',
    local_dir='$MIDIGPT_DIR',
)
print('  ✅ 下载完成:', p)
"
fi

# ============ 5. 训练 ============
echo ""
echo -e "${GREEN}[5/5] 开始训练（${MAX_STEPS} 步，batch ${BATCH_SIZE}×${GRAD_ACCUM}，${PRECISION}）${NC}"
echo "  输出目录: $OUTPUT_DIR"
echo ""

python3 midi_gpt/trainer.py \
    --init-from "$INIT_FROM" \
    --train-data "data/midi_gpt/train.parquet" \
    --eval-data "data/midi_gpt/valid.parquet" \
    --output-dir "$OUTPUT_DIR" \
    --config midi_gpt/train_config.yaml \
    --max-steps "$MAX_STEPS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --precision "$PRECISION" \
    --eval-steps 100 \
    --save-steps 500

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}  ✅ 训练完成！${NC}"
echo "  权重: $OUTPUT_DIR/model_final.safetensors"
echo ""
echo "  回传本机（可选）:"
echo "    modelscope upload --model LinNew233/midi-gpt-finetuned $OUTPUT_DIR --commit-message \"run_001\""
echo -e "${GREEN}==========================================${NC}"
