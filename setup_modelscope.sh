#!/bin/bash
# ============================================================
#  ModelScope Notebook 环境安装脚本（仅安装依赖 & 模型）
#
#  💡 推荐直接使用 auto_run.sh，它包含本脚本的全部功能 + 自动处理
#     用法: bash auto_run.sh
#
#  本脚本仅用于单独安装环境，不运行流水线。
#
#  目标环境: Ubuntu 22.04 + CUDA 12.8.1 + Python 3.12 + PyTorch 2.10.0
#  24GB 显存 GPU
#
#  所有外网下载均走国内镜像，无需代理
# ============================================================
set -e

echo "=========================================="
echo "  AI MIDI Pipeline — ModelScope 环境安装"
echo "  全镜像加速，无需科学上网"
echo ""
echo "  💡 提示：推荐直接用 auto_run.sh 一键完成所有操作"
echo "=========================================="

# ============================================
# === 1. 所有镜像源配置 ===
# ============================================

# pip 阿里云镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

# HuggingFace 镜像（Whisper 模型下载）
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

# PyTorch 清华镜像（如果 notebook 未预装 PyTorch）
# pip config set global.extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu121

echo "✅ 镜像源配置完成：aliyun(pip) + hf-mirror(HF)"


# ============================================
# === 2. 安装核心依赖 ===
# ============================================
echo ""
echo "[1/6] 安装核心依赖..."

pip install --upgrade pip

# 辅助工具（先装，后续依赖可能用到）
pip install numpy pyyaml tqdm soundfile

# faster-whisper 歌词提取（GPU 版，走 HF 镜像）
pip install faster-whisper

# Demucs 音源分离（GPU 版，模型走 FB 官方 CDN）
pip install demucs

# Basic Pitch MIDI 转录（模型走 tfhub.dev，国内可能慢）
pip install basic-pitch

# MIDI 处理
pip install pretty_midi

echo ""
echo "✅ 核心依赖安装完成"


# ============================================
# === 3. 验证 GPU 环境 ===
# ============================================
echo ""
echo "[2/6] 验证 GPU 环境..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  VRAM: {vram:.1f} GB')
else:
    print('  ❌ GPU 不可用，请检查环境！')
    exit(1)
"


# ============================================
# === 4. 预下载 Whisper large-v3 模型（走 HF 镜像）===
# ============================================
echo ""
echo "[3/6] 预下载 Whisper large-v3 模型（约 3GB，走 hf-mirror）..."
python3 -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from faster_whisper import WhisperModel
model = WhisperModel(
    'large-v3',
    device='cuda',
    compute_type='float16',
    download_root='/root/.cache/huggingface',
)
print('✅ Whisper large-v3 模型下载完成')
"


# ============================================
# === 5. 预下载 Demucs 模型 ===
# ============================================
echo ""
echo "[4/6] 预下载 Demucs htdemucs_6s 模型（约 320MB，走 FB CDN）..."
echo "  注意：FB CDN 国内可能较慢，如失败可设代理后重试"
python3 -c "
import torch
# 触发 Demucs 模型下载（不实际分离音频）
from demucs import pretrained
model = pretrained.get_model('htdemucs_6s')
if torch.cuda.is_available():
    model.cuda()
print('✅ Demucs htdemucs_6s 模型下载完成')
"


# ============================================
# === 6. 预下载 Basic Pitch 模型（走 tfhub.dev）===
# ============================================
echo ""
echo "[5/6] 预下载 Basic Pitch 模型（约 60MB，走 tfhub.dev）..."
echo "  注意：tfhub.dev 国内可能较慢，首次推理会自动下载"
python3 -c "
import numpy as np
import soundfile as sf
import tempfile, os

# 生成 2 秒静音作为 dummy 输入，触发模型下载
dummy_audio = np.zeros((44100 * 2,), dtype=np.float32)
tmp_path = os.path.join(tempfile.gettempdir(), '_bp_dummy.wav')
sf.write(tmp_path, dummy_audio, 44100)

from basic_pitch.inference import predict
predict(tmp_path)
os.remove(tmp_path)
print('✅ Basic Pitch 模型下载完成')
"


# ============================================
# === 7. 最终验证 ===
# ============================================
echo ""
echo "[6/6] 最终验证..."

python3 -c "
import torch
import tensorflow as tf

print(f'  PyTorch:     {torch.__version__}')
print(f'  TensorFlow:  {tf.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
print(f'  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

# 验证所有关键模块可导入
import demucs.pretrained
import faster_whisper
import basic_pitch
import pretty_midi
print('  ✅ 所有模块导入成功')
"


echo ""
echo "=========================================="
echo "  环境安装完成！"
echo "=========================================="
echo ""
echo "所有模型已预下载到本地，后续运行无需联网。"
echo ""
echo "🚀 一键启动全自动处理："
echo "  bash auto_run.sh"
echo ""
echo "或手动启动批量处理："
echo "  python3 run_extraction.py batch \\"
echo "    \"input/*.mp3\" \\"
echo "    --output ../output \\"
echo "    --language ja \\"
echo "    --skip-drums \\"
echo "    --parallel 3"
echo ""