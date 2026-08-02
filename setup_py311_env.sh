#!/bin/bash
# ============================================================
#  魔塔（ModelScope Notebook）Python 3.11 环境搭建脚本
#
#  为什么需要 3.11？
#    basic-pitch 官方只支持 Python <=3.11，其依赖 tensorflow<2.15.1
#    在 Python 3.12 上无 wheel；且 requirements.txt 为 3.11 锁定
#    （numpy==1.24.3 / torch==2.1.0 等）。魔塔默认 Python 3.12，
#    直接装会失败，故单独创建 3.11 虚拟环境。
#
#  用法（魔塔终端）:
#    bash setup_py311_env.sh
#    之后每次跑流水线前先激活: source .venv311/bin/activate
#
#  检测顺序: 系统 python3.11 → conda → apt 安装 python3.11
# ============================================================
set -e

ENV_DIR=".venv311"

echo "=========================================="
echo "  🐍 魔塔 Python 3.11 环境搭建"
echo "=========================================="

# --- 找到 python3.11 ---
PY311=""
for cand in python3.11 /usr/bin/python3.11 /usr/local/bin/python3.11; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY311="$cand"
        echo "  ✅ 找到系统 python3.11: $cand ($("$cand" --version))"
        break
    fi
done

# --- conda 兜底 ---
if [ -z "$PY311" ] && command -v conda >/dev/null 2>&1; then
    echo "  未找到系统 python3.11，使用 conda 创建 py311 环境..."
    conda create -n midi311 python=3.11 -y
    PY311="$(conda info --base)/envs/midi311/bin/python"
    echo "  ✅ conda py311: $PY311"
fi

# --- apt 兜底（魔塔是 root，可装系统包） ---
if [ -z "$PY311" ]; then
    echo "  未找到 python3.11 / conda，尝试 apt 安装 python3.11..."
    apt-get update -qq && apt-get install -y -qq python3.11 python3.11-venv
    PY311="python3.11"
    echo "  ✅ apt 安装完成: $(python3.11 --version)"
fi

if [ -z "$PY311" ] || ! command -v "$PY311" >/dev/null 2>&1; then
    echo "  ❌ 无法获取 python3.11，请手动安装后重试。"
    exit 1
fi

# --- 创建虚拟环境 ---
echo ""
echo "[1/3] 创建虚拟环境 $ENV_DIR ..."
if [ ! -d "$ENV_DIR" ]; then
    "$PY311" -m venv "$ENV_DIR"
    echo "  ✅ 已创建 $ENV_DIR"
else
    echo "  ℹ️  $ENV_DIR 已存在，跳过创建"
fi

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "[2/3] 选择最快的 pip 源（魔塔内网优先，回退阿里云公网）..."
# 魔塔实例在阿里云内网，mirrors.cloud.aliyuncs.com 是内网专线（最快）；
# 本机/其他环境连不通时回退到阿里云公网镜像（实测 1.17MB/s，优于清华 1.13MB/s、
# 腾讯 0.87MB/s、官方 0.31MB/s）
if curl -sI -m 8 https://mirrors.cloud.aliyuncs.com/pypi/simple/ -o /dev/null -w "%{http_code}" 2>/dev/null | grep -qE "200|206"; then
    pip config set global.index-url https://mirrors.cloud.aliyuncs.com/pypi/simple/ 2>/dev/null || true
    pip config set global.trusted-host mirrors.cloud.aliyuncs.com 2>/dev/null || true
    echo "  ✅ 使用阿里云内网源（魔塔专线，最快）"
else
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null || true
    pip config set global.trusted-host mirrors.aliyun.com 2>/dev/null || true
    echo "  ✅ 使用阿里云公网源"
fi
pip install -U pip -q

echo "[3/3] 安装 requirements.txt（3.11 锁定版本，含 torch GPU 版）..."
echo "      这步耗时较长（torch 约 2GB+），请耐心等待..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  ✅ 环境搭建完成"
echo "=========================================="
echo "  Python: $(python --version)"
echo "  位置:   $ENV_DIR"
echo ""
echo "  以后跑流水线前先激活环境:"
echo "    source $ENV_DIR/bin/activate"
echo "  然后运行:"
echo "    SKIP_PREPARE=true bash modelscope_run.sh"
echo ""
echo "  验证 basic-pitch:"
echo "    python -c \"import basic_pitch; print('basic-pitch OK')\""
echo ""
echo "  ⚠️  注意: 每次新建魔塔终端（重启实例）都要重新 source 激活。"
echo "=========================================="
