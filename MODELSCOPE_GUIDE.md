# 魔塔（ModelScope Notebook）完整操作指南

> 适用场景：本机 `programs/` 已备齐 MP3（108 首），准备上魔塔跑完整流水线
> （分离 → 转录 → 歌词 → 对齐 → 多轨 MIDI），再把结果下载回本机。

---

## 0. 总览：要传什么、在哪儿跑

| 内容 | 大小 | 上传方式 | 到魔塔后位置 |
|---|---|---|---|
| 代码（本仓库） | ~几 MB | git clone 或 zip 拖拽 | `/mnt/workspace/all-in-one-ai-midi-pipeline/` |
| MP3（`programs/` 全部） | ~几百 MB | **modelscope 私有数据集**（推荐） | Notebook 内 `modelscope download` |
| Whisper 模型（`model/`） | 3.4 GB | **不用传**——魔塔内网直接下载，走阿里云内网很快 | 自动到 `/root/.cache/huggingface` |
| Demucs / Basic Pitch / MIDI-GPT 权重 | ~450 MB | **不用传**——`auto_run.sh` 自动下载 | 缓存目录 |

> 💡 **为什么模型不用传**：魔塔实例在阿里云内网，`auto_run.sh` 阶段一
> 会自动从 ModelScope 内网拉取 `Systran/faster-whisper-large-v3`（比传 3.4GB 快得多），
> Demucs / Basic Pitch / MIDI-GPT 权重也会自动下载。
> 只有当你想要**离线/断网**跑时才需要把 `model/` 一起传上去。

---

## 1. 上传代码（两种方式任选）

### 方式 A：git clone（推荐，保持和 GitHub 同步）

本机仓库 remote 已配好 gh-proxy 代理（含 token），魔塔里直接 clone：

```bash
cd /mnt/workspace
git clone https://ghp_你的token@v4.gh-proxy.org/https://github.com/nigexiaolaji/all-in-one-ai-midi-pipeline.git
```

或直接查看本机配置的完整 URL：

```bash
git -C D:/MIDI/all-in-one-ai-midi-pipeline remote get-url origin
```

> ⚠️ 那个 URL 里带着访问令牌，在魔塔终端粘贴时注意别泄露到公开场合。
> 克隆后如需更新代码：`cd all-in-one-ai-midi-pipeline && git pull`。

### 方式 B：zip 打包拖拽（最简单，一次性）

本机把仓库打成 zip（**排除** .venv / data / 模型等大目录）：

```bash
cd D:/MIDI/all-in-one-ai-midi-pipeline
zip -r ../pipeline.zip . -x ".venv/*" -x "data/*" -x "models/*" -x ".git/*" -x "__pycache__/*"
```

然后进魔塔 Notebook → 左侧「文件」面板 → 拖拽上传 `pipeline.zip` → 终端里解压：

```bash
cd /mnt/workspace
unzip pipeline.zip -d all-in-one-ai-midi-pipeline
```

---

## 2. 上传 MP3（programs/ 108 首）到魔塔私有数据集

> 用 git 传几百 MB 是最差的方案（GitHub 单文件限 100MB、仓库膨胀、clone 慢）。
> 推荐走 **ModelScope 私有数据集**：内网传输、无大小限制、私有不公开。

### 步骤 2.1 本机安装 modelscope 并登录（一次性）

```bash
pip install modelscope -U
modelscope login      # 输入魔塔账号的访问令牌（ModelScope 官网 → 个人中心 → 访问令牌）
```

### 步骤 2.2 本机上传 programs/ 到私有数据集

```bash
# 首次上传整个目录（会先自动创建私有数据集 LinNew233/vocaloid-mp3）
modelscope upload --dataset LinNew233/vocaloid-mp3 D:/MIDI/programs --commit "v1: 108首"
```

> 上传前可以先在 ModelScope 网页端确认数据集已创建为**私有**。
> 之后新增/修改歌曲：同样的命令再跑一次即可增量更新。

### 步骤 2.3 魔塔 Notebook 里下载

```bash
cd /mnt/workspace
modelscope login      # 在魔塔里也要登录一次（终端会提示粘贴令牌）
modelscope download --dataset LinNew233/vocaloid-mp3 --local_dir /mnt/workspace/programs
```

---

## 3. 处理：跑流水线

### 方式 A：一键脚本（推荐，包含去重）

```bash
cd /mnt/workspace/all-in-one-ai-midi-pipeline
bash modelscope_run.sh
```

做的事：
1. `prepare_input.py` 从 `/mnt/workspace/programs` 去重筛选 → `input/`
   （按歌曲名分组，优先 CC字幕/完整版，排除唱见/不全）
2. 交给 `auto_run.sh`：装依赖 → 下载模型 → 跑 batch 流水线 → 结果汇总

> 如果想去掉去重、**108 首全量处理**（不筛版本）：
> 先 `cp /mnt/workspace/programs/*.mp3 input/`（含子目录则用 `find ... -exec cp`），
> 再 `SKIP_PREPARE=true bash modelscope_run.sh`

### 方式 B：直接跑 auto_run.sh（完全默认参数）

```bash
cd /mnt/workspace/all-in-one-ai-midi-pipeline
INPUT_DIR=/mnt/workspace/programs bash auto_run.sh
```

常用参数（环境变量覆盖）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INPUT_DIR` | `pipeline/input` | MP3 目录（外部目录会自动过滤 KAITO/唱见/CC字幕版） |
| `OUTPUT_DIR` | `/mnt/workspace/output` | 结果输出目录 |
| `PARALLEL` | `10` | 并行歌曲数（24G 显存建议 2-3，CPU 可 10+） |
| `WHISPER_MODEL` | `large-v3` | 可换 `small`（更快、准确率略降） |
| `SKIP_DRUMS` | `true` | 跳过鼓转录（加速；想转录鼓组设 `false`） |
| `LANGUAGE` | `ja` | 歌词语言 |

跑完后结果在 `/mnt/workspace/output/<歌名>/`（每首歌：`_lyrics.txt`、`_lyrics_timed.json`、
`_alignment.json`、`_merged.mid`、`_vocals.mid`、`_summary.txt`）。

> 中断续跑：流水线按 manifest 断点续跑，重新执行同一命令即可从失败处继续。
> 只想补歌词（跳过分离/转录）：`SKIP_TO_STAGE=4 bash auto_run.sh`。

---

## 4. 下载结果回本机

### 方式 A：modelscope 私有数据集（推荐，大结果用这个）

```bash
# 魔塔 Notebook 里：把 output/ 打包上传到数据集
cd /mnt/workspace
modelscope upload --dataset LinNew233/vocaloid-output output --commit "v1: 处理结果"

# 本机下载
modelscope download --dataset LinNew233/vocaloid-output --local_dir D:/MIDI/output
```

### 方式 B：zip 打包后从文件面板下载（小结果/一次性）

```bash
cd /mnt/workspace
zip -r output.zip output/
```

然后在魔塔 Notebook 左侧「文件」面板找到 `output.zip` → 右键/勾选 → 下载。

### 方式 C：微调权重传回（<1GB，用模型仓库）

```bash
modelscope upload --model LinNew233/midi-gpt-finetuned checkpoints/midigpt/run_001 --commit "v1"
# 本机下载
modelscope download --model LinNew233/midi-gpt-finetuned --local_dir checkpoints/midigpt/run_001
```

---

## 5. 完整命令速查表

```bash
# ===== 本机（Windows） =====
pip install modelscope -U && modelscope login
modelscope upload --dataset LinNew233/vocaloid-mp3 D:/MIDI/programs --commit "v1"

# ===== 魔塔 Notebook =====
modelscope login
git clone <remote-url>        # 或拖拽 zip 解压
modelscope download --dataset LinNew233/vocaloid-mp3 --local_dir /mnt/workspace/programs
cd all-in-one-ai-midi-pipeline && bash modelscope_run.sh
modelscope upload --dataset LinNew233/vocaloid-output /mnt/workspace/output --commit "v1"

# ===== 本机下载结果 =====
modelscope download --dataset LinNew233/vocaloid-output --local_dir D:/MIDI/output
```

---

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| `modelscope login` 报错 | 令牌过期，去 ModelScope 官网重新生成访问令牌 |
| 数据集上传慢 | 首次几百 MB 正常；内网下载时快，上传仍走公网，耐心等 |
| 魔塔无 GPU / 显存不足 | `PARALLEL=2` 降低并行；`WHISPER_MODEL=small` 省显存 |
| basic-pitch 编译失败 | 先 `pip install Cython wheel` 再重试，或看 `/tmp/basic_pitch_install.log` |
| 歌曲处理失败 | 看 `output/batch_report.txt` / `failed_songs.json`，单个失败不影响其他 |
| 想全量不筛版本 | 见第 3 节方式 A 的备注（`SKIP_PREPARE=true` + 手动复制 MP3 到 input/） |
