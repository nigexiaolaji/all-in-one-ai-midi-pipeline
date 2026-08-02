# ALL-IN-ONE: AI MIDI Pipeline

End-to-end, mostly automatic pipeline for turning your own stereo masters into aligned, labeled multi-track MIDI suitable for model training.

**Stereo in → stems → tempo/meter → transcription → canonical tracks → (optional) key normalization → (optional) cleaning → multi-track MIDI**

> Cleaning is **ON by default**. Disable all MIDI cleaning/post-processing (including transcription-time filters and the final cleanup step) with `--no-clean`.

---

## Features (What It Does)

### 1. Stem Separation (HTDemucs 5-stem)

- Splits each track into:
  - `vocals`, `drums`, `bass`, `guitar`, `other`
- Outputs:
  - `data/stems/<Song>/...`
  - `manifests/<Song>.json`

---

### 2. Tempo, Downbeats, and Meter

- Uses `librosa` (and optionally `madmom` if installed) to estimate:
  - `meter_key.tempo`
  - downbeat positions
  - rough time signature
- Estimated tempo is reused downstream (e.g. as Basic Pitch `midi_tempo`).

---

### 3. Transcription

#### 3.1 Pitched (Basic Pitch 0.2.6)

Run on stems with tempo-aware settings.

Pitched transcription includes optional MIDI cleaning/post-processing (enabled by default). Disable with `--no-clean`.

##### Vocals (`vocals` stem)

- Basic Pitch → note events
- Vocal-specific tweaks (when cleaning is enabled):
  - higher onset/frame thresholds
  - minimum note length
  - merge same-pitch segments (reduce double hits)
  - squash tiny vibrato / slides
- Split into:
  - `voxlead` — highest active line
  - `voxbg` — background / harmonies

##### Bass

- Basic Pitch on `bass`
- Optional filtering to reduce obvious junk / octave errors (when cleaning is enabled)

##### Guitar

- Basic Pitch on `guitar`
- Exported with a guitar-like GM program

##### Other

- Basic Pitch on `other`
- Treated as pads/synths/etc. with a pad-like GM program

Pitched transcription status is stored under:

    "transcription": {
      "pitched": {
        "...": "..."
      }
    }

#### 3.2 Drums (ADTOF)

- `steps/transcribe_drums.py` uses `adtof_pytorch` on the `drums` stem
- Merges hits into a single `drums` kit
- Velocities derived from stem RMS (dynamic, not all-100)

Drum transcription status is stored under:

    "transcription": {
      "drums": {
        "...": "..."
      }
    }

---

### 4. Canonical Track Assignment

`steps/assign_parts.py` maps detected parts into consistent labels:

- `drums`
- `voxlead`
- `voxbg`
- `bass`
- `guitar`
- `keys` (optional)
- `other`

Only non-empty tracks are kept.

Recorded under:

    "assignment": {
      "tracks": {
        "...": "..."
      }
    }

---

### 5. Key Detection and Optional Normalization

`steps/key_normalize.py`:

- Detects a global key from pitched notes (ignoring drums) using `music21`.
- If enabled:
  - major-ish → transposed to **C major**
  - minor-ish → transposed to **A minor**

When enabled:

    "key": {
      "detected_tonic": "...",
      "detected_mode": "...",
      "normalized": true,
      "transpose_semitones": <int>,
      "target": "C major" | "A minor"
    }

When disabled:

    "key": {
      "detected_tonic": "...",
      "detected_mode": "...",
      "normalized": false,
      "transpose_semitones": 0,
      "target": null,
      "reason": "key normalization disabled via CLI"
    }

- Key normalization is **OFF by default**. Enable per run with `--normalize-key`.

---

### 6. Time Signature Injection (Optional)

`steps/meter_apply.py` can inject simple time signature meta events when meter estimation is confident.

---

### 7. Cleanup and Quantization

`steps/clean_quantize.py`:

- Removes obvious junk events
- Applies gentle timing/length cleanup
- Tries not to destroy groove/feel

Cleanup is **enabled by default**. Disable with `--no-clean` to bypass this step *and* transcription-time cleaning inside `steps/transcribe_melodic.py`.

When `--no-clean` is used, the manifest records:

    "pipeline_flags": {
      "no_clean": true
    },
    "cleanup": {
      "enabled": false,
      "reason": "cleanup disabled via CLI (--no-clean)"
    }

---

### 8. Multi-track MIDI Export

`steps/write_midi.py` builds, for each song:

- One multi-track MIDI file:
  - `data/midi/<Song>/<Song>.mid`
- Uses:
  - tempo from `meter_key.tempo`
  - one track per canonical class
  - `is_drum = True` for drums
  - track names = canonical labels

---

### 9. Human-In-The-Loop Hooks

- `python pipeline.py review-pending`
  - surfaces items flagged for human review
- `steps/qc_render.py`
  - optional utilities for quick audio/MIDI spot checks

---

## Install

Supported Python versions:

- **Python 3.11** — local development (Windows/Linux)
- **Python 3.12** — ModelScope Notebook (Ubuntu 22.04 + CUDA 12.8.1 + PyTorch 2.10, pre-installed)

> `requirements.txt` pins packages for the local Python 3.11 venv
> (`numpy==1.24.3` / `torch==2.1.0` have **no Python 3.12 wheels**).
> On ModelScope (Python 3.12) do **not** install it — run `bash auto_run.sh`
> or `bash setup_modelscope.sh` instead, which install unpinned packages and
> reuse the environment's pre-installed PyTorch 2.10.

    # 1) Local: create & activate venv (Python 3.11)
    python3.11 -m venv .venv-ai-midi
    source .venv-ai-midi/bin/activate

    # 2) Install dependencies
    pip install -r requirements.txt

Key dependencies (see `requirements.txt` for exact pins):

- Core: `numpy`, `typing-extensions`, `librosa`, `soundfile`, `scipy`, `pretty_midi`, `mido`
- Separation: `demucs>=4.0.0`
- Key detection: `music21`
- Transcription: `basic-pitch==0.2.6` (+ appropriate `tensorflow` for your platform)
- Drums: `adtof_pytorch`
- CLI / misc: `gradio`, `tqdm`, `pyyaml`
- Optional: `madmom` for extra beat/downbeat features

### Local Whisper Models

`steps/extract_lyrics.py` loads Whisper from the local `model/` folder next to the repo
(fallback: HuggingFace download, `HF_ENDPOINT` mirror supported):

    model/large/   # faster-whisper-large-v3  (model.bin + config.json + tokenizer.json + vocabulary.json)
    model/small/   # faster-whisper-small     (model.bin + config.json + tokenizer.json + vocabulary.txt)

- `--whisper-model large-v3` → `model/large/`（默认）
- `--whisper-model small` → `model/small/`
- 模型根目录可用环境变量 `WHISPER_MODEL_DIR` 覆盖

---

## Usage

### 1. Add Audio

    mkdir -p data/raw
    cp /path/to/YourSong.wav data/raw/

### 2. Run the Pipeline

---

## Select Which Tracks to Output (Optional)

You can limit the final multi-track MIDI to a subset of canonical tracks using `--tracks`.

Examples:

Drums + bass + guitar only:

```bash
python pipeline.py run-batch "data/raw/*.wav" --tracks drums,bass,guitar


Default (no key normalization, cleaning enabled):

    python pipeline.py run-batch "data/raw/*.wav"

With key normalization (C major / A minor):

    python pipeline.py run-batch "data/raw/*.wav" --normalize-key

Disable MIDI cleaning/post-processing (rawer transcription, skips final cleanup):

    python pipeline.py run-batch "data/raw/*.wav" --no-clean

With key normalization but no cleaning:

    python pipeline.py run-batch "data/raw/*.wav" --normalize-key --no-clean

### 3. Inspect Outputs

For `YourSong.wav`:

- Stems: `data/stems/YourSong/...`
- Manifest: `manifests/YourSong.json`
- MIDI: `data/midi/YourSong/YourSong.mid`

### 4. Extra Commands

See items flagged for human review:

    python pipeline.py review-pending

Export all final MIDIs to a flat folder:

    python pipeline.py export-midi --out out_midis/

---

## MIDI-GPT 训练/生成（可选）

在批处理跑完拿到多轨 MIDI 后，可以在这些 MIDI 上微调 [MIDI-GPT](https://github.com/Metacreation-Lab/MIDI-GPT)。
基础模型为官方 `yellow_medium`（HuggingFace 仓库 `Metacreation/MIDI-GPT`，约 57MB，走 hf-mirror）。

> 需要 Python 3.9+（本机 3.11 / 魔塔 3.12 均可）。训练依赖独立于主需求，见 `requirements-midigpt.txt`：
>
>     pip install -r requirements-midigpt.txt

### 1. 准备训练数据

从已处理的 MIDI（支持多轨，如自动批处理产出的 `output/**/*_merged.mid`）生成 GigaMIDI parquet：

    python midi_gpt/prepare_data.py --input "output/**/*_merged.mid" --output-dir data/midigpt

（Windows 下自动使用 `Score.from_bytes` 规避 C++ 读取器对日文路径的 bug。
默认切 15% 做验证集，`--no-valid` 可全部进训练集。）

### 2. 预训练 → 微调

`auto_run.sh` 会按需下载官方 `yellow_medium` 基础权重到 `models/midigpt/`，微调时直接复用。

```bash
python midi_gpt/trainer.py \
    --init-from models/midigpt/yellow_medium-final.safetensors \
    --train-data "data/midigpt/train.parquet" \
    --eval-data "data/midigpt/valid.parquet" \
    --output-dir checkpoints/midigpt/run_001
```

`num_workers` 固定为 0（C++ 解析器不支持 fork）。输出 `checkpoints/…/model_final.safetensors`。
魔塔 Notebook（24G 显存）建议 `--precision bf16 --batch-size 4 --grad-accum 8`。

### 3. 生成

从头生成 8 小节 4 轨节拍：

```bash
    python midi_gpt/generate.py scratch \
    --checkpoint checkpoints/midigpt/run_001/model_final.safetensors \
    --bars 8 --tracks 4 --out generated/song1.mid \
    --attrs '{"note_density": 5, "max_polyphony": 3}'
```

对已有 MIDI 的某一条轨做局部补全：

```bash
    python midi_gpt/generate.py infill \
        --checkpoint checkpoints/midigpt/run_001/model_final.safetensors \
        --midi "../output/歌名/歌名_vocals.mid" \
        --track 0 --bars 4 5 6 7 --out generated/filled.mid
```

查询当前模型支持的生成属性：`python midi_gpt/generate.py --list-attrs`。

### 4. 魔塔（ModelScope）部署

1. clone 本仓库，**不要**安装主 `requirements.txt`（Python 3.12 下无 wheel），只装 `pip install -r requirements-midigpt.txt`。
2. 把 `data/midigpt/*.parquet` 传到 Notebook。
3. 让 `auto_run.sh` 下载 `yellow_medium` 基础权重。
4. 跑 `trainer.py` 微调；微调权重不大（<1GB），用 `modelscope upload` 传回本地。
