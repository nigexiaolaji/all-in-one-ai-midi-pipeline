import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from utils.manifest import song_id_from_path


def _merge_audio(a_path, b_path, out_path):
    """
    Sum two mono/stereo wavs, normalize if needed, write to out_path.
    Returns str(out_path).
    """
    if not a_path and not b_path:
        return None
    if a_path and not b_path:
        return a_path
    if b_path and not a_path:
        return b_path

    a, sr_a = sf.read(a_path)
    b, sr_b = sf.read(b_path)
    if sr_a != sr_b:
        raise RuntimeError(f"Sample rate mismatch: {sr_a} vs {sr_b}")

    # to mono-compatible shapes
    if a.ndim > 1:
        a = a.mean(axis=1)
    if b.ndim > 1:
        b = b.mean(axis=1)

    L = max(len(a), len(b))
    a = np.pad(a, (0, L - len(a)))
    b = np.pad(b, (0, L - len(b)))

    mix = a + b
    maxv = float(np.max(np.abs(mix))) if mix.size else 0.0
    if maxv > 1.0:
        mix = mix / maxv

    sf.write(out_path, mix, sr_a)
    return str(out_path)


def separate_track(audio_path: str, CFG: dict, manifest: dict):
    """
    Use Demucs 6-stem under the hood, but expose a 5-stem layout:

        vocals, drums, bass, guitar, other

    where:
      - guitar = Demucs guitar
      - other  = (Demucs other + Demucs piano), i.e. all non-core pitched stuff
    """
    sid = song_id_from_path(audio_path)

    base_out_dir = Path("data/stems")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    sep_cfg = CFG.get("separation", {})
    model_name = sep_cfg.get("demucs_model", "htdemucs_6s")

    # === GPU 检测：根据硬件自动切换参数策略 ===
    gpu_available = _has_gpu()
    if gpu_available:
        # GPU 满血版：默认参数 = 最高质量，24GB 显存轻松驾驭
        shifts = sep_cfg.get("shifts", 5)       # 5 次偏移取平均，最佳分离质量
        segment = sep_cfg.get("segment", 7.8)   # 默认段长，充足上下文
        overlap = sep_cfg.get("overlap", 0.25)  # 默认交叠，边界平滑
    else:
        # CPU 降级版：激进参数换速度
        shifts = sep_cfg.get("shifts", 1)       # 1 次偏移，~5x 加速
        segment = sep_cfg.get("segment", 4)     # 4 秒段长，~2x 加速
        overlap = sep_cfg.get("overlap", 0.1)   # 低交叠，减少计算

    # Demucs writes: data/stems/<model_name>/<sid>/*.wav
    song_out_dir = base_out_dir / model_name / sid

    if not (song_out_dir.exists() and any(song_out_dir.glob("*.wav"))):
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n", model_name,
            "--shifts", str(shifts),
            "--segment", str(int(round(segment))),   # demucs>=4 的 --segment 只接受整数秒
            "--overlap", str(overlap),
            "-o", str(base_out_dir),
            audio_path,
        ]
        if gpu_available:
            # `-d cuda` 必须跟在 demucs.separate 之后（index 3），
            # 插在 `-m` 和模块名之间会被 python 解释器当作自身参数
            cmd.insert(3, "-d")
            cmd.insert(4, "cuda")
            print(f"[separate] GPU 满血版 shifts={shifts} segment={segment} overlap={overlap}")
        else:
            print(f"[separate] CPU 节能版 shifts={shifts} segment={segment} overlap={overlap} (~10x 加速)")
        print(f"[separate] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    if not song_out_dir.exists():
        raise RuntimeError(f"[separate] Expected stems in {song_out_dir}, but folder is missing.")

    def pick(name: str):
        p = song_out_dir / f"{name}.wav"
        return str(p) if p.exists() else None

    vocals = pick("vocals")
    drums = pick("drums")
    bass = pick("bass")
    guitar = pick("guitar")
    piano = pick("piano")
    other_raw = pick("other")

    # Merge piano into other so we don't treat Demucs "piano" as a separate synth stem.
    merged_other_path = song_out_dir / "other_merged.wav"
    other = _merge_audio(other_raw, piano, merged_other_path)

    stems = {
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "guitar": guitar,
        "other": other,
    }

    # Write to manifest
    manifest.setdefault("separation", {})
    manifest["separation"]["model"] = model_name
    manifest["separation"]["path"] = str(song_out_dir)
    manifest["separation"]["stems"] = {k: v for k, v in stems.items() if v}

    print(f"[separate] 5-stem view for {sid}: {manifest['separation']['stems']}")

    return stems


def _has_gpu() -> bool:
    """
    检测是否有可用的 CUDA GPU。

    @return: 是否有 GPU 可用
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
