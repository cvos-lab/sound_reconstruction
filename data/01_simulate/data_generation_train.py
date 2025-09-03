import os, math, glob, random, yaml
from pathlib import Path

import numpy as np
from numpy.linalg import inv
from PIL import Image
import torch
import torchaudio
from scipy.ndimage import map_coordinates
import soundfile as sf
from tqdm import tqdm  # NEW: progress bar

# --------------------------
# helpers
# --------------------------
def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------
# audio I/O (robust)
# --------------------------
def to_mono_resample(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr_in != sr_out:
        wav = torchaudio.functional.resample(wav, sr_in, sr_out)
    return wav.squeeze(0)

def rms_normalize(wav: torch.Tensor, target_rms=0.1) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
    return wav * (target_rms / rms)

def _load_audio_any(path: Path):
    try:
        wav, sr = torchaudio.load(str(path))
        return wav, sr
    except Exception as e:
        ext = path.suffix.lower()
        try:
            if ext == ".mp3":
                from pydub import AudioSegment  # requires ffmpeg in PATH
                audio = AudioSegment.from_file(str(path), format="mp3")
                sr = audio.frame_rate
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels)).T.mean(axis=0)
                denom = float(1 << (8 * audio.sample_width - 1))
                mono = samples / denom
                wav = torch.from_numpy(mono).unsqueeze(0).contiguous()
                return wav, sr
            else:
                data, sr = sf.read(str(path), always_2d=True)
                wav = torch.from_numpy(data.T).float()
                return wav, sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio {path} with torchaudio and fallback. "
                               f"Original: {e}; Fallback: {e2}")

def load_train_audios(cfg: dict, root: Path):
    folder = root / cfg["audio"]["folder"]
    files = cfg["audio"]["train_files"]
    sr_out = int(cfg["simulation"]["canonical_sr"])
    norm_kind = cfg["audio"].get("norm", "rms-20db").lower()
    out = {}
    for fname in files:
        path = folder / fname
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        wav, sr = _load_audio_any(path)
        wav = to_mono_resample(wav, sr, sr_out)
        if norm_kind.startswith("rms"):
            wav = rms_normalize(wav, target_rms=0.1)
        out[fname] = wav
    return out, sr_out

# --------------------------
# calibration (your legacy constants)
# --------------------------
def build_calibration(calib_cfg: dict):
    alpha = calib_cfg["alpha"]; beta = calib_cfg["beta"]; gamma = calib_cfg["gamma"]
    u0 = calib_cfg["u0"]; v0 = calib_cfg["v0"]
    R = np.array(calib_cfg["R"], dtype=np.float64)
    T = np.array(calib_cfg["T"], dtype=np.float64).reshape(3, 1)

    A = np.array([[alpha, gamma, u0],
                  [0.0,   beta,  v0],
                  [0.0,   0.0,   1.0]], dtype=np.float64)

    Rinv = inv(R)
    Ainv = inv(A)
    K = Rinv @ Ainv
    q = Rinv @ T

    return A, R, T, K, q

# --------------------------
# fixed pattern loader (Speckle only)
# --------------------------
def load_fixed_pattern(pattern_folder: Path, pattern_file_or_token: str,
                       frame_h: int, frame_w: int):
    exts = ("*.png", "*.bmp", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(glob.glob(str(pattern_folder / e)))
    if not files:
        raise FileNotFoundError(f"No pattern images found in {pattern_folder}")

    target = None
    needle = pattern_file_or_token.casefold()
    for p in files:
        if os.path.basename(p).casefold() == needle:
            target = p
            break
    if target is None:
        for p in sorted(files):
            if needle in os.path.basename(p).casefold():
                target = p
                break
    if target is None:
        raise FileNotFoundError(f"Could not find '{pattern_file_or_token}' in {pattern_folder}")

    img = Image.open(target).convert("L")
    img_np = (np.asarray(img, dtype=np.float64) / 255.0)
    H0, W0 = img_np.shape

    if H0 < frame_h or W0 < frame_w:
        rep_y = math.ceil(frame_h / H0)
        rep_x = math.ceil(frame_w / W0)
        img_np = np.tile(img_np, (rep_y, rep_x))
    H1, W1 = img_np.shape
    y0 = max(0, (H1 - frame_h) // 2)
    x0 = max(0, (W1 - frame_w) // 2)
    img_np = img_np[y0:y0+frame_h, x0:x0+frame_w]
    return img_np, os.path.basename(target)

# --------------------------
# pinhole render (vectorized per-frame)
# --------------------------
def render_frames_from_audio(
    audio_1d: torch.Tensor,
    pattern_img: np.ndarray,
    K: np.ndarray,
    q: np.ndarray,
    magnification: float,
):
    n_frames = audio_1d.numel()
    H, W = pattern_img.shape
    frames = np.zeros((n_frames, H, W), dtype=np.float32)

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    P = np.stack([uu, vv, np.ones_like(uu)], axis=0).reshape(3, -1)
    Q = K @ P
    Qx, Qy, Qz = Q[0, :], Q[1, :], Q[2, :]
    qx, qy, qz = float(q[0, 0]), float(q[1, 0]), float(q[2, 0])

    for k in range(n_frames):
        Zw = float(audio_1d[k].item()) * float(magnification)
        Zc = (Zw + qz) / Qz
        Xw = Zc * Qx - qx
        Yw = Zc * Qy - qy
        ii = Xw + (W / 2.0)
        jj = Yw + (H / 2.0)
        samp = map_coordinates(
            pattern_img,
            coordinates=np.vstack([jj, ii]),  # (row=j, col=i)
            order=3, mode="reflect"
        )
        frames[k] = samp.reshape(H, W).astype(np.float32)

    return frames

# --------------------------
# trimming helper
# --------------------------
def maybe_trim(wav_1d: torch.Tensor, fname: str, cfg: dict) -> torch.Tensor:
    tcfg = cfg["audio"].get("trim", {})
    apply_to = set(tcfg.get("apply_to", []))
    fraction = float(tcfg.get("fraction", 1.0))
    if fname in apply_to and 0.0 < fraction < 1.0:
        n = int(round(wav_1d.numel() * fraction))
        n = max(1, min(n, wav_1d.numel()))
        return wav_1d[:n]
    return wav_1d

# --------------------------
# main
# --------------------------
def main():
    # --- Set project root ---
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent.parent

    # 1) cfg & seeds
    cfg_path = ROOT / "configs" / "01_simulate.yaml"
    cfg = load_cfg(str(cfg_path))
    seed_all(cfg.get("seed", 1337))

    # 2) load all audios (full length each, resampled+RMS-normalized)
    audios, sr = load_train_audios(cfg, ROOT)

    # 3) calibration
    _, _, _, K, q = build_calibration(cfg["simulation"]["calibration"])

    # 4) pattern (fixed Speckle)
    H, W = cfg["simulation"]["frame_size"]
    pat_folder = ROOT / cfg["simulation"]["pattern_folder"]
    pat_file = cfg["simulation"].get("pattern_file", "speckle.bmp")
    pattern_img, pattern_name = load_fixed_pattern(pat_folder, pat_file, H, W)

    # 5) output dirs
    out_root = ROOT / cfg["output"]["root"]
    frames_root = out_root / "frames_train"
    packaged_root = out_root / cfg["output"].get("packaged", "packaged")
    ensure_dir(frames_root)
    ensure_dir(packaged_root)

    magnif = float(cfg["simulation"].get("magnification", 10.0))
    print(f"[INFO] canonical_sr={sr} Hz | frame_size=({H},{W}) | pattern={pattern_name}")

    frames_list = []   # list of T_i x H x W (uint8)
    audio_list  = []   # list of T_i (float32)
    sources     = []   # filenames in order

    for fname in cfg["audio"]["train_files"]:
        wav_full = audios[fname]
        wav_used = maybe_trim(wav_full, fname, cfg)
        n_frames = wav_used.numel()
        trimmed_note = ""
        if wav_used.numel() != wav_full.numel():
            trimmed_note = f" (trimmed to {100.0 * n_frames / wav_full.numel():.1f}% of original)"
        print(f"[INFO] {fname}: {n_frames} frames ({n_frames/sr:.2f} s){trimmed_note}")

        # --- SR-suffixed base name ---
        base = f"{Path(fname).stem}_{sr}"

        # Save the EXACT audio used (resampled+normalized and trimmed) as WAV
        audio_wav_path = packaged_root / (base + ".wav")
        sf.write(str(audio_wav_path), wav_used.cpu().numpy(), sr)
        print(f"[OK] saved audio used to {audio_wav_path}")

        # Generate simulated frames from the audio USED
        frames_f32 = render_frames_from_audio(
            audio_1d=wav_used, pattern_img=pattern_img, K=K, q=q, magnification=magnif
        )
        # --- SR-suffixed frame folder ---
        frames_dir = frames_root / base
        ensure_dir(frames_dir)
        print(f"[INFO] Saving {n_frames} frames to {frames_dir} ...")
        for i in tqdm(range(n_frames), desc=f"Frames {base}", unit="img"):
            img = (frames_f32[i] * 255.0).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(frames_dir / f"frame_{i:06d}.bmp")
        print(f"[OK] wrote {n_frames} frames to {frames_dir}")

        frames_u8 = (frames_f32 * 255.0).round().clip(0, 255).astype(np.uint8)
        frames_list.append(torch.from_numpy(frames_u8))
        audio_list.append(wav_used.detach().cpu())
        sources.append(base)
        del frames_f32

    # --- SR-suffixed .pt ---
    combined_path = packaged_root / f"trainset_simulated_{sr}.pt"
    torch.save({
        "frames_list": frames_list,
        "audio_list": audio_list,
        "frame_size": (H, W),
        "sr": sr,
        "magnification": magnif,
        "sources": sources,
    }, combined_path)
    print(f"[OK] saved combined training set to {combined_path}")

    print("[DONE] Stage-1 simulation: frames (BMP, _sr) + combined .pt + wavs (_sr).")

if __name__ == "__main__":
    main()
