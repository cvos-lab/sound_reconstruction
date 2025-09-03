import os, math, glob, random, yaml
from pathlib import Path

import numpy as np
from numpy.linalg import inv
from PIL import Image
import torch
import torchaudio
from scipy.ndimage import map_coordinates
import soundfile as sf
from tqdm import tqdm

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
                import soundfile as sf
                data, sr = sf.read(str(path), always_2d=True)
                wav = torch.from_numpy(data.T).float()
                return wav, sr
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio {path} with torchaudio and fallback. "
                               f"Original: {e}; Fallback: {e2}")

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

def load_fixed_pattern(pattern_folder: Path, pattern_file: str, frame_h: int, frame_w: int):
    exts = ("*.png", "*.bmp", "*.jpg", "*.jpeg")
    files = []
    for e in exts:
        files.extend(glob.glob(str(pattern_folder / e)))
    # Find matching file (case-insensitive, pattern_file can omit extension)
    needle = pattern_file.split(".")[0].lower()
    target = None
    for f in files:
        if needle in os.path.basename(f).lower():
            target = f
            break
    if target is None:
        raise FileNotFoundError(f"Pattern '{pattern_file}' not found in {pattern_folder}")
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

# --- mapping for test audio: how many seconds to use (None means "all")
test_audio_length_override = {
    "Audio_beethoven.wav": 10,
    "Audio_Geico_Original_4s.wav": None,
    "Audio_graduation.wav": 10,
    "Audio_lalaland.wav": 10,
    "Audio_mozarts15a_Original_9s.wav": None,
    "Audio_not-a-dream-whats-happening-to-place.mp3": None,
    "Audio_Sym40_Original_6s.wav": None,
}

def main():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent.parent

    # 1) cfg & seeds
    cfg_path = ROOT / "configs" / "01_simulate.yaml"
    cfg = load_cfg(str(cfg_path))
    seed_all(cfg.get("seed", 1337))

    holdout_files = cfg["audio"]["holdout_files"]
    sr = int(cfg["simulation"]["canonical_sr"])
    pattern_list = ["p_dollar", "p_label", "p_mac", "p_newspaper"]

    # Calibration
    _, _, _, K, q = build_calibration(cfg["simulation"]["calibration"])
    H, W = cfg["simulation"]["frame_size"]
    pat_folder = ROOT / cfg["simulation"]["pattern_folder"]

    out_root = ROOT / cfg["output"]["root"]
    frames_root = out_root / "frames_test"
    packaged_root = out_root / cfg["output"].get("packaged", "packaged")
    ensure_dir(frames_root)
    ensure_dir(packaged_root)
    magnif = float(cfg["simulation"].get("magnification", 10.0))

    frames_list = []
    audio_list = []
    sources = []

    # Main test generation loop
    for audio_file in holdout_files:
        audio_path = ROOT / cfg["audio"]["folder"] / audio_file
        if not audio_path.is_file():
            print(f"[WARN] Test audio file not found: {audio_path}")
            continue
        wav, orig_sr = _load_audio_any(audio_path)
        wav = to_mono_resample(wav, orig_sr, sr)
        wav = rms_normalize(wav)
        # Get length (seconds to use)
        limit_s = test_audio_length_override.get(audio_file, None)
        if limit_s is not None:
            n_frames = min(len(wav), int(sr * limit_s))
            wav = wav[:n_frames]
        else:
            n_frames = len(wav)
        for pattern_name in pattern_list:
            pattern_img, pattern_actual_name = load_fixed_pattern(
                pat_folder, pattern_name, H, W
            )
            base = f"{Path(audio_file).stem}__{Path(pattern_actual_name).stem}_{sr}"
            # Save WAV actually used
            audio_wav_path = packaged_root / (f"{Path(audio_file).stem}_{sr}.wav")
            sf.write(str(audio_wav_path), wav.cpu().numpy(), sr)
            # Simulate
            frames_f32 = render_frames_from_audio(
                audio_1d=wav, pattern_img=pattern_img, K=K, q=q, magnification=magnif
            )
            frames_dir = frames_root / base
            ensure_dir(frames_dir)
            print(f"[INFO] {base}: {n_frames} frames, pattern={pattern_actual_name}")
            for i in tqdm(range(n_frames), desc=f"Frames {base}", unit="img"):
                img = (frames_f32[i] * 255.0).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(img, mode="L").save(frames_dir / f"frame_{i:06d}.bmp")
            # Store for PT
            frames_u8 = (frames_f32 * 255.0).round().clip(0, 255).astype(np.uint8)
            frames_list.append(torch.from_numpy(frames_u8))
            audio_list.append(wav.detach().cpu())
            sources.append( (audio_file, pattern_actual_name, n_frames) )
            del frames_f32

    # Save combined .pt
    combined_path = packaged_root / f"testset_simulated_{sr}.pt"
    torch.save({
        "frames_list": frames_list,
        "audio_list": audio_list,
        "sources": sources,   # tuple (audio_name, pattern_name, n_frames)
        "frame_size": (H, W),
        "sr": sr,
        "magnification": magnif,
    }, combined_path)
    print(f"[OK] saved combined test set to {combined_path}")
    print("[DONE] Test set: frames (BMP, _sr) + combined .pt + wavs (_sr)")

if __name__ == "__main__":
    main()
