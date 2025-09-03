from pathlib import Path
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
import matplotlib.patches as patches
import matplotlib.animation as animation

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

def _load_wav_with_fallback(path: Path):
    try:
        wav, sr = torchaudio.load(str(path))
        print(f"[audio] Loaded WAV with torchaudio: {path.name} (sr={sr})")
        return wav, sr
    except Exception as e:
        print(f"[audio] torchaudio failed for WAV: {e}\n       Falling back to soundfile...")
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(data.T).float()
        print(f"[audio] Loaded WAV with soundfile: {path.name} (sr={sr})")
        return wav, sr

def _load_mp3_with_pydub(path: Path):
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub is not installed. Run: pip install pydub")
    try:
        audio = AudioSegment.from_file(path, format="mp3")
    except Exception as e:
        raise RuntimeError(
            "Failed to decode MP3 with pydub. Ensure FFmpeg is installed and on PATH.\n" + str(e)
        )
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).T
    else:
        samples = samples.reshape((1, -1))
    wav = torch.from_numpy(samples) / (1 << (8 * audio.sample_width - 1))
    print(f"[audio] Loaded MP3 with pydub: {path.name} (sr={sr})")
    return wav, sr

def load_audio_robust(path: Path):
    ext = path.suffix.lower()
    if ext == ".wav":
        return _load_wav_with_fallback(path)
    elif ext == ".mp3":
        return _load_mp3_with_pydub(path)
    else:
        raise ValueError(f"Unsupported audio format: {ext}")

def main():
    # ==== Config ====
    frame_h, frame_w = 128, 128
    length_s = 3.0
    magnif = 10.0

    HERE = Path(__file__).resolve()
    ROOT = HERE.parents[1]
    audio_dir = ROOT / "data" / "raw_audio"
    pattern_dir = ROOT / "data" / "raw_pattern"

    preferred = ["Audio_inwaves.wav", "Audio_great-now-weve-got-time-to-party.mp3", "Audio_FurElise_Original_6s.wav"]
    audio_path = None
    for nm in preferred:
        p = audio_dir / nm
        if p.exists():
            audio_path = p
            break
    if audio_path is None:
        candidates = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3")))
        if not candidates:
            raise FileNotFoundError(f"No .wav/.mp3 found in {audio_dir}")
        audio_path = candidates[0]

    wav, sr = load_audio_robust(audio_path)
    wav_48 = to_mono_resample(wav, sr, 4800)
    wav_48_norm = rms_normalize(wav_48)

    n_frames = int(length_s * 4800)
    if wav_48.numel() < n_frames:
        n_frames = wav_48.numel()
    clip = wav_48[:n_frames]
    clip_norm = wav_48_norm[:n_frames]
    t = np.arange(n_frames) / 4800.0

    # ---- Load pattern ----
    speckle_file = None
    for p in pattern_dir.iterdir():
        if p.is_file() and "speckle" in p.name.lower():
            speckle_file = p
            break
    if speckle_file is None:
        imgs = [*pattern_dir.glob("*.png"), *pattern_dir.glob("*.bmp"),
                *pattern_dir.glob("*.jpg"), *pattern_dir.glob("*.jpeg")]
        if not imgs:
            raise FileNotFoundError(f"No pattern images in {pattern_dir}")
        speckle_file = imgs[0]

    img = Image.open(speckle_file).convert("L")
    pattern = np.asarray(img, dtype=np.float32) / 255.0
    H, W = pattern.shape

    # ==== Camera calibration (real!) ====
    alpha = -1.4458188899075647e+03
    beta  = -1.4446376953530689e+03
    gamma = -1.0322752235572110e+00
    u0    = 2.2464793467423874e+02
    v0    = 1.7895763645710602e+02
    R = np.array([
        [ 9.9983911887424237e-01,  1.7848362129595435e-02, -1.7811058557933288e-03],
        [-1.7829189266673460e-02,  9.9978845369268321e-01,  1.0255138852622416e-02],
        [ 1.9637665013574530e-03, -1.0221733320932092e-02,  9.9994582842724233e-01]
    ], dtype=np.float64)
    T = np.array([[-8.8173675415097435e+01],
                  [-4.5772010598692937e+01],
                  [-1.2899667451947814e+03]], dtype=np.float64)

    A = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

    Rinv = np.linalg.inv(R)
    Ainv = np.linalg.inv(A)
    K = Rinv @ Ainv
    q = Rinv @ T

    # ---- Create output frame grid (128x128) ----
    uu, vv = np.meshgrid(np.arange(frame_w, dtype=np.float64),
                         np.arange(frame_h, dtype=np.float64))
    P = np.stack([uu, vv, np.ones_like(uu)], axis=0).reshape(3, -1)
    Q = K @ P
    Qx, Qy, Qz = Q[0], Q[1], Q[2]

    qx = q[0, 0].item()
    qy = q[1, 0].item()
    qz = q[2, 0].item()

    # Show 3 frames at various times
    idxs = [int(0.10 * n_frames), int(0.30 * n_frames), int(0.60 * n_frames)]
    frames = []
    roi_boxes = []
    roi_centers = []
    for idx in idxs:
        idx = max(0, min(idx, n_frames - 1))
        Zw = float(clip_norm[idx].item()) * magnif
        Zc = (Zw + qz) / Qz
        Xw = Zc * Qx - qx
        Yw = Zc * Qy - qy
        ii = Xw + W / 2.0
        jj = Yw + H / 2.0
        samp = map_coordinates(pattern, [jj, ii], order=3, mode="reflect")
        frames.append(samp.reshape(frame_h, frame_w))
        roi_ii = ii.reshape(frame_h, frame_w)
        roi_jj = jj.reshape(frame_h, frame_w)
        min_i, max_i = roi_ii.min(), roi_ii.max()
        min_j, max_j = roi_jj.min(), roi_jj.max()
        roi_boxes.append((min_i, max_i, min_j, max_j))
        roi_centers.append((roi_ii[frame_h//2, frame_w//2], roi_jj[frame_h//2, frame_w//2]))

    # ==== Plotting ====
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    # --- Top row as before ---
    axs[0, 0].plot(t, clip.numpy())
    axs[0, 0].set_title(f"Raw audio (first {n_frames/4800:.2f} s)\n{audio_path.name}")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(t, clip_norm.numpy())
    axs[0, 1].set_title("RMS-normalized audio (target RMS ≈ 0.1)")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True, alpha=0.3)

    # Pattern with ROI for frame 1
    min_i, max_i, min_j, max_j = roi_boxes[0]
    roi_cx, roi_cy = roi_centers[0]
    pat_cx, pat_cy = W/2, H/2
    axs[0, 2].imshow(pattern, cmap="gray", vmin=0, vmax=1)
    rect = patches.Rectangle(
        (min_i, min_j),
        max_i - min_i, max_j - min_j,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    axs[0, 2].add_patch(rect)
    axs[0, 2].plot([roi_cx], [roi_cy], marker='o', color='blue', markersize=8, label='ROI center')
    axs[0, 2].plot([pat_cx], [pat_cy], marker='x', color='orange', markersize=8, label='Pattern center')
    axs[0,2].arrow(
        pat_cx, pat_cy,
        roi_cx - pat_cx, roi_cy - pat_cy,
        head_width=20, head_length=20, fc='blue', ec='blue', alpha=0.6, length_includes_head=True
    )
    axs[0, 2].set_title(
        f"Base pattern (grayscale): {speckle_file.name}\n"
        "ROI for frame 1 (red), arrow = shift, blue dot = ROI center\n"
        f"Pattern shape = {H}×{W}"
    )
    axs[0, 2].legend(loc='upper right')
    axs[0, 2].axis("off")

    # --- Second row ---
    # Warped frame 1
    axs[1, 0].imshow(frames[0], cmap="gray", vmin=0, vmax=1)
    axs[1, 0].set_title(f"Warped frame 1\n t = {idxs[0]/4800:.3f} s")

    # Warped frame 2
    axs[1, 1].imshow(frames[1], cmap="gray", vmin=0, vmax=1)
    axs[1, 1].set_title(f"Warped frame 2\n t = {idxs[1]/4800:.3f} s")

    # Pixel flow plot (for frame 1)
    Zw = float(clip_norm[idxs[0]].item()) * magnif
    Zc = (Zw + qz) / Qz
    Xw = Zc * Qx - qx
    Yw = Zc * Qy - qy
    ii = Xw + W / 2.0
    jj = Yw + H / 2.0
    ii = ii.reshape(frame_h, frame_w)
    jj = jj.reshape(frame_h, frame_w)
    sample_points = [
        (0, 0),  # top-left
        (0, frame_w-1),  # top-right
        (frame_h-1, 0),  # bottom-left
        (frame_h-1, frame_w-1),  # bottom-right
        (frame_h//2, frame_w//2)  # center
    ]
    colors = ['red', 'blue', 'green', 'magenta', 'orange']
    axs[1, 2].imshow(pattern, cmap='gray', vmin=0, vmax=1)
    axs[1, 2].set_title("Pixel flow: Output→Pattern (frame 1)")
    for idx_pt, (y, x) in enumerate(sample_points):
        pattern_x = ii[y, x]
        pattern_y = jj[y, x]
        axs[1, 2].plot([pattern_x], [pattern_y], 'o', color=colors[idx_pt], markersize=10, label=f'Output ({y},{x})')
        output_x = (min_i + (max_i - min_i) * x / (frame_w - 1))
        output_y = (min_j + (max_j - min_j) * y / (frame_h - 1))
        axs[1, 2].arrow(output_x, output_y, pattern_x - output_x, pattern_y - output_y,
                        color=colors[idx_pt], length_includes_head=True, head_width=6, head_length=8, alpha=0.9)
    axs[1, 2].legend()
    axs[1, 2].axis("off")

    fig.suptitle("Pinhole Simulation — Camera ROI, Warp, and Pixel Flow", fontsize=16)
    fig.tight_layout()
    plt.show()

    # ==== Animation of motion/vibration with overlaid frame number ====
    n_anim_frames = 60
    frame_idxs = np.linspace(0, n_frames-1, num=n_anim_frames, dtype=int)
    anim_frames = []
    for idx in frame_idxs:
        Zw = float(clip_norm[idx].item()) * magnif
        Zc = (Zw + qz) / Qz
        Xw = Zc * Qx - qx
        Yw = Zc * Qy - qy
        ii = Xw + W / 2.0
        jj = Yw + H / 2.0
        samp = map_coordinates(pattern, [jj, ii], order=3, mode="reflect")
        anim_frames.append(samp.reshape(frame_h, frame_w))
    anim_frames = np.stack(anim_frames)

    fig_anim, ax_anim = plt.subplots(figsize=(5,5))
    im = ax_anim.imshow(anim_frames[0], cmap='gray', vmin=0, vmax=1)
    ax_anim.axis("off")
    frame_text = ax_anim.text(5, 15, f"Frame 1/{n_anim_frames}", color='w', fontsize=18, weight='bold',
                              bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    def update(frame_idx):
        im.set_array(anim_frames[frame_idx])
        frame_text.set_text(f"Frame {frame_idx+1}/{n_anim_frames}")
        return [im, frame_text]

    ani = animation.FuncAnimation(fig_anim, update, frames=n_anim_frames, interval=40, blit=True)
    plt.show()
    # Optionally save as GIF:
    # ani.save("speckle_motion.gif", writer="pillow", fps=25)

if __name__ == "__main__":
    main()
