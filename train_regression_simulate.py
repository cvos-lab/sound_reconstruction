import os
from datetime import datetime

# ---- Max reproducibility: set BEFORE torch import ----
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import random
import numpy as np
import torch

def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

seed = 1337
seed_everything(seed)

import yaml
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
from models.convnextv2_tiny import ConvNeXtV2TinyRegressor
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.io_utils import get_gpu_temperatures

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# =========================================================
# Dataset keeps per-clip structure + ranges (for no-leakage)
# =========================================================
class FrameToAudioDataset(Dataset):
    """
    Loads a packaged .pt where frames/audio are stored as lists per clip.
    Exposes flattened tensors for __getitem__, and also keeps per-clip
    lengths/ranges so we can split by clip (no leakage).
    """
    def __init__(self, pt_path):
        pkg = torch.load(pt_path, map_location="cpu")

        # Keep per-clip lists (uint8 frames in [0..255], audio float)
        self.frames_list = [f.float() / 255.0 for f in pkg["frames_list"]]  # list of [T_i, H, W]
        self.audios_list = [a.view(-1)         for a in pkg["audio_list"]]   # list of [T_i]

        # Per-clip lengths and global offsets
        self.clip_lengths = [int(f.shape[0]) for f in self.frames_list]
        self.offsets = np.cumsum([0] + self.clip_lengths[:-1]).astype(int)  # [0, T1, T1+T2, ...]

        # Flatten for normal __getitem__
        self.frames = torch.cat(self.frames_list, dim=0)  # [sum T_i, H, W]
        self.audios = torch.cat(self.audios_list, dim=0)  # [sum T_i]
        self.N = int(self.frames.shape[0])

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.frames[idx][None, ...], self.audios[idx]

    def get_clip_ranges(self):
        """
        Returns list of (start_idx, end_idx) for each clip in the flattened view.
        """
        ranges = []
        for off, L in zip(self.offsets, self.clip_lengths):
            ranges.append((int(off), int(off + L)))
        return ranges

def compute_mae(pred, target):
    return (pred - target).abs().mean().item()

def compute_r2(pred, target):
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    return float(1 - (ss_res / (ss_tot + 1e-8)))

def compute_snr_db(pred, target):
    signal_power = (target ** 2).mean()
    noise_power = ((target - pred) ** 2).mean() + 1e-12
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def plot_losses(train_losses, val_losses, best_epoch, best_val_loss, ckpt_dir, model_name="convnextv2_tiny"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_losses, label="Train MSE")
    ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name.upper()} Loss Curve (Audio Regression)")
    ax.legend()
    ax.grid(True)

    # ---- Zoom around validation losses with extra padding ----
    val_min = float(min(val_losses))
    val_max = float(max(val_losses))
    margin = (val_max - val_min) * 0.5  # larger padding
    if margin == 0:                       # handle perfectly flat curves
        margin = max(val_max, 1e-8) * 0.1
    ax.set_ylim(val_min - margin, val_max + margin)

    if best_epoch >= 0:
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.05
        ax.annotate(
            f"{best_val_loss:.6f}",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch, best_val_loss + y_offset),
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            fontsize=12,
            color="red",
            weight="bold",
            ha="center"
        )

    # ---- Save instead of show ----
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"{model_name.lower()}_loss.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

# --------------------------------------------
# Target-scaling aware training/validation step
# --------------------------------------------
def run_epoch(model, loader, loss_fn, optimizer, device,
              is_train=True, grad_clip=0.0,
              mu=None, std=None, use_target_scaling=False):
    """
    If use_target_scaling: compute loss on normalized targets/preds.
    Metrics are computed on *denormalized* predictions for readability.
    """
    model.train() if is_train else model.eval()
    total_loss, total_mae, total_r2, total_snr = 0.0, 0.0, 0.0, 0.0
    count = 0
    loop = tqdm(loader, disable=not is_train, desc="Train" if is_train else "Val", ncols=150)

    for x, y in loop:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            if use_target_scaling:
                y_n = (y - mu) / std
                y_pred_n = model(x).squeeze()
                loss = loss_fn(y_pred_n, y_n)
                y_pred = y_pred_n * std + mu  # for metrics
            else:
                y_pred = model(x).squeeze()
                loss = loss_fn(y_pred, y)

            mae = compute_mae(y_pred, y)
            r2 = compute_r2(y_pred, y)
            snr = compute_snr_db(y_pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            total_mae  += mae * bsz
            total_r2   += r2 * bsz
            total_snr  += snr * bsz
            count      += bsz

            loop.set_postfix(loss=f"{loss.item():.6f}", mae=f"{mae:.6f}", r2=f"{r2:.5f}", snr=f"{snr:.4f}dB")

    avg_loss = total_loss / count
    avg_mae  = total_mae  / count
    avg_r2   = total_r2   / count
    avg_snr  = total_snr  / count
    return avg_loss, avg_mae, avg_r2, avg_snr

def main():
    cfg = load_yaml("configs/regression_simulate.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Flexible experiment type ("simulate" or "practical") ----
    experiment_type = cfg.get("experiment_type", "simulate")  # e.g. "simulate" or "practical"
    run_date = datetime.now().strftime("%Y%m%d")
    run_tag = cfg["output"].get("run_tag", "") or run_date
    model_name = cfg["model"]["backbone"]

    # ---- Folder: checkpoints/simulate/convnextv2_tiny_YYYYMMDD/
    ckpt_dir = Path(cfg["output"]["root"]) / experiment_type / f"{model_name}_{run_tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"===> Saving all checkpoints/logs to: {ckpt_dir}")

    # =========================================================
    # Dataset + group-wise split by clip (no leakage)
    # =========================================================
    ds = FrameToAudioDataset(cfg["data"]["pt_path"])
    clip_ranges = ds.get_clip_ranges()                     # [(s0,e0), (s1,e1), ...]
    num_clips = len(clip_ranges)
    total_frames = sum(e - s for (s, e) in clip_ranges)

    val_ratio = float(cfg["train"].get("val_ratio", 0.2))
    target_val_frames = int(round(total_frames * val_ratio))

    # reproducible shuffle of clip ids
    g = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(num_clips, generator=g).tolist()

    # take whole clips into val until we reach ~target frames
    val_clip_ids, acc = [], 0
    for ci in perm:
        s, e = clip_ranges[ci]
        val_clip_ids.append(ci)
        acc += (e - s)
        if acc >= target_val_frames:
            break
    val_clip_ids = set(val_clip_ids)

    # expand to frame indices
    train_indices, val_indices = [], []
    for ci, (s, e) in enumerate(clip_ranges):
        if ci in val_clip_ids:
            val_indices.extend(range(s, e))
        else:
            train_indices.extend(range(s, e))

    # sanity checks
    assert set(train_indices).isdisjoint(set(val_indices)), "Leakage: frame index overlap!"
    print(f"Split: {len(train_indices)} train frames | {len(val_indices)} val frames "
          f"({len(val_indices) / (len(train_indices)+len(val_indices)):.1%} val) "
          f"from {num_clips} clips (val clips: {len(val_clip_ids)})")

    train_ds = Subset(ds, train_indices)
    val_ds   = Subset(ds, val_indices)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_mem,
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_mem,
    )

    # -----------------------------------
    # (6) Compute target scaling (mu, std)
    # -----------------------------------
    with torch.no_grad():
        y_train = ds.audios[train_indices]  # 1D tensor of training targets
        mu  = y_train.mean().item()
        std = float(y_train.std().clamp_min(1e-8))
    print(f"[Scaler] mu={mu:.6e}  std={std:.6e}  (computed on training set only)")

    # --- Model (7) with improved MLP head (see models file)
    model = ConvNeXtV2TinyRegressor(pretrained=cfg["model"]["pretrained"])
    model.to(device)
    loss_fn = nn.MSELoss()

    lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"]["min_lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    epochs = int(cfg["train"]["epochs"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    best_loss = float("inf")
    best_epoch = -1
    train_losses, val_losses = [], []

    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['train']['epochs']} ===")
        lr_now = optimizer.param_groups[0]['lr']
        print(f"LR: {lr_now:.2e}")

        # ---- Show GPU temperature
        temps = get_gpu_temperatures()
        if temps:
            text = " | ".join([f"GPU{i}: {t}°C" for i, t in temps])
            print(f"🌡️ {text}")
        else:
            print("🌡️ GPU temp not available (CPU run / NVML not found / nvidia-smi missing)")

        train_loss, train_mae, train_r2, train_snr = run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            is_train=True, grad_clip=cfg["train"].get("grad_clip", 0),
            mu=mu, std=std, use_target_scaling=True
        )
        val_loss, val_mae, val_r2, val_snr = run_epoch(
            model, val_loader, loss_fn, optimizer, device, is_train=False,
            mu=mu, std=std, use_target_scaling=True
        )

        print(f"[Train] MSE: {train_loss:.6f} | MAE: {train_mae:.6f} | R2: {train_r2:.5f} | SNR: {train_snr:.5f} dB")
        print(f"[ Val ] MSE: {val_loss:.6f} | MAE: {val_mae:.6f} | R2: {val_r2:.5f} | SNR: {val_snr:.5f} dB")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        # Save best
        if val_loss < best_loss and cfg["output"]["save_best"]:
            best_loss = val_loss
            best_epoch = epoch
            ckpt_path = ckpt_dir / f"best.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)  # Defensive
            torch.save({
                "model": model.state_dict(),
                "mu": mu, "std": std,                 # save scaler for inference
                "cfg": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"[OK] Saved best model+scaler to {ckpt_path}")

        # Save every N epochs
        save_every = cfg["output"].get("save_every", 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch{epoch+1}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "mu": mu, "std": std,
                "cfg": cfg,
                "epoch": epoch+1,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"[OK] Saved model+scaler at epoch {epoch+1}")

    plot_losses(train_losses, val_losses, best_epoch, best_loss, str(ckpt_dir), model_name=cfg["model"]["backbone"])
    print(f"\n[Done] Training finished. Best val_loss: {best_loss:.5f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    main()
