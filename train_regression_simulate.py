import os

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

# ---- Set seed ASAP for all libraries ----
seed = 1337
seed_everything(seed)

# --- Usual imports ---
import yaml
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from models.convnextv2_tiny import ConvNeXtV2TinyRegressor
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# ---- GPU temperature helper ----
def get_gpu_temperature():
    """Returns the temperature (°C) of the first available GPU, or None if not available."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception:
        return None

# ---- YAML loader ----
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---- Dataset ----
class FrameToAudioDataset(Dataset):
    def __init__(self, pt_path):
        pkg = torch.load(pt_path, map_location="cpu")
        self.frames = torch.cat(pkg["frames_list"], dim=0).float() / 255.0  # [N,128,128]
        self.audios = torch.cat([a.view(-1) for a in pkg["audio_list"]], dim=0)  # [N]
        self.N = self.frames.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.frames[idx][None, ...], self.audios[idx]  # [1,128,128], float32

# ---- Metrics ----
def compute_mae(pred, target):
    return (pred - target).abs().mean().item()

def compute_r2(pred, target):
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    return 1 - (ss_res / (ss_tot + 1e-8))

def compute_snr_db(pred, target):
    signal_power = (target ** 2).mean()
    noise_power = ((target - pred) ** 2).mean() + 1e-12
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

# ---- Loss curve plotter ----
def plot_losses(train_losses, val_losses, best_epoch, best_val_loss, ckpt_dir, model_name="convnextv2_tiny"):
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name.upper()} Loss Curve (Audio Regression)")
    plt.legend()
    plt.grid(True)
    if best_epoch >= 0:
        ymin, ymax = plt.gca().get_ylim()
        y_offset = (ymax - ymin) * 0.05
        plt.annotate(
            f"{best_val_loss:.6f}",
            xy=(best_epoch, best_val_loss),
            xytext=(best_epoch, best_val_loss + y_offset),
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
            fontsize=12,
            color="red",
            weight="bold",
            ha="center"
        )
    lossplot_name = f"{model_name}_loss.png"
    plt.savefig(os.path.join(ckpt_dir, lossplot_name))
    plt.close()

# ---- Training/Validation Loops ----
def run_epoch(model, loader, loss_fn, optimizer, device, is_train=True, grad_clip=0.0):
    model.train() if is_train else model.eval()
    total_loss, total_mae, total_r2, total_snr = 0.0, 0.0, 0.0, 0.0
    count = 0
    loop = tqdm(loader, disable=not is_train, desc="Train" if is_train else "Val")
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
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
            total_loss += loss.item() * x.size(0)
            total_mae += mae * x.size(0)
            total_r2  += r2 * x.size(0)
            total_snr += snr * x.size(0)
            count += x.size(0)
    avg_loss = total_loss / count
    avg_mae = total_mae / count
    avg_r2  = total_r2 / count
    avg_snr = total_snr / count
    return avg_loss, avg_mae, avg_r2, avg_snr

# ---- Main ----
def main():
    # --- Config, device, seeds
    cfg = load_yaml("configs/train_regression.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset and Split
    ds = FrameToAudioDataset(cfg["data"]["pt_path"])
    n_total = len(ds)
    n_val = int(cfg["train"].get("val_ratio", 0.2) * n_total)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(cfg["seed"])
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    # --- Model
    model = ConvNeXtV2TinyRegressor(pretrained=cfg["model"]["pretrained"])
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_lr"]
    )

    # --- Output/checkpoints
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = cfg["output"].get("run_tag", "") or dt_str
    ckpt_dir = Path(cfg["output"]["root"]) / f"{cfg['model']['backbone']}_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    train_losses, val_losses = [], []

    # --- Training Loop
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch+1}/{cfg['train']['epochs']} ===")
        lr = optimizer.param_groups[0]['lr']
        print(f"LR: {lr:.2e}")

        # ---- Show GPU temperature
        gpu_temp = get_gpu_temperature()
        if gpu_temp is not None:
            print(f"🌡️ GPU Temp: {gpu_temp}°C")

        train_loss, train_mae, train_r2, train_snr = run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            is_train=True, grad_clip=cfg["train"].get("grad_clip", 0)
        )
        val_loss, val_mae, val_r2, val_snr = run_epoch(
            model, val_loader, loss_fn, optimizer, device, is_train=False
        )

        print(f"[Train] MSE: {train_loss:.5f} | MAE: {train_mae:.5f} | R2: {train_r2:.4f} | SNR: {train_snr:.2f} dB")
        print(f"[ Val ] MSE: {val_loss:.5f} | MAE: {val_mae:.5f} | R2: {val_r2:.4f} | SNR: {val_snr:.2f} dB")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        # Save best
        if val_loss < best_loss and cfg["output"]["save_best"]:
            best_loss = val_loss
            best_epoch = epoch
            ckpt_path = ckpt_dir / f"best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[OK] Saved best model to {ckpt_path}")

        # Save every N epochs
        save_every = cfg["output"].get("save_every", 0)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch{epoch+1}.pt")
            print(f"[OK] Saved model at epoch {epoch+1}")

    # ---- Plot and save the loss curve
    plot_losses(train_losses, val_losses, best_epoch, best_loss, str(ckpt_dir), model_name=cfg["model"]["backbone"])
    print(f"\n[Done] Training finished. Best val_loss: {best_loss:.5f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    main()
