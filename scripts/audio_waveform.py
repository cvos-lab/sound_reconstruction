import matplotlib.pyplot as plt
import numpy as np

def draw_waveform(ax, x, y, length=2.7, label=None, color='#2696C6', n_samples=36):
    t = np.linspace(0, 1, n_samples)
    # Musical, smooth waveform: sum of harmonics
    amp = (
        0.8 * np.sin(2 * np.pi * 1.25 * t + 0.3)
        + 0.32 * np.sin(2 * np.pi * 2.7 * t + 1.2)
        + 0.15 * np.sin(2 * np.pi * 5.2 * t + 2.3)
    ) * (0.97 - 0.15 * t**1.7)
    ax.plot(x + t * length, y + 1.5 + amp, c=color, lw=2.7, zorder=12)
    ax.plot(x + t * length, [y + 1.5] * len(t), c='#888', lw=1.1, ls="--")
    if label:
        ax.annotate(label, (x + length / 2, y + 2.41), fontsize=18, ha='center', color='#155769', fontweight='bold')
        ax.arrow(x - 0.14, y + 1.2, 0, 1.12, head_width=0.10, head_length=0.16, fc='k', ec='k', lw=2.2, zorder=50)
        ax.text(x - 0.26, y + 2.1, 'Amplitude', fontsize=16, ha='right', va='center', rotation=90, fontweight='bold', zorder=50)
        ax.arrow(x, y + 1.1, length * 0.97, 0, head_width=0.09, head_length=0.13, fc='k', ec='k', lw=2.2, zorder=50)
        ax.text(x + length * 0.87, y + 0.78, 'Time (s)', fontsize=16, ha='center', va='top', fontweight='bold', zorder=50)
    return t, amp

def draw_sampling(ax, x, y, length, t_cont, amp_cont):
    x_samples = x + t_cont * length
    y_samples = y + 1.5 + amp_cont
    for xi, yi in zip(x_samples, y_samples):
        ax.plot([xi, xi], [y + 1.5, yi], color='crimson', lw=1.3, ls='-', alpha=0.92, zorder=22)
        ax.plot(xi, yi, 'o', color='crimson', markersize=6, zorder=23)

if __name__ == "__main__":
    n_audio_samples = 24
    audio_length = 2.7  # Wider waveform
    fig, ax = plt.subplots(figsize=(5.5, 2.6))  # Double previous size!
    ax.axis('off')
    t_cont, amp_cont = draw_waveform(ax, x=0.8, y=0.3, length=audio_length, label="Audio waveform", color="#2696C6", n_samples=n_audio_samples)
    draw_sampling(ax, x=0.8, y=0.3, length=audio_length, t_cont=t_cont, amp_cont=amp_cont)
    plt.tight_layout()
    plt.show()
