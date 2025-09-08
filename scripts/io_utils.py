import subprocess

def get_gpu_temperatures():
    """
    Returns a list of (index, tempC) for all visible GPUs.
    Tries NVML (pynvml) first, then falls back to `nvidia-smi`.
    Returns [] if no GPU or cannot query.
    """
    # --- Try NVML ---
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            temps = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                temps.append((i, int(t)))
            return temps
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    # --- Fallback: nvidia-smi ---
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True
        )
        temps = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                temps.append((int(parts[0]), int(parts[1])))
        return temps
    except Exception:
        return []
