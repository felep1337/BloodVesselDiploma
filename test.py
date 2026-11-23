import os
import hdf5plugin  # ОБЯЗАТЕЛЬНО до import h5py
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import butter, filtfilt, hilbert, lfilter
import cv2
from huggingface_hub import hf_hub_download, list_repo_files

# 1. Ищем и скачиваем файл 15_10_50_19.hdf5
repo_id = "nvidia/Open-H-RF"
target_name = "15_10_50_19.hdf5"

all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
matched_files = [f for f in all_files if f.endswith(target_name)]

os.makedirs("data", exist_ok=True)
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=matched_files[0],
    repo_type="dataset",
    local_dir="data"
)

# 2. Открываем HDF5 по точным путям структуры tracks/track_0/data/...
h5_file = h5py.File(local_path, "r")

raw_rf_ds = h5_file["tracks/track_0/data/raw_data"]          # (60, 360, 8192, 1, 1)
orig_bmode_ds = h5_file["tracks/track_0/data/image/values"]  # (60, 985, 986)

# Считываем реальную частоту дискретизации из метаданных файла (в Гц -> переводим в МГц)
fs_hz = float(h5_file["tracks/track_0/scan/sampling_frequency"][()])
fs_mhz = fs_hz / 1e6 if fs_hz > 1e6 else 1000.0
num_frames = raw_rf_ds.shape[0]

print(f"Файл открыт! Кадров: {num_frames} | Частота дискретизации: {fs_mhz:.1f} МГц")

# 3. Настройка полосового фильтра (25–85 МГц)
b, a = butter(N=4, Wn=[25.0 / (0.5 * fs_mhz), 85.0 / (0.5 * fs_mhz)], btype="band")

def reconstruct_bmode(frame_idx, dynamic_range_db=55.0, img_size=985):
    """Выполняет DSP-реконструкцию B-режима из RF-сигнала для одного кадра."""
    # Убираем лишние оси (1, 1) -> получаем форму (360 углов, 8192 отсчета глубины)
    raw_rf = np.squeeze(raw_rf_ds[frame_idx]).astype(np.float32)
    
    # Транспонируем в (8192 глубины, 360 углов)
    if raw_rf.shape[0] == 360:
        raw_rf = raw_rf.T

    # 1. Полосовая фильтрация вдоль глубины (axis=0)
    rf_filtered = lfilter(b, a, raw_rf, axis=0)
    
    # 2. Огибающая Гильберта
    envelope = np.abs(hilbert(rf_filtered, axis=0))

    # 3. Логарифмическое сжатие
    envelope_norm = envelope / (np.max(envelope) + 1e-12)
    log_compressed = 20 * np.log10(envelope_norm + 1e-12)
    log_clipped = np.clip(log_compressed, -dynamic_range_db, 0)
    polar_img = ((log_clipped + dynamic_range_db) / dynamic_range_db * 255).astype(np.uint8)

    # 4. Полярно-декартова развертка (cv2.warpPolar ожидает форму: 360 углов x глубина
    center = (img_size // 2, img_size // 2)
    cartesian_img = cv2.warpPolar(
        polar_img.T,
        dsize=(img_size, img_size),
        center=center,
        maxRadius=img_size // 2,
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR
    )
    return polar_img, cartesian_img

# 4. Отрисовка 3 окон: Полярный -> Наша реконструкция -> Эталон рядом
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
plt.subplots_adjust(bottom=0.18)

init_polar, init_cart = reconstruct_bmode(0)
init_ref = orig_bmode_ds[0]

im0 = axes[0].imshow(init_polar, cmap="gray", aspect="auto")
axes[0].set_title("1. Полярный B-режим (из RF)")
axes[0].set_xlabel("Углы (0° – 360°)")
axes[0].set_ylabel("Глубина (отсчеты)")

im1 = axes[1].imshow(init_cart, cmap="gray")
axes[1].set_title("2. Наша реконструкция (25–85 МГц)")
axes[1].axis("off")

im2 = axes[2].imshow(init_ref, cmap="gray")
axes[2].set_title("3. Эталон (image/values)")
axes[2].axis("off")

# 5. Ползунок кадров
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
frame_slider = Slider(
    ax=ax_slider,
    label="Кадр ",
    valmin=0,
    valmax=num_frames - 1,
    valinit=0,
    valstep=1
)

def update(val):
    idx = int(frame_slider.val)
    polar_img, cart_img = reconstruct_bmode(idx)
    
    im0.set_data(polar_img)
    im1.set_data(cart_img)
    im2.set_data(orig_bmode_ds[idx])
    
    fig.suptitle(f"Файл: {target_name} | Кадр: {idx} / {num_frames - 1}", fontsize=12)
    fig.canvas.draw_idle()

frame_slider.on_changed(update)
fig.suptitle(f"Файл: {target_name} | Кадр: 0 / {num_frames - 1}", fontsize=12)

plt.show()
h5_file.close()
