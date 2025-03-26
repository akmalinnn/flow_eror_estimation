import torch
import timm
import numpy as np
from pathlib import Path
import json

# ===========================
# 🔹 Konfigurasi Model
# ===========================
model_path = "efficientnet_b0_epoch200.pth"  # Path model yang sudah dilatih
of_dir = "flow/06_0491"  # Folder optical flow
labels_f = "data_filtered.json"  # JSON yang menyimpan informasi speed

# Periksa apakah CUDA tersedia
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model EfficientNet
v = 0  # EfficientNet-b0
in_c = 2  # Optical flow memiliki 2 channel
num_c = 1  # Model hanya memprediksi 1 nilai (kecepatan)

# Buat model yang sesuai dengan yang dilatih
model = timm.create_model(f'efficientnet_b{v}', pretrained=False, num_classes=num_c)
model.conv_stem = torch.nn.Conv2d(in_c, model.conv_stem.out_channels, 
                                  kernel_size=model.conv_stem.kernel_size, 
                                  stride=model.conv_stem.stride, 
                                  padding=model.conv_stem.padding, 
                                  bias=False)

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    model.classifier
)

# Load model dari file
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  


with open(labels_f, "r") as f:
    labels = json.load(f)

speeds = [labels[f]["speed"] for f in labels]
min_speed, max_speed = min(speeds), max(speeds)
print(f"✅ Loaded Min Speed: {min_speed}, Max Speed: {max_speed}")


def predict_speed_batch(npy_files):
    tensors = []
    valid_files = []

    for npy_file in npy_files:
        npy_path = Path(of_dir) / npy_file

        if not npy_path.exists():
            print(f"❌ File tidak ditemukan: {npy_path}")
            continue

        of_array = np.load(npy_path).astype(np.float32)
        of_tensor = torch.tensor(of_array, dtype=torch.float32)

        if of_tensor.dim() == 3:
            of_tensor = of_tensor.permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        tensors.append(of_tensor)
        valid_files.append(npy_file)

    if not tensors:
        print("❌ Tidak ada file valid untuk diproses.")
        return {}

    batch_tensor = torch.stack(tensors).to(device)


    with torch.no_grad():
        pred_norms = model(batch_tensor).squeeze().cpu().numpy()  # Konversi ke NumPy array
    pred_speeds = pred_norms * (max_speed - min_speed) + min_speed

    # Cetak hasil
    results = {valid_files[i]: pred_speeds[i] for i in range(len(valid_files))}
    for file, speed in results.items():
        print(f"✅ {file} → Predicted Speed: {speed:.2f} km/h")

    return results


all_npy_files = sorted([f.name for f in Path(of_dir).glob("*.npy")])

batch_files = all_npy_files[:10]

predict_speed_batch(batch_files)
