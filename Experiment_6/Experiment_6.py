import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import models


"""
=====================================================================
EXPERIMENT 6
LIDAR -> RGB RECONSTRUCTION
=====================================================================

GOAL:
Generate RGB images from pseudo LiDAR .bin files

PIPELINE:

Pseudo LiDAR (.bin)
        ↓
Sparse Depth Map
        ↓
Dense Geometry Encoder
        ↓
RGB Reconstruction Network
        ↓
RGB Image

=====================================================================
OUTPUTS
=====================================================================

models/
    best_lidar2rgb_model.pth
    final_lidar2rgb_model.pth

results/
    lidar2rgb_loss_curve.png
    metrics.txt
    predictions/

=====================================================================
"""


# ================================================================
# CONFIG
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")

# YOUR GENERATED BIN FILES
BIN_DIR = "pseudo_lidar_bin_training"

MODEL_DIR = "models"

RESULTS_DIR = "results_lidar2rgb"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4
EPOCHS = 200

LEARNING_RATE = 1e-4

PATIENCE = 10
MIN_DELTA = 1e-5

MAX_DEPTH = 80.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# ================================================================
# RANDOM SEED
# ================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()


# ================================================================
# LOAD FILES
# ================================================================

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"\nTotal Images: {len(image_files)}")


# ================================================================
# TRAIN / VAL SPLIT
# ================================================================

train_files, val_files = train_test_split(
    image_files,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Training Samples: {len(train_files)}")
print(f"Validation Samples: {len(val_files)}")


# ================================================================
# BIN -> DEPTH MAP
# ================================================================

def bin_to_depth_map(bin_path):

    point_cloud = np.fromfile(
        bin_path,
        dtype=np.float32
    ).reshape(-1, 4)

    points = point_cloud[:, :3]

    depth_map = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32
    )

    fx = 721.5377
    fy = 721.5377

    cx = 609.5593
    cy = 172.8540

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    valid = z > 0

    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = ((x * fx) / z + cx).astype(np.int32)
    v = ((y * fy) / z + cy).astype(np.int32)

    valid = (
        (u >= 0) &
        (u < IMAGE_WIDTH) &
        (v >= 0) &
        (v < IMAGE_HEIGHT)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    depth_map[v, u] = z

    depth_map = np.clip(
        depth_map,
        0,
        MAX_DEPTH
    )

    depth_map = depth_map / MAX_DEPTH

    return depth_map


# ================================================================
# DATASET
# ================================================================

class LiDAR2RGBDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        file_id = Path(image_path).stem

        bin_path = os.path.join(
            BIN_DIR,
            file_id + ".bin"
        )

        rgb = cv2.imread(image_path)

        rgb = cv2.cvtColor(
            rgb,
            cv2.COLOR_BGR2RGB
        )

        rgb = cv2.resize(
            rgb,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        rgb = rgb.astype(np.float32) / 255.0

        depth_map = bin_to_depth_map(bin_path)

        validity_mask = (
            depth_map > 0
        ).astype(np.float32)

        depth_tensor = torch.tensor(
            depth_map,
            dtype=torch.float32
        ).unsqueeze(0)

        mask_tensor = torch.tensor(
            validity_mask,
            dtype=torch.float32
        ).unsqueeze(0)

        input_tensor = torch.cat(
            [depth_tensor, mask_tensor],
            dim=0
        )

        rgb_tensor = torch.tensor(
            rgb,
            dtype=torch.float32
        ).permute(2, 0, 1)

        return input_tensor, rgb_tensor


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = LiDAR2RGBDataset(train_files)
val_dataset = LiDAR2RGBDataset(val_files)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ================================================================
# LIDAR -> RGB NETWORK
# ================================================================

class LiDAR2RGBNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ========================================================
        # ENCODER
        # ========================================================

        self.encoder = nn.Sequential(

            nn.Conv2d(2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # ========================================================
        # DECODER
        # ========================================================

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                256,
                256,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                3,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


# ================================================================
# MODEL
# ================================================================

model = LiDAR2RGBNet().to(DEVICE)

criterion = nn.L1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)


# ================================================================
# TRAINING
# ================================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")
early_stop_counter = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # ============================================================
    # TRAIN
    # ============================================================

    model.train()

    running_train_loss = 0

    for inputs, gt_rgb in tqdm(train_loader):

        inputs = inputs.to(DEVICE)
        gt_rgb = gt_rgb.to(DEVICE)

        optimizer.zero_grad()

        pred_rgb = model(inputs)

        loss = criterion(
            pred_rgb,
            gt_rgb
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = (
        running_train_loss /
        len(train_loader)
    )

    # ============================================================
    # VALIDATION
    # ============================================================

    model.eval()

    running_val_loss = 0

    with torch.no_grad():

        for inputs, gt_rgb in val_loader:

            inputs = inputs.to(DEVICE)
            gt_rgb = gt_rgb.to(DEVICE)

            pred_rgb = model(inputs)

            loss = criterion(
                pred_rgb,
                gt_rgb
            )

            running_val_loss += loss.item()

    avg_val_loss = (
        running_val_loss /
        len(val_loader)
    )

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
        f"\nTrain Loss: {avg_train_loss:.6f}"
        f"\nVal Loss: {avg_val_loss:.6f}"
    )

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    if avg_val_loss < best_val_loss - MIN_DELTA:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_lidar2rgb_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("\nBest model saved")

    else:

        early_stop_counter += 1

        print(
            f"\nNo improvement for "
            f"{early_stop_counter} epoch(s)"
        )

    # ============================================================
    # EARLY STOPPING
    # ============================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break


# ================================================================
# SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_lidar2rgb_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print("\nFinal model saved")


# ================================================================
# LOSS CURVE
# ================================================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("LiDAR -> RGB Loss Curve")

plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "lidar2rgb_loss_curve.png"
)

plt.savefig(loss_curve_path)
plt.close()

print("\nLoss curve saved")


# ================================================================
# VISUALIZATION
# ================================================================

print("\nGenerating predictions...")

sample_batch = next(iter(val_loader))

inputs, gt_rgb = sample_batch

inputs = inputs.to(DEVICE)

with torch.no_grad():

    pred_rgb = model(inputs)

inputs = inputs.cpu().numpy()
gt_rgb = gt_rgb.numpy()
pred_rgb = pred_rgb.cpu().numpy()

for i in range(min(5, len(inputs))):

    sparse_depth = inputs[i][0]

    pred = np.transpose(
        pred_rgb[i],
        (1, 2, 0)
    )

    gt = np.transpose(
        gt_rgb[i],
        (1, 2, 0)
    )

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(sparse_depth, cmap="plasma")
    plt.title("Input LiDAR Depth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred)
    plt.title("Predicted RGB")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(gt)
    plt.title("Ground Truth RGB")
    plt.axis("off")

    pred_path = os.path.join(
        PRED_DIR,
        f"prediction_{i+1}.png"
    )

    plt.savefig(pred_path)
    plt.close()

    print(f"Saved: {pred_path}")


# ================================================================
# METRICS
# ================================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("EXPERIMENT 6 REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Best Validation Loss: "
        f"{best_val_loss:.6f}\n"
    )

    f.write(
        f"Epochs Completed: "
        f"{len(train_losses)}\n"
    )

    f.write(
        f"Learning Rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

print("\nMetrics saved")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n================================================")
print("EXPERIMENT 6 COMPLETED")
print("================================================")

print("\nGenerated Files:")
print("- best_lidar2rgb_model.pth")
print("- final_lidar2rgb_model.pth")
print("- prediction visualizations")
print("- metrics.txt")

print("\nPipeline:")
print("Pseudo LiDAR -> RGB Reconstruction")

print("\nTraining completed successfully.")