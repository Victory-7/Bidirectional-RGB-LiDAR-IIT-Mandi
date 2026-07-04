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
EXPERIMENT 5
RGB + SPARSE DEPTH COMPLETION + PSEUDO LIDAR GENERATION
=====================================================================

GOAL:
Improve pseudo LiDAR quality by using:

1. RGB Image
2. Sparse LiDAR Depth
3. Depth Completion Network
4. Better Geometry
5. Early Stopping
6. Smooth Dense Depth

PIPELINE:

RGB + Sparse Depth
        ↓
Depth Completion Network
        ↓
Dense Depth
        ↓
Pseudo LiDAR Projection
        ↓
KITTI .bin Export

=====================================================================
OUTPUTS
=====================================================================

models/
    best_depth_completion_model.pth
    final_depth_completion_model.pth

dense_depth/
    000001.npy

pseudo_lidar_bin/
    000001.bin

results/
    loss_curve.png
    metrics.txt
    predictions/

=====================================================================
"""


# ================================================================
# CONFIG
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
VELODYNE_DIR = os.path.join(KITTI_PATH, "velodyne")
CALIB_DIR = os.path.join(KITTI_PATH, "calib")

MODEL_DIR = "models"

DENSE_DEPTH_DIR = "dense_depth"
BIN_OUTPUT_DIR = "pseudo_lidar_bin"

RESULTS_DIR = "results"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DENSE_DEPTH_DIR, exist_ok=True)
os.makedirs(BIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4
EPOCHS = 5000

LEARNING_RATE = 1e-4

PATIENCE = 7
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
# CALIBRATION
# ================================================================

def read_calib(calib_path):

    calib = {}

    with open(calib_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        calib[key] = np.array(
            [float(x) for x in value.strip().split()]
        )

    return calib


# ================================================================
# SPARSE DEPTH MAP
# ================================================================

def generate_sparse_depth_map(
    velodyne_path,
    calib_path,
    image_shape
):

    height, width = image_shape[:2]

    calib = read_calib(calib_path)

    P2 = calib["P2"].reshape(3, 4)

    R0 = calib["R0_rect"].reshape(3, 3)

    Tr = calib["Tr_velo_to_cam"].reshape(3, 4)

    R0_rect = np.eye(4)
    R0_rect[:3, :3] = R0

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :] = Tr

    lidar = np.fromfile(
        velodyne_path,
        dtype=np.float32
    ).reshape(-1, 4)

    points = lidar[:, :3]

    points = points[points[:, 0] > 0]

    points_hom = np.hstack((
        points,
        np.ones((points.shape[0], 1))
    ))

    cam_points = (
        R0_rect @
        Tr_velo_to_cam @
        points_hom.T
    )

    img_points = P2 @ cam_points

    z = img_points[2]

    valid = z > 0

    img_points = img_points[:, valid]
    z = z[valid]

    img_points[:2] /= z

    u = img_points[0].astype(np.int32)
    v = img_points[1].astype(np.int32)

    valid = (
        (u >= 0) &
        (u < width) &
        (v >= 0) &
        (v < height)
    )

    u = u[valid]
    v = v[valid]
    z = z[valid]

    depth_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

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

class DepthCompletionDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        file_id = Path(image_path).stem

        velodyne_path = os.path.join(
            VELODYNE_DIR,
            file_id + ".bin"
        )

        calib_path = os.path.join(
            CALIB_DIR,
            file_id + ".txt"
        )

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        image = image.astype(np.float32) / 255.0

        sparse_depth = generate_sparse_depth_map(
            velodyne_path,
            calib_path,
            image.shape
        )

        sparse_depth = cv2.resize(
            sparse_depth,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        rgb_tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        depth_tensor = torch.tensor(
            sparse_depth,
            dtype=torch.float32
        ).unsqueeze(0)

        input_tensor = torch.cat(
            [rgb_tensor, depth_tensor],
            dim=0
        )

        return input_tensor, depth_tensor


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = DepthCompletionDataset(train_files)
val_dataset = DepthCompletionDataset(val_files)

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
# DEPTH COMPLETION NETWORK
# ================================================================

class DepthCompletionNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


# ================================================================
# MODEL
# ================================================================

model = DepthCompletionNet().to(DEVICE)

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

    for inputs, gt_depth in tqdm(train_loader):

        inputs = inputs.to(DEVICE)
        gt_depth = gt_depth.to(DEVICE)

        optimizer.zero_grad()

        pred_depth = model(inputs)

        mask = gt_depth > 0

        loss = criterion(
            pred_depth[mask],
            gt_depth[mask]
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

        for inputs, gt_depth in val_loader:

            inputs = inputs.to(DEVICE)
            gt_depth = gt_depth.to(DEVICE)

            pred_depth = model(inputs)

            mask = gt_depth > 0

            loss = criterion(
                pred_depth[mask],
                gt_depth[mask]
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
    # EARLY STOPPING
    # ============================================================

    if avg_val_loss < best_val_loss - MIN_DELTA:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_depth_completion_model.pth"
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

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break


# ================================================================
# SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_depth_completion_model.pth"
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

plt.title("Loss Curve")

plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

plt.savefig(loss_curve_path)
plt.close()

print("\nLoss curve saved")


# ================================================================
# DEPTH -> POINT CLOUD
# ================================================================

def depth_to_pointcloud(depth_map):

    h, w = depth_map.shape

    fx = 721.5377
    fy = 721.5377

    cx = 609.5593
    cy = 172.8540

    points = []

    for v in range(h):

        for u in range(w):

            z = depth_map[v, u] * MAX_DEPTH

            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

    return np.array(
        points,
        dtype=np.float32
    )


# ================================================================
# SAVE DENSE DEPTH + BIN FILES
# ================================================================

print("\nGenerating pseudo LiDAR .bin files...")

model.eval()

for image_path in tqdm(val_files[:100]):

    file_id = Path(image_path).stem

    velodyne_path = os.path.join(
        VELODYNE_DIR,
        file_id + ".bin"
    )

    calib_path = os.path.join(
        CALIB_DIR,
        file_id + ".txt"
    )

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    image = image.astype(np.float32) / 255.0

    sparse_depth = generate_sparse_depth_map(
        velodyne_path,
        calib_path,
        image.shape
    )

    rgb_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1)

    sparse_tensor = torch.tensor(
        sparse_depth,
        dtype=torch.float32
    ).unsqueeze(0)

    input_tensor = torch.cat(
        [rgb_tensor, sparse_tensor],
        dim=0
    ).unsqueeze(0)

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        pred_depth = model(input_tensor)

    pred_depth = pred_depth.squeeze().cpu().numpy()

    dense_path = os.path.join(
        DENSE_DEPTH_DIR,
        file_id + ".npy"
    )

    np.save(dense_path, pred_depth)

    point_cloud = depth_to_pointcloud(
        pred_depth
    )

    intensity = np.ones(
        (point_cloud.shape[0], 1),
        dtype=np.float32
    )

    point_cloud_bin = np.hstack((
        point_cloud,
        intensity
    ))

    bin_path = os.path.join(
        BIN_OUTPUT_DIR,
        file_id + ".bin"
    )

    point_cloud_bin.astype(
        np.float32
    ).tofile(bin_path)

print("\nPseudo LiDAR .bin files saved")


# ================================================================
# VISUALIZATIONS
# ================================================================

print("\nGenerating visualizations...")

sample_batch = next(iter(val_loader))

inputs, gt_depth = sample_batch

inputs = inputs.to(DEVICE)

with torch.no_grad():

    pred_depth = model(inputs)

inputs = inputs.cpu().numpy()
gt_depth = gt_depth.numpy()
pred_depth = pred_depth.cpu().numpy()

for i in range(min(5, len(inputs))):

    rgb = np.transpose(
        inputs[i][:3],
        (1, 2, 0)
    )

    sparse = inputs[i][3]

    pred = pred_depth[i][0]
    gt = gt_depth[i][0]

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(sparse, cmap="plasma")
    plt.title("Sparse Depth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="plasma")
    plt.title("Predicted Dense Depth")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gt, cmap="plasma")
    plt.title("Ground Truth Sparse")
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

    f.write("EXPERIMENT 5 REPORT\n")
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
print("EXPERIMENT 5 COMPLETED")
print("================================================")

print("\nGenerated Files:")
print("- best_depth_completion_model.pth")
print("- final_depth_completion_model.pth")
print("- dense depth maps")
print("- pseudo LiDAR .bin files")
print("- prediction visualizations")
print("- metrics.txt")

print("\nPipeline:")
print("RGB + Sparse Depth -> Dense Depth -> Pseudo LiDAR")

print("\nTraining completed successfully.")