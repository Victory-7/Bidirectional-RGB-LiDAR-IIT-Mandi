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


# ================================================================
# GPU
# ================================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("\n================================================")
    print("GPU ENABLED")
    print("================================================")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.backends.cudnn.benchmark = True

else:

    DEVICE = torch.device("cpu")

    print("\nRunning on CPU")


# ================================================================
# CONFIG
# ================================================================

KITTI_PATH = "/home/teaching/Suhani/project/kitti_object/training"

IMAGE_DIR = os.path.join(
    KITTI_PATH,
    "image_2"
)

VELODYNE_DIR = (
    "/home/teaching/Suhani/project/"
    "pseudo_lidar_bin_training"
)

MODEL_DIR = "models"

RESULTS_DIR = "results_lidm_rgb"

PRED_DIR = os.path.join(
    RESULTS_DIR,
    "predictions"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

RANGE_HEIGHT = 64
RANGE_WIDTH = 1024

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 2

EPOCHS = 100

LEARNING_RATE = 5e-5

PATIENCE = 10
MIN_DELTA = 1e-5

print(f"\nUsing device: {DEVICE}")


# ================================================================
# SEED
# ================================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


seed_everything()


# ================================================================
# FILES
# ================================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)

print(f"\nTotal Images: {len(image_files)}")


# ================================================================
# TRAIN VAL SPLIT
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
# LiDAR -> RANGE IMAGE
# ================================================================

def lidar_to_range_image(bin_path):

    points = np.fromfile(
        bin_path,
        dtype=np.float32
    ).reshape(-1, 4)

    xyz = points[:, :3]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    depth = np.linalg.norm(
        xyz,
        axis=1
    )

    mask = depth > 0

    x = x[mask]
    y = y[mask]
    z = z[mask]
    depth = depth[mask]

    yaw = -np.arctan2(y, x)

    ratio = z / (depth + 1e-8)

    ratio = np.clip(
        ratio,
        -1.0,
        1.0
    )

    pitch = np.arcsin(ratio)

    fov_up = np.radians(3.0)
    fov_down = np.radians(-25.0)

    fov = abs(fov_down) + abs(fov_up)

    proj_x = 0.5 * (
        yaw / np.pi + 1.0
    )

    proj_x *= RANGE_WIDTH

    proj_y = 1.0 - (
        (pitch + abs(fov_down)) / fov
    )

    proj_y *= RANGE_HEIGHT

    proj_x = np.floor(
        proj_x
    ).astype(np.int32)

    proj_y = np.floor(
        proj_y
    ).astype(np.int32)

    proj_x = np.clip(
        proj_x,
        0,
        RANGE_WIDTH - 1
    )

    proj_y = np.clip(
        proj_y,
        0,
        RANGE_HEIGHT - 1
    )

    range_image = np.zeros(
        (RANGE_HEIGHT, RANGE_WIDTH),
        dtype=np.float32
    )

    order = np.argsort(depth)[::-1]

    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    range_image[
        proj_y,
        proj_x
    ] = depth

    valid = range_image > 0

    if np.sum(valid) > 0:

        min_val = range_image[valid].min()

        max_val = range_image[valid].max()

        range_image = (
            range_image - min_val
        ) / (
            max_val - min_val + 1e-6
        )

    range_image = np.nan_to_num(
        range_image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return range_image


# ================================================================
# DATASET
# ================================================================

class LiDMRGBDataset(Dataset):

    def __init__(self, image_paths):

        self.image_paths = image_paths

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        file_id = Path(image_path).stem

        bin_path = os.path.join(
            VELODYNE_DIR,
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

        rgb = rgb.astype(
            np.float32
        ) / 255.0

        range_image = lidar_to_range_image(
            bin_path
        )

        range_image = cv2.resize(
            range_image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        range_tensor = torch.tensor(
            range_image,
            dtype=torch.float32
        ).unsqueeze(0)

        rgb_tensor = torch.tensor(
            rgb,
            dtype=torch.float32
        ).permute(2, 0, 1)

        return range_tensor, rgb_tensor


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = LiDMRGBDataset(
    train_files
)

val_dataset = LiDMRGBDataset(
    val_files
)

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
# CONV BLOCK
# ================================================================

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ================================================================
# UNET
# ================================================================

class LiDM_UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = ConvBlock(1, 64)

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)

        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            512,
            1024
        )

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            2,
            stride=2
        )

        self.dec4 = ConvBlock(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec3 = ConvBlock(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = ConvBlock(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec1 = ConvBlock(
            128,
            64
        )

        self.final = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        p1 = self.pool1(e1)

        e2 = self.enc2(p1)

        p2 = self.pool2(e2)

        e3 = self.enc3(p2)

        p3 = self.pool3(e3)

        e4 = self.enc4(p3)

        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        out = self.final(d1)

        return torch.sigmoid(out)


# ================================================================
# MODEL
# ================================================================

model = LiDM_UNet().to(DEVICE)

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

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
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

    valid_batches = 0

    for inputs, gt_rgb in tqdm(train_loader):

        inputs = inputs.to(
            DEVICE,
            non_blocking=True
        )

        gt_rgb = gt_rgb.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available()
        ):

            pred_rgb = model(inputs)

            if torch.isnan(pred_rgb).any():

                continue

            loss = criterion(
                pred_rgb,
                gt_rgb
            )

            if torch.isnan(loss):

                continue

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_train_loss += loss.item()

        valid_batches += 1

    avg_train_loss = (
        running_train_loss /
        max(valid_batches, 1)
    )

    # ============================================================
    # VALIDATION
    # ============================================================

    model.eval()

    running_val_loss = 0

    valid_val_batches = 0

    with torch.no_grad():

        for inputs, gt_rgb in val_loader:

            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            gt_rgb = gt_rgb.to(
                DEVICE,
                non_blocking=True
            )

            pred_rgb = model(inputs)

            if torch.isnan(pred_rgb).any():

                continue

            loss = criterion(
                pred_rgb,
                gt_rgb
            )

            if torch.isnan(loss):

                continue

            running_val_loss += loss.item()

            valid_val_batches += 1

    avg_val_loss = (
        running_val_loss /
        max(valid_val_batches, 1)
    )

    train_losses.append(
        avg_train_loss
    )

    val_losses.append(
        avg_val_loss
    )

    scheduler.step(avg_val_loss)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
        f"\nTrain Loss: {avg_train_loss:.6f}"
        f"\nVal Loss: {avg_val_loss:.6f}"
    )

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================

    if avg_val_loss < (
        best_val_loss - MIN_DELTA
    ):

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_lidm_rgb_model.pth"
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
    "final_lidm_rgb_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print("\nFinal model saved")


# ================================================================
# SAVE LOSS CURVE
# ================================================================

plt.figure(figsize=(10, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "LiDM RGB Reconstruction Loss"
)

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
# PREDICTIONS
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

for i in range(
    min(5, len(inputs))
):

    range_img = inputs[i][0]

    pred = np.transpose(
        pred_rgb[i],
        (1, 2, 0)
    )

    gt = np.transpose(
        gt_rgb[i],
        (1, 2, 0)
    )

    pred = np.clip(pred, 0, 1)

    gt = np.clip(gt, 0, 1)

    plt.figure(figsize=(18, 5))

    # ============================================================
    # INPUT
    # ============================================================

    plt.subplot(1, 3, 1)

    plt.imshow(
        range_img,
        cmap="turbo"
    )

    plt.title(
        "Input Range Image"
    )

    plt.axis("off")

    # ============================================================
    # PREDICTION
    # ============================================================

    plt.subplot(1, 3, 2)

    plt.imshow(pred)

    plt.title(
        "Predicted RGB"
    )

    plt.axis("off")

    # ============================================================
    # GROUND TRUTH
    # ============================================================

    plt.subplot(1, 3, 3)

    plt.imshow(gt)

    plt.title(
        "Ground Truth RGB"
    )

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

    f.write(
        "LiDM RGB RECONSTRUCTION REPORT\n"
    )

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
print("LiDM RGB RECONSTRUCTION COMPLETED")
print("================================================")

print("\nGenerated Files:")

print(
    "- models/best_lidm_rgb_model.pth"
)

print(
    "- models/final_lidm_rgb_model.pth"
)

print(
    "- results_lidm_rgb/loss_curve.png"
)

print(
    "- results_lidm_rgb/metrics.txt"
)

print(
    "- results_lidm_rgb/predictions/"
)

print(
    "\nPipeline:"
)

print(
    "LiDAR Range Image -> RGB Reconstruction"
)

print(
    "\nTraining completed successfully."
)