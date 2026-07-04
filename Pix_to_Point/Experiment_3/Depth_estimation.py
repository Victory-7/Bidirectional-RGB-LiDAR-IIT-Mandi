import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models


"""
=====================================================================
RESNET34 + U-NET KITTI DEPTH ESTIMATION PIPELINE
=====================================================================

IMPROVEMENTS:
1. ResNet34 Encoder
2. U-Net Decoder
3. Skip Connections
4. Mixed Precision Training
5. Stable Depth Projection
6. Masked Loss
7. Smooth Depth Learning
8. Early Stopping
9. Better Generalization

=====================================================================
OUTPUTS
=====================================================================

models/
    best_depth_model.pth
    final_depth_model.pth

results/
    combined_loss_curve.png
    metrics.txt

results/predictions/
    prediction_1.png
    prediction_2.png

=====================================================================
"""


# ================================================================
# CONFIGURATION
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
VELODYNE_DIR = os.path.join(KITTI_PATH, "velodyne")
CALIB_DIR = os.path.join(KITTI_PATH, "calib")

OUTPUT_MODEL_DIR = "models"
OUTPUT_RESULTS_DIR = "results"
PREDICTION_DIR = os.path.join(
    OUTPUT_RESULTS_DIR,
    "predictions"
)

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)


# ================================================================
# TRAINING PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4
EPOCHS = 150

LEARNING_RATE = 1e-4

PATIENCE = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# ================================================================
# LOAD IMAGE FILES
# ================================================================

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"Total RGB Images: {len(image_files)}")


# ================================================================
# TRAIN / VALIDATION SPLIT
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
# DEPTH MAP GENERATION
# ================================================================

def generate_depth_map(
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

    valid_z = z > 1e-5

    img_points = img_points[:, valid_z]
    z = z[valid_z]

    img_points[:2] /= z

    u = img_points[0]
    v = img_points[1]

    finite_mask = (
        np.isfinite(u) &
        np.isfinite(v) &
        np.isfinite(z)
    )

    u = u[finite_mask]
    v = v[finite_mask]
    z = z[finite_mask]

    u = u.astype(np.int32)
    v = v.astype(np.int32)

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

    # Fill sparse holes

    mask = (depth_map == 0).astype(np.uint8)

    depth_map = cv2.inpaint(
        depth_map,
        mask,
        5,
        cv2.INPAINT_NS
    )

    depth_map = cv2.GaussianBlur(
        depth_map,
        (5, 5),
        0
    )

    # Normalize

    if depth_map.max() > 0:

        depth_map = (
            depth_map /
            depth_map.max()
        )

    return depth_map


# ================================================================
# DATASET
# ================================================================

class KITTIDepthDataset(Dataset):

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

        depth_map = generate_depth_map(
            velodyne_path,
            calib_path,
            image.shape
        )

        depth_map = cv2.resize(
            depth_map,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        depth_map = torch.tensor(
            depth_map,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, depth_map


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = KITTIDepthDataset(train_files)
val_dataset = KITTIDepthDataset(val_files)

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
# RESNET U-NET
# ================================================================

class ResNetUNet(nn.Module):

    def __init__(self):

        super().__init__()

        base_model = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        # Encoder

        self.initial = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )

        self.maxpool = base_model.maxpool

        self.encoder1 = base_model.layer1
        self.encoder2 = base_model.layer2
        self.encoder3 = base_model.layer3
        self.encoder4 = base_model.layer4

        # Decoder

        self.up1 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )

        self.dec1 = nn.Sequential(

            nn.Conv2d(
                512,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(
                256,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )

        self.dec2 = nn.Sequential(

            nn.Conv2d(
                256,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                128,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.dec3 = nn.Sequential(

            nn.Conv2d(
                128,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up4 = nn.ConvTranspose2d(
            64,
            64,
            2,
            stride=2
        )

        self.final = nn.Sequential(

            nn.Conv2d(
                128,
                64,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                1,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x1 = self.initial(x)

        x2 = self.maxpool(x1)

        x3 = self.encoder1(x2)
        x4 = self.encoder2(x3)
        x5 = self.encoder3(x4)
        x6 = self.encoder4(x5)

        d1 = self.up1(x6)

        d1 = torch.cat([d1, x5], dim=1)

        d1 = self.dec1(d1)

        d2 = self.up2(d1)

        d2 = torch.cat([d2, x4], dim=1)

        d2 = self.dec2(d2)

        d3 = self.up3(d2)

        d3 = torch.cat([d3, x3], dim=1)

        d3 = self.dec3(d3)

        d4 = self.up4(d3)

        d4 = torch.cat([d4, x1], dim=1)

        out = self.final(d4)

        out = torch.nn.functional.interpolate(
            out,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            mode='bilinear',
            align_corners=False
        )

        return out


# ================================================================
# MODEL
# ================================================================

model = ResNetUNet().to(DEVICE)

criterion = nn.L1Loss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

scaler = torch.amp.GradScaler('cuda')


# ================================================================
# TRAINING
# ================================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

early_stop_counter = 0

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_train_loss = 0.0

    for images, depths in tqdm(train_loader):

        images = images.to(DEVICE)
        depths = depths.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):

            outputs = model(images)

            mask = depths > 0

            loss = criterion(
                outputs[mask],
                depths[mask]
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_train_loss += loss.item()

    avg_train_loss = (
        running_train_loss /
        len(train_loader)
    )

    # Validation

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, depths in val_loader:

            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            outputs = model(images)

            mask = depths > 0

            loss = criterion(
                outputs[mask],
                depths[mask]
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

    # Save best

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            OUTPUT_MODEL_DIR,
            "best_depth_model.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(f"\nBest model saved")

    else:

        early_stop_counter += 1

        print(
            f"\nNo improvement for "
            f"{early_stop_counter} epoch(s)"
        )

    # Early stopping

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break


# ================================================================
# SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(
    OUTPUT_MODEL_DIR,
    "final_depth_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print(f"\nFinal model saved")


# ================================================================
# LOSS CURVE
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

plt.title("Loss Curve")

plt.legend()

plt.grid(True)

loss_curve_path = os.path.join(
    OUTPUT_RESULTS_DIR,
    "combined_loss_curve.png"
)

plt.savefig(loss_curve_path)

plt.close()

print(f"\nLoss curve saved")


# ================================================================
# PREDICTIONS
# ================================================================

print("\nGenerating prediction visualizations...")

model.eval()

sample_batch = next(iter(val_loader))

images, depths = sample_batch

images = images.to(DEVICE)

with torch.no_grad():

    predictions = model(images)

images = images.cpu().numpy()
depths = depths.numpy()
predictions = predictions.cpu().numpy()

for i in range(min(5, len(images))):

    rgb = np.transpose(images[i], (1, 2, 0))

    pred = predictions[i][0]
    gt = depths[i][0]

    plt.figure(figsize=(15, 5))

    # RGB

    plt.subplot(1, 3, 1)

    plt.imshow(rgb)

    plt.title("RGB")

    plt.axis("off")

    # Predicted

    plt.subplot(1, 3, 2)

    plt.imshow(pred, cmap="plasma")

    plt.title("Predicted Depth")

    plt.axis("off")

    # GT

    plt.subplot(1, 3, 3)

    plt.imshow(gt, cmap="plasma")

    plt.title("Ground Truth")

    plt.axis("off")

    prediction_path = os.path.join(
        PREDICTION_DIR,
        f"prediction_{i+1}.png"
    )

    plt.savefig(prediction_path)

    plt.close()

    print(f"Saved: {prediction_path}")


# ================================================================
# METRICS
# ================================================================

metrics_path = os.path.join(
    OUTPUT_RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("RESNET34 U-NET DEPTH REPORT\n")

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Epochs Completed: "
        f"{len(train_losses)}\n"
    )

    f.write(
        f"Best Validation Loss: "
        f"{best_val_loss}\n"
    )

    f.write(
        f"Learning Rate: "
        f"{LEARNING_RATE}\n"
    )

    f.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

print(f"\nMetrics saved")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n=================================================")
print("TRAINING COMPLETED")
print("=================================================")

print("\nGenerated Files:")
print("- best_depth_model.pth")
print("- final_depth_model.pth")
print("- combined_loss_curve.png")
print("- metrics.txt")
print("- prediction visualization images")

print("\nPipeline:")
print("RGB -> Depth -> Pseudo LiDAR")

print("\nNext Stage:")
print("Depth -> Point Cloud Projection")

print("\nTraining finished successfully.")