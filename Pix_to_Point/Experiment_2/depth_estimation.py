# ================================================================
# FIXED KITTI DEPTH ESTIMATION PIPELINE
# RESNET34 + U-NET DECODER
# FULLY FIXED VERSION
# ================================================================

import os
import glob
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader


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
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4
EPOCHS = 150

LEARNING_RATE = 1e-5

PATIENCE = 7

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

print(f"Total RGB Images: {len(image_files)}")


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
# READ CALIBRATION
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
# GENERATE DEPTH MAP
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

    # =========================================================
    # LOAD LIDAR
    # =========================================================

    lidar = np.fromfile(
        velodyne_path,
        dtype=np.float32
    ).reshape(-1, 4)

    points = lidar[:, :3]

    # REMOVE POINTS BEHIND CAMERA
    points = points[points[:, 0] > 0]

    # =========================================================
    # HOMOGENEOUS COORDS
    # =========================================================

    points_hom = np.hstack((
        points,
        np.ones((points.shape[0], 1))
    ))

    # =========================================================
    # PROJECT TO CAMERA
    # =========================================================

    cam_points = (
        R0_rect @
        Tr_velo_to_cam @
        points_hom.T
    )

    # =========================================================
    # PROJECT TO IMAGE
    # =========================================================

    img_points = P2 @ cam_points

    z = img_points[2]

    # =========================================================
    # REMOVE INVALID DEPTH
    # =========================================================

    valid_depth = z > 0.1

    img_points = img_points[:, valid_depth]

    z = z[valid_depth]

    # =========================================================
    # SAFE DIVISION
    # =========================================================

    img_points[:2] /= z

    u = img_points[0]
    v = img_points[1]

    # =========================================================
    # REMOVE NaN / INF
    # =========================================================

    valid_projection = (
        np.isfinite(u) &
        np.isfinite(v)
    )

    u = u[valid_projection]
    v = v[valid_projection]

    z = z[valid_projection]

    # =========================================================
    # CONVERT TO INT
    # =========================================================

    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # =========================================================
    # IMAGE BOUNDS
    # =========================================================

    valid = (
        (u >= 0) &
        (u < width) &
        (v >= 0) &
        (v < height)
    )

    u = u[valid]
    v = v[valid]

    z = z[valid]

    # =========================================================
    # DEPTH MAP
    # =========================================================

    depth_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    depth_map[v, u] = z

    # =========================================================
    # SPARSE DEPTH FILLING
    # =========================================================

    depth_map = cv2.dilate(
        depth_map,
        np.ones((5, 5), np.uint8)
    )

    depth_map = cv2.GaussianBlur(
        depth_map,
        (5, 5),
        0
    )

    # =========================================================
    # FIXED DEPTH NORMALIZATION
    # =========================================================

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

        # =====================================================
        # RGB IMAGE
        # =====================================================

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

        # =====================================================
        # DEPTH MAP
        # =====================================================

        depth_map = generate_depth_map(
            velodyne_path,
            calib_path,
            image.shape
        )

        depth_map = cv2.resize(
            depth_map,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        # =====================================================
        # VALID DEPTH MASK
        # =====================================================

        mask = (depth_map > 0).astype(np.float32)

        # =====================================================
        # TENSOR
        # =====================================================

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        depth_map = torch.tensor(
            depth_map,
            dtype=torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, depth_map, mask


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
# CONV BLOCK
# ================================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ================================================================
# UP BLOCK
# ================================================================

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


# ================================================================
# RESNET34 U-NET MODEL
# ================================================================

class ResNetUNetDepth(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        )

        # =====================================================
        # ENCODER
        # =====================================================

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # =====================================================
        # DECODER
        # =====================================================

        self.up4 = UpBlock(512, 256, 256)

        self.up3 = UpBlock(256, 128, 128)

        self.up2 = UpBlock(128, 64, 64)

        self.up1 = UpBlock(64, 64, 32)

        self.final_up = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x0 = self.initial(x)

        x1 = self.maxpool(x0)

        x1 = self.encoder1(x1)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        d4 = self.up4(x4, x3)

        d3 = self.up3(d4, x2)

        d2 = self.up2(d3, x1)

        d1 = self.up1(d2, x0)

        out = self.final_up(d1)

        out = self.final(out)

        return out


# ================================================================
# MASKED LOSS
# ================================================================

class MaskedL1Loss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        pred,
        target,
        mask
    ):

        diff = torch.abs(pred - target)

        diff = diff * mask

        loss = diff.sum() / (
            mask.sum() + 1e-8
        )

        return loss


# ================================================================
# MODEL
# ================================================================

model = ResNetUNetDepth().to(DEVICE)

criterion = MaskedL1Loss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# FIXED AMP
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

    for images, depths, masks in tqdm(train_loader):

        images = images.to(DEVICE)

        depths = depths.to(DEVICE)

        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        # FIXED AMP
        with torch.amp.autocast('cuda'):

            outputs = model(images)

            loss = criterion(
                outputs,
                depths,
                masks
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_train_loss += loss.item()

    avg_train_loss = (
        running_train_loss / len(train_loader)
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, depths, masks in val_loader:

            images = images.to(DEVICE)

            depths = depths.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                depths,
                masks
            )

            running_val_loss += loss.item()

    avg_val_loss = (
        running_val_loss / len(val_loader)
    )

    train_losses.append(avg_train_loss)

    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
        f"\nTrain Loss: {avg_train_loss:.6f}"
        f"\nVal Loss: {avg_val_loss:.6f}"
        f"\nLearning Rate: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    # =========================================================
    # SAVE BEST MODEL
    # =========================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        model_path = os.path.join(
            OUTPUT_MODEL_DIR,
            "best_depth_model.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(f"\nBest model saved: {model_path}")

    else:

        early_stop_counter += 1

        print(
            f"\nNo improvement for "
            f"{early_stop_counter} epoch(s)"
        )

    # =========================================================
    # EARLY STOPPING
    # =========================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered.")

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

print(f"\nFinal model saved: {final_model_path}")


# ================================================================
# SAVE LOSS CURVE
# ================================================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")

plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

loss_curve_path = os.path.join(
    OUTPUT_RESULTS_DIR,
    "combined_loss_curve.png"
)

plt.savefig(loss_curve_path)

plt.close()

print(f"\nLoss graph saved: {loss_curve_path}")