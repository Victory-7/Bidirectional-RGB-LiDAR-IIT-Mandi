import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


"""
=====================================================================
EXPERIMENT 3
PSEUDO LIDAR GENERATION + POINT CLOUD REFINEMENT
=====================================================================

PIPELINE:

RGB
 -> Depth Estimation
 -> Geometric Projection
 -> Raw Pseudo LiDAR
 -> PointNet Refinement
 -> Final Point Cloud

=====================================================================
OUTPUTS
=====================================================================

models/
    best_refinement_model.pth
    final_refinement_model.pth

results/
    combined_loss_curve.png
    metrics.txt

results/predictions/
    prediction_1.png
    prediction_2.png

pseudo_lidar/
    000001.npy
    000002.npy

refined_pointclouds/
    000001.npy
    000002.npy

=====================================================================
"""


# ================================================================
# CONFIGURATION
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
VELODYNE_DIR = os.path.join(KITTI_PATH, "velodyne")
CALIB_DIR = os.path.join(KITTI_PATH, "calib")

DEPTH_MODEL_PATH = "models/best_depth_model.pth"

PSEUDO_DIR = "pseudo_lidar"
REFINED_DIR = "refined_pointclouds"

MODEL_DIR = "models"
RESULTS_DIR = "results"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(REFINED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4
EPOCHS = 50

LEARNING_RATE = 1e-4

NUM_POINTS = 2048

PATIENCE = 7
MIN_DELTA = 1e-5

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
# CALIBRATION READER
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
# DEPTH TO POINT CLOUD
# ================================================================

def depth_to_pointcloud(depth_map, calib):

    height, width = depth_map.shape

    P2 = calib["P2"].reshape(3, 4)

    fx = P2[0, 0]
    fy = P2[1, 1]

    cx = P2[0, 2]
    cy = P2[1, 2]

    points = []

    for v in range(height):

        for u in range(width):

            z = depth_map[v, u]

            if z <= 0:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            points.append([x, y, z])

    points = np.array(
        points,
        dtype=np.float32
    )

    if len(points) > 0:

        points = points[
            np.isfinite(points).all(axis=1)
        ]

    return points


# ================================================================
# DEPTH MODEL
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
# LOAD DEPTH MODEL
# ================================================================

print("\nLoading depth estimation model...")

depth_model = ResNetUNet().to(DEVICE)

depth_model.load_state_dict(
    torch.load(
        DEPTH_MODEL_PATH,
        map_location=DEVICE
    )
)

depth_model.eval()

print("Depth model loaded successfully")


# ================================================================
# GENERATE RAW PSEUDO LIDAR
# ================================================================

print("\nGenerating pseudo LiDAR point clouds...")

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

for image_path in tqdm(image_files):

    file_id = Path(image_path).stem

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

    tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        pred_depth = depth_model(tensor)

    pred_depth = pred_depth.squeeze().cpu().numpy()

    calib = read_calib(calib_path)

    pseudo_cloud = depth_to_pointcloud(
        pred_depth,
        calib
    )

    pseudo_path = os.path.join(
        PSEUDO_DIR,
        file_id + ".npy"
    )

    np.save(
        pseudo_path,
        pseudo_cloud
    )

print("\nPseudo LiDAR generation completed")


# ================================================================
# LOAD FILES
# ================================================================

pseudo_files = sorted(
    glob.glob(os.path.join(PSEUDO_DIR, "*.npy"))
)

lidar_files = sorted(
    glob.glob(os.path.join(VELODYNE_DIR, "*.bin"))
)

print(f"\nPseudo LiDAR Files: {len(pseudo_files)}")

print(f"Real LiDAR Files: {len(lidar_files)}")


# ================================================================
# TRAIN / VALIDATION SPLIT
# ================================================================

train_pseudo, val_pseudo, train_gt, val_gt = train_test_split(
    pseudo_files,
    lidar_files,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTraining Samples: {len(train_pseudo)}")

print(f"Validation Samples: {len(val_pseudo)}")


# ================================================================
# DATASET
# ================================================================

class PseudoLidarDataset(Dataset):

    def __init__(
        self,
        pseudo_paths,
        gt_paths
    ):

        self.pseudo_paths = pseudo_paths
        self.gt_paths = gt_paths

    def __len__(self):

        return len(self.pseudo_paths)

    def sample_points(self, cloud):

        if cloud.shape[0] == 0:

            return np.zeros(
                (NUM_POINTS, 3),
                dtype=np.float32
            )

        if cloud.shape[0] > NUM_POINTS:

            idx = np.random.choice(
                cloud.shape[0],
                NUM_POINTS,
                replace=False
            )

            cloud = cloud[idx]

        elif cloud.shape[0] < NUM_POINTS:

            padding = np.zeros(
                (
                    NUM_POINTS - cloud.shape[0],
                    3
                ),
                dtype=np.float32
            )

            cloud = np.vstack((cloud, padding))

        return cloud

    def __getitem__(self, idx):

        pseudo = np.load(
            self.pseudo_paths[idx]
        ).astype(np.float32)

        gt = np.fromfile(
            self.gt_paths[idx],
            dtype=np.float32
        ).reshape(-1, 4)[:, :3]

        pseudo = self.sample_points(pseudo)
        gt = self.sample_points(gt)

        pseudo_max = np.max(np.abs(pseudo))

        if pseudo_max > 0:

            pseudo = pseudo / pseudo_max

        gt_max = np.max(np.abs(gt))

        if gt_max > 0:

            gt = gt / gt_max

        pseudo = torch.tensor(
            pseudo,
            dtype=torch.float32
        )

        gt = torch.tensor(
            gt,
            dtype=torch.float32
        )

        return pseudo, gt


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = PseudoLidarDataset(
    train_pseudo,
    train_gt
)

val_dataset = PseudoLidarDataset(
    val_pseudo,
    val_gt
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ================================================================
# POINTNET REFINEMENT NETWORK
# ================================================================

class PointRefinementNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 3, 1)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.encoder(x)

        x = self.decoder(x)

        x = x.permute(0, 2, 1)

        return x


# ================================================================
# CHAMFER LOSS
# ================================================================

class ChamferLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, pred, target):

        dist = torch.cdist(
            pred,
            target
        )

        pred_to_gt = dist.min(dim=2)[0]

        gt_to_pred = dist.min(dim=1)[0]

        loss = (
            pred_to_gt.mean() +
            gt_to_pred.mean()
        )

        return loss


# ================================================================
# RMSE
# ================================================================

def compute_rmse(pred, gt):

    return torch.sqrt(
        torch.mean((pred - gt) ** 2)
    ).item()


# ================================================================
# MAE
# ================================================================

def compute_mae(pred, gt):

    return torch.mean(
        torch.abs(pred - gt)
    ).item()


# ================================================================
# MODEL SETUP
# ================================================================

model = PointRefinementNet().to(DEVICE)

criterion = ChamferLoss()

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

train_rmses = []
val_rmses = []

train_maes = []
val_maes = []

best_val_loss = float('inf')

early_stop_counter = 0

print("\nStarting refinement training...\n")

for epoch in range(EPOCHS):

    # =========================================================
    # TRAIN
    # =========================================================

    model.train()

    running_train_loss = 0

    running_train_rmse = 0
    running_train_mae = 0

    for pseudo, gt in tqdm(train_loader):

        pseudo = pseudo.to(DEVICE)

        gt = gt.to(DEVICE)

        optimizer.zero_grad()

        pred = model(pseudo)

        loss = criterion(pred, gt)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_train_loss += loss.item()

        running_train_rmse += compute_rmse(
            pred,
            gt
        )

        running_train_mae += compute_mae(
            pred,
            gt
        )

    avg_train_loss = (
        running_train_loss /
        len(train_loader)
    )

    avg_train_rmse = (
        running_train_rmse /
        len(train_loader)
    )

    avg_train_mae = (
        running_train_mae /
        len(train_loader)
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    model.eval()

    running_val_loss = 0

    running_val_rmse = 0
    running_val_mae = 0

    with torch.no_grad():

        for pseudo, gt in val_loader:

            pseudo = pseudo.to(DEVICE)

            gt = gt.to(DEVICE)

            pred = model(pseudo)

            loss = criterion(pred, gt)

            running_val_loss += loss.item()

            running_val_rmse += compute_rmse(
                pred,
                gt
            )

            running_val_mae += compute_mae(
                pred,
                gt
            )

    avg_val_loss = (
        running_val_loss /
        len(val_loader)
    )

    avg_val_rmse = (
        running_val_rmse /
        len(val_loader)
    )

    avg_val_mae = (
        running_val_mae /
        len(val_loader)
    )

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    train_rmses.append(avg_train_rmse)
    val_rmses.append(avg_val_rmse)

    train_maes.append(avg_train_mae)
    val_maes.append(avg_val_mae)

    scheduler.step(avg_val_loss)

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}]"
        f"\nTrain Loss: {avg_train_loss:.6f}"
        f"\nVal Loss: {avg_val_loss:.6f}"
        f"\nTrain RMSE: {avg_train_rmse:.6f}"
        f"\nVal RMSE: {avg_val_rmse:.6f}"
        f"\nTrain MAE: {avg_train_mae:.6f}"
        f"\nVal MAE: {avg_val_mae:.6f}"
    )

    # =========================================================
    # SAVE BEST MODEL
    # =========================================================

    if avg_val_loss < best_val_loss - MIN_DELTA:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_refinement_model.pth"
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

    # =========================================================
    # EARLY STOPPING
    # =========================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break


# ================================================================
# SAVE FINAL MODEL
# ================================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "final_refinement_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print(f"\nFinal model saved: {final_model_path}")


# ================================================================
# LOSS CURVE
# ================================================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Chamfer Loss Curve")

plt.legend()
plt.grid(True)

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "combined_loss_curve.png"
)

plt.savefig(loss_curve_path)
plt.close()

print(f"\nLoss curve saved: {loss_curve_path}")


# ================================================================
# METRICS TXT
# ================================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, 'w') as f:

    f.write("EXPERIMENT 3 REPORT\n")

    f.write("=" * 60 + "\n\n")

    f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")

    f.write(f"Final Validation RMSE: {avg_val_rmse:.6f}\n")

    f.write(f"Final Validation MAE: {avg_val_mae:.6f}\n")

print(f"\nMetrics saved: {metrics_path}")


# ================================================================
# SAVE REFINED POINT CLOUDS
# ================================================================

print("\nSaving refined point clouds...")

model.eval()

for pseudo_path in tqdm(pseudo_files[:100]):

    file_id = Path(pseudo_path).stem

    pseudo = np.load(
        pseudo_path
    ).astype(np.float32)

    if pseudo.shape[0] > NUM_POINTS:

        idx = np.random.choice(
            pseudo.shape[0],
            NUM_POINTS,
            replace=False
        )

        pseudo = pseudo[idx]

    elif pseudo.shape[0] < NUM_POINTS:

        padding = np.zeros(
            (
                NUM_POINTS - pseudo.shape[0],
                3
            ),
            dtype=np.float32
        )

        pseudo = np.vstack((pseudo, padding))

    pseudo_max = np.max(np.abs(pseudo))

    if pseudo_max > 0:

        pseudo = pseudo / pseudo_max

    tensor = torch.tensor(
        pseudo,
        dtype=torch.float32
    ).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        refined = model(tensor)

    refined = refined.squeeze().cpu().numpy()

    refined_path = os.path.join(
        REFINED_DIR,
        file_id + ".npy"
    )

    np.save(
        refined_path,
        refined
    )

print("\nRefined point clouds saved")


# ================================================================
# VISUALIZATIONS
# ================================================================

print("\nGenerating prediction visualizations...")

sample_batch = next(iter(val_loader))

pseudo_batch, gt_batch = sample_batch

pseudo_batch = pseudo_batch.to(DEVICE)

with torch.no_grad():

    refined_batch = model(pseudo_batch)

refined_batch = refined_batch.cpu().numpy()

gt_batch = gt_batch.numpy()

for i in range(min(3, len(refined_batch))):

    pred = refined_batch[i]

    gt = gt_batch[i]

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(
        121,
        projection='3d'
    )

    ax1.scatter(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        s=1
    )

    ax1.set_title("Refined Point Cloud")

    ax2 = fig.add_subplot(
        122,
        projection='3d'
    )

    ax2.scatter(
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        s=1
    )

    ax2.set_title("Ground Truth LiDAR")

    pred_path = os.path.join(
        PRED_DIR,
        f"prediction_{i+1}.png"
    )

    plt.savefig(pred_path)

    plt.close()

    print(f"Saved: {pred_path}")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n================================================")
print("EXPERIMENT 3 COMPLETED")
print("================================================")

print("\nGenerated Files:")

print("- best_refinement_model.pth")
print("- final_refinement_model.pth")

print("- pseudo LiDAR point clouds")
print("- refined point clouds")

print("- loss curve")
print("- metrics.txt")

print("- prediction visualizations")

print("\nPipeline:")
print("RGB -> Depth -> Projection -> Pseudo LiDAR -> Refinement")

print("\nResearch Stage:")
print("Geometry-Aware Pseudo LiDAR Generation")

print("\nTraining finished successfully.")