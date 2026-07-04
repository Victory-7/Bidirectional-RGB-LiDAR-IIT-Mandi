import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


"""
===============================================================
DEPTH -> POINT CLOUD TRAINING PIPELINE
===============================================================

INPUT:
    depth/*.npy

TARGET:
    pointcloud/*.npy

OUTPUTS:
    best_point_model.pth
    combined_loss_curve.png
    metrics.txt
    prediction visualization images

===============================================================
"""


# ============================================================
# CONFIGURATION
# ============================================================

DEPTH_DIR = "dataset/depth"
POINTCLOUD_DIR = "dataset/pointcloud"

MODEL_SAVE_PATH = "best_point_model.pth"

RESULTS_DIR = "results"
PREDICTION_DIR = os.path.join(
    RESULTS_DIR,
    "predictions"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 4
LEARNING_RATE = 0.000001
EPOCHS = 150

PATIENCE = 10
MIN_DELTA = 1e-5

NUM_POINTS = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# ============================================================
# LOAD FILES
# ============================================================

depth_files = sorted(
    glob.glob(os.path.join(DEPTH_DIR, "*.npy"))
)

pointcloud_files = sorted(
    glob.glob(os.path.join(POINTCLOUD_DIR, "*.npy"))
)

print(f"\nDepth Files: {len(depth_files)}")
print(f"Point Cloud Files: {len(pointcloud_files)}")

assert len(depth_files) == len(pointcloud_files), \
    "Mismatch between depth and pointcloud files"

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_depth, val_depth, train_pc, val_pc = train_test_split(
    depth_files,
    pointcloud_files,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTraining Samples: {len(train_depth)}")
print(f"Validation Samples: {len(val_depth)}")


# ============================================================
# DATASET
# ============================================================

class DepthPointDataset(Dataset):

    def __init__(self, depth_paths, pointcloud_paths):

        self.depth_paths = depth_paths
        self.pointcloud_paths = pointcloud_paths

    def __len__(self):

        return len(self.depth_paths)

    def __getitem__(self, idx):

        # ----------------------------------------------------
        # LOAD DEPTH
        # ----------------------------------------------------

        depth = np.load(
            self.depth_paths[idx]
        ).astype(np.float32)

        # ----------------------------------------------------
        # LOAD POINT CLOUD
        # ----------------------------------------------------

        pointcloud = np.load(
            self.pointcloud_paths[idx]
        ).astype(np.float32)

        # ----------------------------------------------------
        # NORMALIZE DEPTH
        # ----------------------------------------------------

        depth = (
            depth - depth.min()
        ) / (
            depth.max() -
            depth.min() +
            1e-8
        )

        # ----------------------------------------------------
        # FIX POINT COUNT
        # ----------------------------------------------------

        if pointcloud.shape[0] > NUM_POINTS:

            indices = np.random.choice(
                pointcloud.shape[0],
                NUM_POINTS,
                replace=False
            )

            pointcloud = pointcloud[indices]

        elif pointcloud.shape[0] < NUM_POINTS:

            padding = np.zeros(
                (
                    NUM_POINTS -
                    pointcloud.shape[0],
                    3
                ),
                dtype=np.float32
            )

            pointcloud = np.vstack(
                (pointcloud, padding)
            )

        # ----------------------------------------------------
        # TENSOR
        # ----------------------------------------------------

        depth = torch.tensor(
            depth,
            dtype=torch.float32
        ).unsqueeze(0)

        pointcloud = torch.tensor(
            pointcloud,
            dtype=torch.float32
        )

        return depth, pointcloud


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = DepthPointDataset(
    train_depth,
    train_pc
)

val_dataset = DepthPointDataset(
    val_depth,
    val_pc
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


# ============================================================
# MODEL
# ============================================================

class PointCloudNet(nn.Module):

    def __init__(self):

        super(PointCloudNet, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256, 1024),
            nn.ReLU(),

            nn.Linear(
                1024,
                NUM_POINTS * 3
            )
        )

    def forward(self, x):

        x = self.encoder(x)

        x = self.fc(x)

        x = x.view(
            -1,
            NUM_POINTS,
            3
        )

        return x


# ============================================================
# CHAMFER DISTANCE LOSS
# ============================================================

class ChamferLoss(nn.Module):

    def __init__(self):

        super(ChamferLoss, self).__init__()

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


# ============================================================
# MODEL SETUP
# ============================================================

model = PointCloudNet().to(DEVICE)

criterion = ChamferLoss()

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


# ============================================================
# TRAINING
# ============================================================

train_losses = []
val_losses = []

best_val_loss = float("inf")

early_stop_counter = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_train_loss = 0

    for depth, pointcloud in tqdm(train_loader):

        depth = depth.to(DEVICE)
        pointcloud = pointcloud.to(DEVICE)

        optimizer.zero_grad()

        pred = model(depth)

        loss = criterion(
            pred,
            pointcloud
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = (
        running_train_loss /
        len(train_loader)
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    running_val_loss = 0

    with torch.no_grad():

        for depth, pointcloud in val_loader:

            depth = depth.to(DEVICE)
            pointcloud = pointcloud.to(DEVICE)

            pred = model(depth)

            loss = criterion(
                pred,
                pointcloud
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

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if avg_val_loss < best_val_loss - MIN_DELTA:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("\nBest model saved")

    else:

        early_stop_counter += 1

        print(
            f"\nNo improvement for "
            f"{early_stop_counter} epoch(s)"
        )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping triggered")

        break


# ============================================================
# SAVE LOSS CURVE
# ============================================================

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

loss_path = os.path.join(
    RESULTS_DIR,
    "combined_loss_curve.png"
)

plt.savefig(loss_path)

plt.close()

print(f"\nLoss curve saved: {loss_path}")


# ============================================================
# PREDICTION VISUALIZATIONS
# ============================================================

print("\nGenerating prediction visualizations...")

model.eval()

sample_batch = next(iter(val_loader))

depths, gt_clouds = sample_batch

depths = depths.to(DEVICE)

with torch.no_grad():

    pred_clouds = model(depths)

pred_clouds = pred_clouds.cpu().numpy()
gt_clouds = gt_clouds.numpy()

for i in range(min(3, len(pred_clouds))):

    pred = pred_clouds[i]
    gt = gt_clouds[i]

    fig = plt.figure(figsize=(12, 6))

    # --------------------------------------------------------
    # PREDICTED
    # --------------------------------------------------------

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

    ax1.set_title("Predicted Point Cloud")

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

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

    ax2.set_title("Ground Truth Point Cloud")

    pred_path = os.path.join(
        PREDICTION_DIR,
        f"prediction_{i+1}.png"
    )

    plt.savefig(pred_path)

    plt.close()

    print(f"Saved: {pred_path}")


# ============================================================
# METRICS REPORT
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("DEPTH TO POINT CLOUD REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Epochs Completed: {len(train_losses)}\n")
    f.write(f"Best Validation Loss: {best_val_loss}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Number of Points: {NUM_POINTS}\n")

    f.write("\nTraining Losses:\n")

    for idx, loss in enumerate(train_losses):

        f.write(
            f"Epoch {idx+1}: "
            f"{loss:.6f}\n"
        )

    f.write("\nValidation Losses:\n")

    for idx, loss in enumerate(val_losses):

        f.write(
            f"Epoch {idx+1}: "
            f"{loss:.6f}\n"
        )

print(f"\nMetrics saved: {metrics_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n================================================")
print("TRAINING COMPLETED")
print("================================================")

print("\nGenerated Files:")
print("- best_point_model.pth")
print("- combined_loss_curve.png")
print("- metrics.txt")
print("- prediction visualization images")

print("\nPipeline:")
print("Depth Map -> Point Cloud")

print("\nResearch Stage:")
print("Pseudo LiDAR Generation")