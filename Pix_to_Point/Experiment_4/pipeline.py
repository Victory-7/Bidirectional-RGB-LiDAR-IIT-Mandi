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
EXPERIMENT 4
DIRECT RGB -> LIDAR POINT CLOUD GENERATION
=====================================================================

PIPELINE:

RGB IMAGE
    ->
RESNET34 ENCODER
    ->
FEATURE VECTOR
    ->
POINT CLOUD DECODER
    ->
PREDICTED LIDAR POINT CLOUD

=====================================================================
OUTPUTS
=====================================================================

models/
    best_rgb_to_lidar_model.pth
    final_rgb_to_lidar_model.pth

results/
    loss_curve.png
    metrics.txt

results/predictions/
    prediction_1.png
    prediction_2.png

predicted_pointclouds/
    000001.npy
    000001.bin

=====================================================================
"""


# ================================================================
# CONFIGURATION
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
VELODYNE_DIR = os.path.join(KITTI_PATH, "velodyne")

MODEL_DIR = "models"

RESULTS_DIR = "results"

PRED_DIR = os.path.join(
    RESULTS_DIR,
    "predictions"
)

OUTPUT_CLOUD_DIR = "predicted_pointclouds"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(OUTPUT_CLOUD_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

BATCH_SIZE = 4

EPOCHS = 100

LEARNING_RATE = 1e-4

NUM_POINTS = 2048

PATIENCE = 10

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

lidar_files = sorted(
    glob.glob(os.path.join(VELODYNE_DIR, "*.bin"))
)

print(f"\nRGB Images: {len(image_files)}")

print(f"LiDAR Files: {len(lidar_files)}")


# ================================================================
# TRAIN / VALIDATION SPLIT
# ================================================================

train_imgs, val_imgs, train_lidar, val_lidar = train_test_split(
    image_files,
    lidar_files,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\nTraining Samples: {len(train_imgs)}")

print(f"Validation Samples: {len(val_imgs)}")


# ================================================================
# DATASET
# ================================================================

class RGBLiDARDataset(Dataset):

    def __init__(
        self,
        image_paths,
        lidar_paths
    ):

        self.image_paths = image_paths

        self.lidar_paths = lidar_paths

    def __len__(self):

        return len(self.image_paths)

    def sample_points(self, points):

        if points.shape[0] > NUM_POINTS:

            idx = np.random.choice(
                points.shape[0],
                NUM_POINTS,
                replace=False
            )

            points = points[idx]

        elif points.shape[0] < NUM_POINTS:

            padding = np.zeros(
                (
                    NUM_POINTS - points.shape[0],
                    3
                ),
                dtype=np.float32
            )

            points = np.vstack((
                points,
                padding
            ))

        return points

    def __getitem__(self, idx):

        # =====================================================
        # RGB IMAGE
        # =====================================================

        image = cv2.imread(
            self.image_paths[idx]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        image = image.astype(np.float32) / 255.0

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(2, 0, 1)

        # =====================================================
        # LIDAR
        # =====================================================

        lidar = np.fromfile(
            self.lidar_paths[idx],
            dtype=np.float32
        ).reshape(-1, 4)

        points = lidar[:, :3]

        # Remove far/noisy points

        mask = (
            (points[:, 0] > 0) &
            (points[:, 0] < 50)
        )

        points = points[mask]

        points = self.sample_points(points)

        # =====================================================
        # GLOBAL NORMALIZATION
        # =====================================================

        points[:, 0] /= 50.0
        points[:, 1] /= 50.0
        points[:, 2] /= 5.0

        points = torch.tensor(
            points,
            dtype=torch.float32
        )

        return image, points


# ================================================================
# DATALOADERS
# ================================================================

train_dataset = RGBLiDARDataset(
    train_imgs,
    train_lidar
)

val_dataset = RGBLiDARDataset(
    val_imgs,
    val_lidar
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
# MODEL
# ================================================================

class RGBToPointCloud(nn.Module):

    def __init__(self):

        super().__init__()

        base_model = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        self.encoder = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        self.decoder = nn.Sequential(

            nn.Linear(512, 1024),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                1024,
                NUM_POINTS * 3
            )
        )

    def forward(self, x):

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        x = self.decoder(x)

        x = x.view(
            -1,
            NUM_POINTS,
            3
        )

        return x


# ================================================================
# CHAMFER LOSS
# ================================================================

class ChamferLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, pred, gt):

        dist = torch.cdist(pred, gt)

        pred_to_gt = dist.min(dim=2)[0]

        gt_to_pred = dist.min(dim=1)[0]

        loss = (
            pred_to_gt.mean() +
            gt_to_pred.mean()
        )

        return loss


# ================================================================
# MODEL SETUP
# ================================================================

model = RGBToPointCloud().to(DEVICE)

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

best_val_loss = float("inf")

early_stop_counter = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # =========================================================
    # TRAIN
    # =========================================================

    model.train()

    running_train_loss = 0

    for images, gt_points in tqdm(train_loader):

        images = images.to(DEVICE)

        gt_points = gt_points.to(DEVICE)

        optimizer.zero_grad()

        pred_points = model(images)

        loss = criterion(
            pred_points,
            gt_points
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

    # =========================================================
    # VALIDATION
    # =========================================================

    model.eval()

    running_val_loss = 0

    with torch.no_grad():

        for images, gt_points in val_loader:

            images = images.to(DEVICE)

            gt_points = gt_points.to(DEVICE)

            pred_points = model(images)

            loss = criterion(
                pred_points,
                gt_points
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

    # =========================================================
    # SAVE BEST MODEL
    # =========================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        early_stop_counter = 0

        best_model_path = os.path.join(
            MODEL_DIR,
            "best_rgb_to_lidar_model.pth"
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
    "final_rgb_to_lidar_model.pth"
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

plt.title("Chamfer Loss Curve")

plt.legend()

plt.grid(True)

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

plt.savefig(loss_curve_path)

plt.close()

print(f"\nLoss curve saved")


# ================================================================
# SAVE PREDICTED POINT CLOUDS
# ================================================================

print("\nGenerating predicted point clouds...")

model.eval()

sample_images = val_imgs[:100]

for image_path in tqdm(sample_images):

    file_id = Path(image_path).stem

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

        pred = model(tensor)

    pred = pred.squeeze().cpu().numpy()

    # =========================================================
    # DENORMALIZE
    # =========================================================

    pred[:, 0] *= 50.0
    pred[:, 1] *= 50.0
    pred[:, 2] *= 5.0

    # =========================================================
    # SAVE NPY
    # =========================================================

    npy_path = os.path.join(
        OUTPUT_CLOUD_DIR,
        file_id + ".npy"
    )

    np.save(npy_path, pred)

    # =========================================================
    # SAVE BIN
    # =========================================================

    intensity = np.ones(
        (pred.shape[0], 1),
        dtype=np.float32
    )

    pred_bin = np.hstack((
        pred,
        intensity
    )).astype(np.float32)

    bin_path = os.path.join(
        OUTPUT_CLOUD_DIR,
        file_id + ".bin"
    )

    pred_bin.tofile(bin_path)

print("\nPredicted point clouds saved")


# ================================================================
# VISUALIZATION
# ================================================================

print("\nGenerating visualizations...")

sample_batch = next(iter(val_loader))

images, gt_points = sample_batch

images = images.to(DEVICE)

with torch.no_grad():

    pred_points = model(images)

pred_points = pred_points.cpu().numpy()

gt_points = gt_points.numpy()

for i in range(min(3, len(pred_points))):

    pred = pred_points[i]

    gt = gt_points[i]

    fig = plt.figure(figsize=(12, 6))

    # =========================================================
    # PREDICTED
    # =========================================================

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

    ax1.set_title(
        "Predicted Point Cloud"
    )

    # =========================================================
    # GROUND TRUTH
    # =========================================================

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

    ax2.set_title(
        "Ground Truth LiDAR"
    )

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

    f.write("EXPERIMENT 4 REPORT\n")

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

    f.write(
        f"Number of Points: "
        f"{NUM_POINTS}\n"
    )

print(f"\nMetrics saved")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n================================================")
print("EXPERIMENT 4 COMPLETED")
print("================================================")

print("\nGenerated Files:")
print("- best_rgb_to_lidar_model.pth")
print("- final_rgb_to_lidar_model.pth")
print("- predicted point clouds (.npy)")
print("- predicted KITTI .bin files")
print("- loss curve")
print("- metrics.txt")
print("- prediction visualizations")

print("\nPipeline:")
print("RGB -> Direct LiDAR Generation")

print("\nResearch Stage:")
print("Direct Image-to-PointCloud Learning")

print("\nTraining finished successfully.")