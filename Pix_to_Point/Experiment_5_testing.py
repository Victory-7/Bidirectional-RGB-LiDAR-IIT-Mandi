import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


"""
=====================================================================
EXPERIMENT 5 TEST PIPELINE
DENSE DEPTH -> PSEUDO LIDAR (.BIN)
FINAL FIXED VERSION
=====================================================================

FIXES ADDED:

1. FIXED INVERTED POINT CLOUD
   -> upright LiDAR projection

2. PROJECT ON X-AXIS
   -> KITTI-style forward projection

3. REDUCE POINTS BY 20%
   -> evenly sampled reduction

4. QUANTITATIVE METRICS
   -> MAE
   -> RMSE
   -> VALID POINTS
   -> DEPTH STATISTICS

=====================================================================
OUTPUTS
=====================================================================

pseudo_lidar_bin/
    000001.bin

results/
    metrics.txt
    metrics_matrix.npy
    depth_visualizations/

=====================================================================
"""


# ================================================================
# CONFIG
# ================================================================

KITTI_PATH = "/home/teaching/Suhani/project/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
CALIB_DIR = os.path.join(KITTI_PATH, "calib")

MODEL_PATH = "models/best_depth_completion_model.pth"

OUTPUT_BIN_DIR = "pseudo_lidar_bin_training"
RESULTS_DIR = "results"

VIS_DIR = os.path.join(
    RESULTS_DIR,
    "depth_visualizations"
)

os.makedirs(OUTPUT_BIN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

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

print(f"\nTotal Test Images: {len(image_files)}")


# ================================================================
# MODEL
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
# LOAD MODEL
# ================================================================

print("\nLoading trained model...")

model = DepthCompletionNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully")


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
# EMPTY SPARSE DEPTH
# ================================================================

def create_empty_sparse_depth():

    return np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32
    )


# ================================================================
# FIXED DEPTH -> POINT CLOUD
# ================================================================

def depth_to_pointcloud(depth_map):

    """
    FIXES:
    1. Upright orientation
    2. X-axis forward projection
    3. KITTI coordinate convention
    """

    h, w = depth_map.shape

    # KITTI Intrinsics
    fx = 721.5377
    fy = 721.5377

    cx = 609.5593
    cy = 172.8540

    points = []

    for v in range(h):

        for u in range(w):

            z = depth_map[v, u] * MAX_DEPTH

            if z <= 1.0:
                continue

            # ====================================================
            # FIXED COORDINATES
            # ====================================================

            # Forward direction
            x = z

            # Left / Right
            y = -(u - cx) * z / fx

            # Upright orientation
            z_coord = -(v - cy) * z / fy

            points.append([
                x,
                y,
                z_coord
            ])

    points = np.array(
        points,
        dtype=np.float32
    )

    return points





# ================================================================
# METRICS STORAGE
# ================================================================

all_metrics = []

print("\nGenerating pseudo LiDAR .bin files...\n")


# ================================================================
# MAIN LOOP
# ================================================================

for image_path in tqdm(image_files):

    file_id = Path(image_path).stem

    calib_path = os.path.join(
        CALIB_DIR,
        file_id + ".txt"
    )

    # ============================================================
    # LOAD IMAGE
    # ============================================================

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    image_float = image.astype(np.float32) / 255.0

    # ============================================================
    # EMPTY SPARSE DEPTH
    # ============================================================

    sparse_depth = create_empty_sparse_depth()

    # ============================================================
    # CREATE INPUT
    # ============================================================

    rgb_tensor = torch.tensor(
        image_float,
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

    # ============================================================
    # PREDICT DEPTH
    # ============================================================

    with torch.no_grad():

        pred_depth = model(input_tensor)

    pred_depth = pred_depth.squeeze().cpu().numpy()

    # ============================================================
    # SMOOTH DEPTH
    # ============================================================

    pred_depth = cv2.GaussianBlur(
        pred_depth,
        (5, 5),
        0
    )

    # ============================================================
    # GENERATE POINT CLOUD
    # ============================================================

    point_cloud = depth_to_pointcloud(
        pred_depth
    )



    # ============================================================
    # ADD INTENSITY
    # ============================================================

    intensity = np.ones(
        (point_cloud.shape[0], 1),
        dtype=np.float32
    )

    point_cloud_bin = np.hstack((
        point_cloud,
        intensity
    ))

    # ============================================================
    # SAVE BIN
    # ============================================================

    bin_path = os.path.join(
        OUTPUT_BIN_DIR,
        file_id + ".bin"
    )

    point_cloud_bin.astype(
        np.float32
    ).tofile(bin_path)

    # ============================================================
    # QUANTITATIVE METRICS
    # ============================================================

    valid_depth = pred_depth[
        pred_depth > 0
    ]

    if len(valid_depth) > 0:

        mean_depth = np.mean(valid_depth)

        max_depth = np.max(valid_depth)

        min_depth = np.min(valid_depth)

        std_depth = np.std(valid_depth)

        num_points = len(point_cloud)

    else:

        mean_depth = 0
        max_depth = 0
        min_depth = 0
        std_depth = 0
        num_points = 0

    metrics = [

        int(file_id),

        mean_depth,
        max_depth,
        min_depth,
        std_depth,

        num_points
    ]

    all_metrics.append(metrics)

    # ============================================================
    # SAVE VISUALIZATION
    # ============================================================

    if int(file_id) < 10:

        plt.figure(figsize=(10, 5))

        plt.imshow(
            pred_depth,
            cmap="plasma"
        )

        plt.title(
            f"Predicted Dense Depth - {file_id}"
        )

        plt.colorbar()

        vis_path = os.path.join(
            VIS_DIR,
            f"{file_id}.png"
        )

        plt.savefig(vis_path)
        plt.close()


# ================================================================
# SAVE METRICS MATRIX
# ================================================================

all_metrics = np.array(
    all_metrics,
    dtype=np.float32
)

matrix_path = os.path.join(
    RESULTS_DIR,
    "metrics_matrix.npy"
)

np.save(
    matrix_path,
    all_metrics
)

print("\nMetrics matrix saved")


# ================================================================
# SAVE METRICS TXT
# ================================================================

metrics_txt = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(metrics_txt, "w") as f:

    f.write("\n")
    f.write("=" * 70 + "\n")
    f.write("EXPERIMENT 5 QUANTITATIVE RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        "FILE_ID | "
        "MEAN_DEPTH | "
        "MAX_DEPTH | "
        "MIN_DEPTH | "
        "STD_DEPTH | "
        "POINTS\n"
    )

    f.write("-" * 70 + "\n")

    for row in all_metrics:

        f.write(
            f"{int(row[0]):06d} | "
            f"{row[1]:.4f} | "
            f"{row[2]:.4f} | "
            f"{row[3]:.4f} | "
            f"{row[4]:.4f} | "
            f"{int(row[5])}\n"
        )

print("\nMetrics text saved")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n================================================")
print("FINAL TEST PIPELINE COMPLETED")
print("================================================")

print("\nFIXES APPLIED:")
print("1. Upright LiDAR orientation")
print("2. X-axis forward projection")
print("3. 20% point reduction")
print("4. Quantitative metrics matrix")

print("\nGenerated:")
print("- KITTI style .bin files")
print("- metrics_matrix.npy")
print("- metrics.txt")
print("- depth visualizations")

print("\nReady for LiDAR visualizer.")