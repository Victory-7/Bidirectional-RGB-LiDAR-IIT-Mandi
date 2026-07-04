import os
import glob
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn


# ============================================================
# CONFIG
# ============================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")
VELODYNE_DIR = os.path.join(KITTI_PATH, "velodyne")
CALIB_DIR = os.path.join(KITTI_PATH, "calib")

MODEL_PATH = "models/best_depth_completion_model.pth"

OUTPUT_BIN_DIR = "pseudo_lidar_bin"
OUTPUT_DEPTH_DIR = "dense_depth"

os.makedirs(OUTPUT_BIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

MAX_DEPTH = 80.0

print(f"\nUsing device: {DEVICE}")


# ============================================================
# MODEL
# MUST MATCH TRAINING FILE EXACTLY
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

model = DepthCompletionNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("\nModel loaded successfully")


# ============================================================
# CALIBRATION
# ============================================================

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


# ============================================================
# GENERATE SPARSE DEPTH
# ============================================================

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


# ============================================================
# DEPTH -> POINT CLOUD
# ============================================================

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

            if z <= 0.1:
                continue

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            intensity = 1.0

            points.append([
                x,
                y,
                z,
                intensity
            ])

    return np.array(
        points,
        dtype=np.float32
    )


# ============================================================
# LOAD FILES
# ============================================================

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"\nTotal Images: {len(image_files)}")


# ============================================================
# GENERATE BIN FILES
# ============================================================

print("\nGenerating pseudo LiDAR bin files...\n")

for image_path in tqdm(image_files):

    file_id = Path(image_path).stem

    velodyne_path = os.path.join(
        VELODYNE_DIR,
        file_id + ".bin"
    )

    calib_path = os.path.join(
        CALIB_DIR,
        file_id + ".txt"
    )

    # ========================================================
    # LOAD RGB
    # ========================================================

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

    # ========================================================
    # SPARSE DEPTH
    # ========================================================

    sparse_depth = generate_sparse_depth_map(
        velodyne_path,
        calib_path,
        image.shape
    )

    sparse_depth = cv2.resize(
        sparse_depth,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    # ========================================================
    # MODEL INPUT
    # ========================================================

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

    # ========================================================
    # PREDICT DENSE DEPTH
    # ========================================================

    with torch.no_grad():

        pred_depth = model(input_tensor)

    pred_depth = pred_depth.squeeze().cpu().numpy()

    # ========================================================
    # SAVE DENSE DEPTH
    # ========================================================

    depth_save_path = os.path.join(
        OUTPUT_DEPTH_DIR,
        file_id + ".npy"
    )

    np.save(
        depth_save_path,
        pred_depth
    )

    # ========================================================
    # DEPTH -> POINT CLOUD
    # ========================================================

    point_cloud = depth_to_pointcloud(
        pred_depth
    )

    # ========================================================
    # SAVE BIN
    # ========================================================

    bin_save_path = os.path.join(
        OUTPUT_BIN_DIR,
        file_id + ".bin"
    )

    point_cloud.astype(
        np.float32
    ).tofile(bin_save_path)

print("\n================================================")
print("DONE")
print("================================================")

print("\nDense Depth Saved To:")
print(OUTPUT_DEPTH_DIR)

print("\nPseudo LiDAR BIN Saved To:")
print(OUTPUT_BIN_DIR)

print("\nNow open the generated .bin files")
print("inside your LiDAR visualizer.")