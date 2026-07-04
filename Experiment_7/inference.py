import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm
from pathlib import Path

from project.Experiment_7.model import build_model


# ==========================================================
# CONFIG
# ==========================================================

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

RESULTS_DIR = "results"

RANGE_DIR = os.path.join(
    RESULTS_DIR,
    "predicted_range"
)

PLY_DIR = os.path.join(
    RESULTS_DIR,
    "pointclouds"
)

BIN_DIR = os.path.join(
    RESULTS_DIR,
    "generated_bins"
)

VIS_DIR = os.path.join(
    RESULTS_DIR,
    "visualizations"
)

for d in [
    RANGE_DIR,
    PLY_DIR,
    BIN_DIR,
    VIS_DIR
]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# MODEL
# ==========================================================

print("Loading model...")

model = build_model(config)

checkpoint = torch.load(
    "models/best_model.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("Model loaded.")

# ==========================================================
# DATA
# ==========================================================

image_dir = os.path.join(
    config["dataset"]["root"],
    config["dataset"]["image_dir"]
)

image_files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png")
])

# ==========================================================
# RANGE -> POINT CLOUD
# ==========================================================

def range_to_pointcloud(
    range_image,
    max_range=120.0,
    fov_up=3.0,
    fov_down=-25.0
):

    H, W = range_image.shape

    range_image = range_image * max_range

    yaw = np.linspace(
        -np.pi,
        np.pi,
        W
    )

    pitch = np.linspace(
        np.deg2rad(fov_down),
        np.deg2rad(fov_up),
        H
    )

    pitch, yaw = np.meshgrid(
        pitch,
        yaw,
        indexing="ij"
    )

    x = (
        range_image
        * np.cos(pitch)
        * np.cos(yaw)
    )

    y = (
        range_image
        * np.cos(pitch)
        * np.sin(yaw)
    )

    z = (
        range_image
        * np.sin(pitch)
    )

    points = np.stack(
        [x, y, z],
        axis=-1
    )

    points = points.reshape(
        -1,
        3
    )

    valid = (
        np.linalg.norm(
            points,
            axis=1
        ) > 0.1
    )

    points = points[valid]

    return points.astype(
        np.float32
    )

# ==========================================================
# SAVE PLY
# ==========================================================

def save_ply(
    points,
    filename
):

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    o3d.io.write_point_cloud(
        filename,
        pcd
    )

# ==========================================================
# SAVE KITTI BIN
# ==========================================================

def save_bin(
    points,
    filename
):

    intensity = np.ones(
        (
            points.shape[0],
            1
        ),
        dtype=np.float32
    )

    output = np.hstack(
        [
            points,
            intensity
        ]
    )

    output.astype(
        np.float32
    ).tofile(
        filename
    )

# ==========================================================
# SAVE VISUALIZATION
# ==========================================================

def save_visualization(
    points,
    filename
):

    fig = plt.figure(
        figsize=(8,8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        s=0.1
    )

    ax.set_title(
        "Generated Point Cloud"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

# ==========================================================
# INFERENCE
# ==========================================================

print(
    f"\nProcessing {len(image_files)} images..."
)

for image_path in tqdm(image_files):

    file_id = Path(
        image_path
    ).stem

    image = cv2.imread(
        image_path
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (640,192)
    )

    image = (
        image.astype(
            np.float32
        ) / 255.0
    )

    image_tensor = (
        torch.tensor(
            image
        )
        .permute(2,0,1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    with torch.no_grad():

        pred_range = model(
            image_tensor
        )

    pred_range = (
        pred_range
        .squeeze()
        .cpu()
        .numpy()
    )

    # ==========================================
    # SAVE RANGE IMAGE
    # ==========================================

    np.save(

        os.path.join(
            RANGE_DIR,
            file_id + ".npy"
        ),

        pred_range
    )

    # ==========================================
    # RANGE -> POINT CLOUD
    # ==========================================

    points = range_to_pointcloud(
        pred_range,
        max_range=config[
            "sensor"
        ][
            "max_range"
        ],
        fov_up=config[
            "sensor"
        ][
            "fov_up"
        ],
        fov_down=config[
            "sensor"
        ][
            "fov_down"
        ]
    )

    # ==========================================
    # SAVE PLY
    # ==========================================

    save_ply(

        points,

        os.path.join(
            PLY_DIR,
            file_id + ".ply"
        )
    )

    # ==========================================
    # SAVE BIN
    # ==========================================

    save_bin(

        points,

        os.path.join(
            BIN_DIR,
            file_id + ".bin"
        )
    )

    # ==========================================
    # SAVE PNG
    # ==========================================

    save_visualization(

        points,

        os.path.join(
            VIS_DIR,
            file_id + ".png"
        )
    )

print("\n===================================")
print("Inference Complete")
print("===================================")

print("\nSaved:")

print(RANGE_DIR)
print(PLY_DIR)
print(BIN_DIR)
print(VIS_DIR)