import os
import cv2
import torch
import random
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = "/home/teaching/Suhani/project"

IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "kitti_object/training/image_2"
)

CALIB_DIR = os.path.join(
    PROJECT_ROOT,
    "kitti_object/training/calib"
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

DEPTH_NPY_DIR = os.path.join(OUTPUT_ROOT, "depth_npy")
DEPTH_PNG_DIR = os.path.join(OUTPUT_ROOT, "depth_png")
HEATMAP_DIR = os.path.join(OUTPUT_ROOT, "heatmaps")
PLY_DIR = os.path.join(OUTPUT_ROOT, "pointcloud_ply")
DENSE_BIN_DIR = os.path.join(OUTPUT_ROOT, "dense_bin")
VELODYNE_LIKE_DIR = os.path.join(OUTPUT_ROOT, "velodyne_like")

for d in [
    DEPTH_NPY_DIR,
    DEPTH_PNG_DIR,
    HEATMAP_DIR,
    PLY_DIR,
    DENSE_BIN_DIR,
    VELODYNE_LIKE_DIR
]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# GPU
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Device:", DEVICE)
print("=" * 60)

# ============================================================
# LOAD METRIC3D
# ============================================================

print("Loading Metric3D...")

model = torch.hub.load(
    "yvanyin/metric3d",
    "metric3d_vit_small",
    pretrain=True
)

model.to(DEVICE)
model.eval()

print("Metric3D Loaded")

# ============================================================
# KITTI CALIB PARSER
# ============================================================

def read_kitti_calib(calib_file):

    with open(calib_file, "r") as f:
        lines = f.readlines()

    P2 = None

    for line in lines:

        if line.startswith("P2:"):

            values = np.array(
                list(map(float, line.strip().split()[1:])),
                dtype=np.float32
            )

            P2 = values.reshape(3, 4)
            break

    if P2 is None:
        raise ValueError(f"P2 not found in {calib_file}")

    fx = P2[0, 0]
    fy = P2[1, 1]
    cx = P2[0, 2]
    cy = P2[1, 2]

    return fx, fy, cx, cy

# ============================================================
# METRIC3D PREPROCESS
# ============================================================

def preprocess_metric3d(rgb_origin, intrinsic):

    input_size = (616, 1064)

    h, w = rgb_origin.shape[:2]

    scale = min(
        input_size[0] / h,
        input_size[1] / w
    )

    rgb = cv2.resize(
        rgb_origin,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_LINEAR
    )

    intrinsic = [
        intrinsic[0] * scale,
        intrinsic[1] * scale,
        intrinsic[2] * scale,
        intrinsic[3] * scale,
    ]

    padding_color = [123.675, 116.28, 103.53]

    h2, w2 = rgb.shape[:2]

    pad_h = input_size[0] - h2
    pad_w = input_size[1] - w2

    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2

    rgb = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding_color
    )

    pad_info = [
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half
    ]

    mean = torch.tensor(
        [123.675, 116.28, 103.53]
    ).float()[:, None, None]

    std = torch.tensor(
        [58.395, 57.12, 57.375]
    ).float()[:, None, None]

    rgb = torch.from_numpy(
        rgb.transpose((2, 0, 1))
    ).float()

    rgb = (rgb - mean) / std

    rgb = rgb.unsqueeze(0).to(DEVICE)

    return rgb, intrinsic, pad_info

# ============================================================
# DEPTH INFERENCE
# ============================================================

def predict_depth(image_path, calib_path):

    fx, fy, cx, cy = read_kitti_calib(calib_path)

    rgb_origin = cv2.imread(image_path)

    rgb_origin = cv2.cvtColor(
        rgb_origin,
        cv2.COLOR_BGR2RGB
    )

    rgb, intrinsic, pad_info = preprocess_metric3d(
        rgb_origin,
        [fx, fy, cx, cy]
    )

    with torch.no_grad():

        pred_depth, confidence, output_dict = model.inference(
            {"input": rgb}
        )

    pred_depth = pred_depth.squeeze()

    pred_depth = pred_depth[
        pad_info[0]:pred_depth.shape[0]-pad_info[1],
        pad_info[2]:pred_depth.shape[1]-pad_info[3]
    ]

    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None],
        rgb_origin.shape[:2],
        mode="bilinear",
        align_corners=False
    ).squeeze()

    scale_factor = intrinsic[0] / 1000.0

    pred_depth = pred_depth * scale_factor

    pred_depth = torch.clamp(
        pred_depth,
        min=0,
        max=300
    )

    return (
        pred_depth.cpu().numpy(),
        rgb_origin,
        fx,
        fy,
        cx,
        cy
    )

# ============================================================
# DEPTH -> POINT CLOUD
# ============================================================

def depth_to_points(depth, fx, fy, cx, cy):

    h, w = depth.shape

    u, v = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    z = depth

    valid = z > 0.1

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack(
        [x, y, z],
        axis=-1
    )

    points = points[valid]

    return points.astype(np.float32)

# ============================================================
# SAVE BIN
# ============================================================

def save_bin(points, path):

    intensity = np.ones(
        (points.shape[0], 1),
        dtype=np.float32
    )

    cloud = np.concatenate(
        [points, intensity],
        axis=1
    )

    cloud.astype(np.float32).tofile(path)

# ============================================================
# MAIN LOOP
# ============================================================

images = sorted(
    Path(IMAGE_DIR).glob("*.png")
)

print("Total Images:", len(images))

for image_path in tqdm(images):

    stem = image_path.stem

    calib_path = os.path.join(
        CALIB_DIR,
        f"{stem}.txt"
    )

    try:

        depth, rgb, fx, fy, cx, cy = predict_depth(
            str(image_path),
            calib_path
        )

        # --------------------------------
        # Save depth npy
        # --------------------------------

        np.save(
            os.path.join(
                DEPTH_NPY_DIR,
                f"{stem}.npy"
            ),
            depth
        )

        # --------------------------------
        # Save depth png
        # --------------------------------

        depth_vis = (
            depth /
            (depth.max() + 1e-8)
            * 255
        ).astype(np.uint8)

        cv2.imwrite(
            os.path.join(
                DEPTH_PNG_DIR,
                f"{stem}.png"
            ),
            depth_vis
        )

        # --------------------------------
        # Save heatmap
        # --------------------------------

        plt.figure(figsize=(12,4))
        plt.imshow(depth, cmap="inferno")
        plt.axis("off")

        plt.savefig(
            os.path.join(
                HEATMAP_DIR,
                f"{stem}.png"
            ),
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()

        # --------------------------------
        # Point Cloud
        # --------------------------------

        points = depth_to_points(
            depth,
            fx,
            fy,
            cx,
            cy
        )

        cloud = o3d.geometry.PointCloud()

        cloud.points = o3d.utility.Vector3dVector(
            points
        )

        o3d.io.write_point_cloud(
            os.path.join(
                PLY_DIR,
                f"{stem}.ply"
            ),
            cloud
        )

        # --------------------------------
        # Dense BIN
        # --------------------------------

        save_bin(
            points,
            os.path.join(
                DENSE_BIN_DIR,
                f"{stem}.bin"
            )
        )

        # --------------------------------
        # Velodyne-like BIN
        # --------------------------------

        target_points = min(
            100000,
            len(points)
        )

        idx = np.random.choice(
            len(points),
            target_points,
            replace=False
        )

        sparse_points = points[idx]

        save_bin(
            sparse_points,
            os.path.join(
                VELODYNE_LIKE_DIR,
                f"{stem}.bin"
            )
        )

    except Exception as e:
        print(
            f"\nFailed {stem}: {e}"
        )

print("\nFinished.")