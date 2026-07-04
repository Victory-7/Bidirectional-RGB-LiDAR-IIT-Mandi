import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.spatial import cKDTree

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "/home/teaching/Suhani/project/Pix_to_Point/Experiment_2/final_point_model.pth"

DEPTH_DIR = "/home/teaching/Suhani/project/Pix_to_Point/Experiment_2/dataset/depth"

GT_DIR = "/home/teaching/Suhani/project/kitti_object/training/velodyne"

OUTPUT_DIR = "visualization_results_matrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_POINTS = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")

# ============================================================
# METRICS
# ============================================================

def chamfer_distance(pc1, pc2):

    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)

    dist1, _ = tree1.query(pc2)
    dist2, _ = tree2.query(pc1)

    cd = np.mean(dist1 ** 2) + np.mean(dist2 ** 2)

    return cd


def rmse(pc1, pc2):

    min_points = min(len(pc1), len(pc2))

    pc1 = pc1[:min_points]
    pc2 = pc2[:min_points]

    return np.sqrt(
        np.mean((pc1 - pc2) ** 2)
    )


def mae(pc1, pc2):

    min_points = min(len(pc1), len(pc2))

    pc1 = pc1[:min_points]
    pc2 = pc2[:min_points]

    return np.mean(
        np.abs(pc1 - pc2)
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
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = PointCloudNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully!")

# ============================================================
# LOAD DEPTH FILES
# ============================================================

depth_files = sorted(
    glob.glob(os.path.join(DEPTH_DIR, "*.npy"))
)

print(f"\nFound {len(depth_files)} depth files")

# ============================================================
# RANDOM SAMPLE
# ============================================================

sample_file = random.choice(depth_files)

print(f"\nUsing sample:\n{sample_file}")

sample_name = os.path.basename(
    sample_file
).replace(".npy", ".bin")

gt_file = os.path.join(
    GT_DIR,
    sample_name
)

print(f"\nGround Truth:\n{gt_file}")

# ============================================================
# LOAD DEPTH MAP
# ============================================================

depth = np.load(sample_file).astype(np.float32)

# normalize
depth = (
    depth - depth.min()
) / (
    depth.max() - depth.min() + 1e-8
)

original_depth = depth.copy()

# tensor
depth_tensor = torch.tensor(
    depth,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(DEVICE)

# ============================================================
# LOAD GROUND TRUTH POINT CLOUD
# ============================================================

gt_cloud = np.fromfile(
    gt_file,
    dtype=np.float32
).reshape(-1, 4)[:, :3]

print(f"\nOriginal GT points: {len(gt_cloud)}")

# ============================================================
# SAMPLE SAME NUMBER OF POINTS
# ============================================================

if len(gt_cloud) > NUM_POINTS:

    indices = np.random.choice(
        len(gt_cloud),
        NUM_POINTS,
        replace=False
    )

    gt_cloud = gt_cloud[indices]

else:

    pad = NUM_POINTS - len(gt_cloud)

    extra = gt_cloud[
        np.random.choice(
            len(gt_cloud),
            pad
        )
    ]

    gt_cloud = np.vstack([
        gt_cloud,
        extra
    ])

print(f"GT points after sampling: {len(gt_cloud)}")

# ============================================================
# PREDICT POINT CLOUD
# ============================================================

print("\nGenerating point cloud...")

with torch.no_grad():

    pred_cloud = model(depth_tensor)

pred_cloud = pred_cloud.squeeze(0).cpu().numpy()

print("Prediction complete!")

# ============================================================
# METRICS
# ============================================================

print("\nCalculating metrics...")

cd = chamfer_distance(
    pred_cloud,
    gt_cloud
)

rmse_value = rmse(
    pred_cloud,
    gt_cloud
)

mae_value = mae(
    pred_cloud,
    gt_cloud
)

print("\n================ METRICS ================")

print(f"Chamfer Distance : {cd:.6f}")

print(f"RMSE             : {rmse_value:.6f}")

print(f"MAE              : {mae_value:.6f}")

print("=========================================")

# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(16, 7))

# ------------------------------------------------------------
# DEPTH MAP
# ------------------------------------------------------------

ax1 = fig.add_subplot(121)

ax1.imshow(
    original_depth,
    cmap='plasma'
)

ax1.set_title("Input Depth Map")

ax1.axis("off")

# ------------------------------------------------------------
# POINT CLOUDS
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    122,
    projection='3d'
)

# Ground Truth
ax2.scatter(
    gt_cloud[:, 0],
    gt_cloud[:, 1],
    gt_cloud[:, 2],
    s=1,
    label='Ground Truth'
)

# Prediction
ax2.scatter(
    pred_cloud[:, 0],
    pred_cloud[:, 1],
    pred_cloud[:, 2],
    s=1,
    label='Predicted'
)

ax2.set_title("GT vs Predicted Point Cloud")

ax2.set_xlabel("X")

ax2.set_ylabel("Y")

ax2.set_zlabel("Z")

ax2.legend()

# ============================================================
# SAVE FIGURE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "prediction_visualization.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print(f"\nSaved visualization:")
print(output_path)

# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("POINT CLOUD METRICS\n")
    f.write("===================\n\n")

    f.write(f"Sample File : {sample_file}\n")
    f.write(f"GT File     : {gt_file}\n\n")

    f.write(f"Chamfer Distance : {cd:.6f}\n")
    f.write(f"RMSE             : {rmse_value:.6f}\n")
    f.write(f"MAE              : {mae_value:.6f}\n")

print(f"\nSaved metrics:")
print(metrics_path)

print("\nDone.")