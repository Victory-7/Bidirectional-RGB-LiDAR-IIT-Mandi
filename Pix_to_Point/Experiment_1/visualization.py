import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "best_point_model.pth"

DEPTH_DIR = "dataset/depth"

OUTPUT_DIR = "visualization_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_POINTS = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")

# ============================================================
# MODEL (EXACT SAME AS TRAINING)
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
# PREDICT POINT CLOUD
# ============================================================

print("\nGenerating point cloud...")

with torch.no_grad():

    pred_cloud = model(depth_tensor)

pred_cloud = pred_cloud.squeeze(0).cpu().numpy()

print("Prediction complete!")

# ============================================================
# VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(14, 6))

# ------------------------------------------------------------
# DEPTH MAP
# ------------------------------------------------------------

ax1 = fig.add_subplot(121)

ax1.imshow(original_depth, cmap='plasma')

ax1.set_title("Input Depth Map")

ax1.axis("off")

# ------------------------------------------------------------
# POINT CLOUD
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    122,
    projection='3d'
)

ax2.scatter(
    pred_cloud[:, 0],
    pred_cloud[:, 1],
    pred_cloud[:, 2],
    s=1
)

ax2.set_title("Predicted Point Cloud")

# axis labels
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

# ============================================================
# SAVE
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

print("\nDone.")