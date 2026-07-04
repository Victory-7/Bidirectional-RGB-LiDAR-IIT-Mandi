import os
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
from torchvision import models


"""
===========================================================
RGB -> DEPTH GENERATION USING TRAINED MODEL
===========================================================

INPUT:
RGB Images

OUTPUT:
Predicted Depth Maps

Generated Files:
----------------
generated_depth/
    depth_png/
    depth_npy/
    visualizations/

===========================================================
"""


# =========================================================
# CONFIG
# =========================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# RGB image folder
INPUT_IMAGE_DIR = "/DATA/suhani/kitti_object/training/image_2"

# Trained model
MODEL_PATH = "models/best_depth_model.pth"

# Output folders
OUTPUT_DIR = "generated_depth"

PNG_DIR = os.path.join(OUTPUT_DIR, "depth_png")
NPY_DIR = os.path.join(OUTPUT_DIR, "depth_npy")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

print(f"\nUsing device: {DEVICE}")


# =========================================================
# MODEL
# =========================================================

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

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
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

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
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

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
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

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 1, 1),

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


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading trained model...")

model = ResNetUNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")


# =========================================================
# LOAD IMAGES
# =========================================================

image_files = sorted(
    glob.glob(
        os.path.join(INPUT_IMAGE_DIR, "*.png")
    )
)

print(f"\nTotal Images Found: {len(image_files)}")


# =========================================================
# DEPTH GENERATION
# =========================================================

print("\nGenerating depth maps...\n")

for image_path in tqdm(image_files):

    file_id = Path(image_path).stem

    # -----------------------------------------------------
    # LOAD IMAGE
    # -----------------------------------------------------

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_h, original_w = image_rgb.shape[:2]

    resized = cv2.resize(
        image_rgb,
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )

    normalized = resized.astype(
        np.float32
    ) / 255.0

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32
    ).permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # -----------------------------------------------------
    # PREDICT DEPTH
    # -----------------------------------------------------

    with torch.no_grad():

        prediction = model(tensor)

    depth = prediction.squeeze().cpu().numpy()

    # -----------------------------------------------------
    # RESIZE BACK
    # -----------------------------------------------------

    depth = cv2.resize(
        depth,
        (original_w, original_h)
    )

    # -----------------------------------------------------
    # SAVE NPY
    # -----------------------------------------------------

    npy_path = os.path.join(
        NPY_DIR,
        file_id + ".npy"
    )

    np.save(npy_path, depth)

    # -----------------------------------------------------
    # SAVE PNG
    # -----------------------------------------------------

    depth_normalized = (
        depth - depth.min()
    ) / (
        depth.max() - depth.min() + 1e-8
    )

    depth_uint8 = (
        depth_normalized * 255
    ).astype(np.uint8)

    png_path = os.path.join(
        PNG_DIR,
        file_id + ".png"
    )

    cv2.imwrite(
        png_path,
        depth_uint8
    )

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.imshow(image_rgb)

    plt.title("RGB")

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(depth, cmap="plasma")

    plt.title("Predicted Depth")

    plt.axis("off")

    vis_path = os.path.join(
        VIS_DIR,
        file_id + ".png"
    )

    plt.savefig(vis_path)

    plt.close()

print("\n================================================")
print("DEPTH GENERATION COMPLETED")
print("================================================")

print("\nGenerated:")
print("- Depth PNG files")
print("- Raw depth NPY files")
print("- Visualization images")

print("\nPipeline:")
print("RGB -> Predicted Depth")

print("\nNext Stage:")
print("Depth -> Point Cloud Projection")