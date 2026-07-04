# ================================================================
# GENERATE DEPTH DATASET (.NPY)
# USING TRAINED RESNET34 U-NET MODEL
# ================================================================

import os
import glob
import cv2
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path

import torch.nn as nn
import torchvision.models as models


# ================================================================
# CONFIGURATION
# ================================================================

KITTI_PATH = "/DATA/suhani/kitti_object/training"

IMAGE_DIR = os.path.join(KITTI_PATH, "image_2")

MODEL_PATH = "models/best_depth_model.pth"

OUTPUT_DEPTH_DIR = "generated_depth_npy"

os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# ================================================================
# LOAD IMAGE FILES
# ================================================================

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)

print(f"Total RGB Images: {len(image_files)}")


# ================================================================
# CONV BLOCK
# ================================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


# ================================================================
# UP BLOCK
# ================================================================

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:

            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv(x)

        return x


# ================================================================
# RESNET34 U-NET MODEL
# ================================================================

class ResNetUNetDepth(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet34(
            weights=None
        )

        # =====================================================
        # ENCODER
        # =====================================================

        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # =====================================================
        # DECODER
        # =====================================================

        self.up4 = UpBlock(512, 256, 256)

        self.up3 = UpBlock(256, 128, 128)

        self.up2 = UpBlock(128, 64, 64)

        self.up1 = UpBlock(64, 64, 32)

        self.final_up = nn.Sequential(

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(

            nn.Conv2d(
                16,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x0 = self.initial(x)

        x1 = self.maxpool(x0)

        x1 = self.encoder1(x1)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        d4 = self.up4(x4, x3)

        d3 = self.up3(d4, x2)

        d2 = self.up2(d3, x1)

        d1 = self.up1(d2, x0)

        out = self.final_up(d1)

        out = self.final(out)

        return out


# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading trained model...")

model = ResNetUNetDepth().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ================================================================
# GENERATE DEPTH MAPS
# ================================================================

print("\nGenerating depth dataset...\n")

with torch.no_grad():

    for image_path in tqdm(image_files):

        file_id = Path(image_path).stem

        # =====================================================
        # LOAD IMAGE
        # =====================================================

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_height, original_width = image.shape[:2]

        image_resized = cv2.resize(
            image,
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        image_resized = (
            image_resized.astype(np.float32) / 255.0
        )

        # =====================================================
        # TO TENSOR
        # =====================================================

        tensor = torch.tensor(
            image_resized,
            dtype=torch.float32
        ).permute(2, 0, 1)

        tensor = tensor.unsqueeze(0).to(DEVICE)

        # =====================================================
        # PREDICT DEPTH
        # =====================================================

        prediction = model(tensor)

        depth_map = prediction.squeeze().cpu().numpy()

        # =====================================================
        # RESIZE BACK TO ORIGINAL
        # =====================================================

        depth_map = cv2.resize(
            depth_map,
            (original_width, original_height)
        )

        # =====================================================
        # SAVE AS .NPY
        # =====================================================

        save_path = os.path.join(
            OUTPUT_DEPTH_DIR,
            file_id + ".npy"
        )

        np.save(save_path, depth_map)


print("\n=================================================")
print("DEPTH DATASET GENERATION COMPLETED")
print("=================================================")

print(f"\nSaved depth maps to:")
print(OUTPUT_DEPTH_DIR)

print("\nEach file contains:")
print("- Predicted dense depth map")
print("- Format: .npy")
print("- Shape: Original KITTI image resolution")

print("\nExample:")
print("000001.npy")
print("000002.npy")

print("\nPipeline:")
print("RGB -> Trained ResNet34 U-Net -> Depth .NPY")