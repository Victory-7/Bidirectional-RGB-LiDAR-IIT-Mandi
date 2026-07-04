import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms


# =========================================================
# CONFIGURATION
# =========================================================

# KITTI image folder
KITTI_IMAGE_DIR = "/DATA/suhani/kitti_object/training/image_2"

# Path to trained depth model
MODEL_PATH = "models/best_depth_model.pth"

# Output depth folder
OUTPUT_DEPTH_DIR = "dataset/depth"

# Image size (MUST MATCH TRAINING)
IMAGE_HEIGHT = 192
IMAGE_WIDTH = 640

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(OUTPUT_DEPTH_DIR, exist_ok=True)


# =========================================================
# EXACT TRAINED MODEL ARCHITECTURE
# =========================================================

class ImprovedDepthNet(nn.Module):

    def __init__(self):

        super(ImprovedDepthNet, self).__init__()

        # ENCODER

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # DECODER

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                1,
                2,
                stride=2
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

print("\nLoading trained depth model...")

model = ImprovedDepthNet().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Depth model loaded successfully")


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (IMAGE_HEIGHT, IMAGE_WIDTH)
    ),

    transforms.ToTensor()
])


# =========================================================
# LOAD IMAGE FILES
# =========================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            KITTI_IMAGE_DIR,
            "*.png"
        )
    )
)

print(f"\nTotal KITTI images found: {len(image_files)}")


# =========================================================
# GENERATE DEPTH MAPS
# =========================================================

print("\nGenerating depth maps...\n")

with torch.no_grad():

    for image_path in tqdm(image_files):

        # -------------------------------------------------
        # LOAD RGB IMAGE
        # -------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        input_tensor = transform(
            image_rgb
        ).unsqueeze(0).to(DEVICE)

        # -------------------------------------------------
        # DEPTH PREDICTION
        # -------------------------------------------------

        pred_depth = model(input_tensor)

        pred_depth = (
            pred_depth
            .squeeze()
            .cpu()
            .numpy()
        )

        # -------------------------------------------------
        # NORMALIZE DEPTH
        # -------------------------------------------------

        pred_depth = (
            pred_depth - pred_depth.min()
        ) / (
            pred_depth.max() -
            pred_depth.min() +
            1e-8
        )

        # -------------------------------------------------
        # SAVE DEPTH MAP
        # -------------------------------------------------

        filename = os.path.basename(image_path)

        filename = filename.replace(
            ".png",
            ".npy"
        )

        save_path = os.path.join(
            OUTPUT_DEPTH_DIR,
            filename
        )

        np.save(save_path, pred_depth)

print("\n================================================")
print("DEPTH GENERATION COMPLETED")
print("================================================")

print(f"\nSaved depth maps to:")
print(OUTPUT_DEPTH_DIR)

print("\nGenerated:")
print("- depth/*.npy")

print("\nNext Stage:")
print("Depth Map -> Point Cloud Training")