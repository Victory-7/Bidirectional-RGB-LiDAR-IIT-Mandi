import os
import cv2
import yaml
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


# ==========================================================
# CONFIG
# ==========================================================

def load_config(config_path="config.yaml"):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ==========================================================
# RANGE IMAGE PROJECTION
# ==========================================================

def pointcloud_to_range_image(
    points,
    H=64,
    W=2048,
    fov_up=3.0,
    fov_down=-25.0,
    max_range=120.0
):

    range_image = np.zeros(
        (H, W),
        dtype=np.float32
    )

    xyz = points[:, :3]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    depth = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    valid = (
        (depth > 1.0) &
        (depth < max_range)
    )

    x = x[valid]
    y = y[valid]
    z = z[valid]
    depth = depth[valid]

    yaw = np.arctan2(y, x)

    pitch = np.arcsin(
        z / depth
    )

    fov_up_rad = np.radians(fov_up)
    fov_down_rad = np.radians(fov_down)

    total_fov = abs(
        fov_down_rad
    ) + abs(
        fov_up_rad
    )

    proj_x = (
        0.5 *
        (yaw / np.pi + 1.0)
    )

    proj_y = (
        1.0 -
        (
            pitch +
            abs(fov_down_rad)
        ) /
        total_fov
    )

    proj_x *= W
    proj_y *= H

    proj_x = np.clip(
        proj_x.astype(np.int32),
        0,
        W - 1
    )

    proj_y = np.clip(
        proj_y.astype(np.int32),
        0,
        H - 1
    )

    order = np.argsort(
        depth
    )[::-1]

    proj_x = proj_x[order]
    proj_y = proj_y[order]
    depth = depth[order]

    range_image[
        proj_y,
        proj_x
    ] = depth

    range_image /= max_range

    return range_image


# ==========================================================
# DATASET
# ==========================================================

class LiDARRangeDataset(Dataset):

    def __init__(
        self,
        image_files,
        config
    ):

        self.image_files = image_files
        self.cfg = config

        root = config["dataset"]["root"]

        self.velodyne_dir = os.path.join(
            root,
            config["dataset"]["velodyne_dir"]
        )

        self.H = config["sensor"][
            "range_image_height"
        ]

        self.W = config["sensor"][
            "range_image_width"
        ]

        self.max_range = config[
            "sensor"
        ]["max_range"]

        self.fov_up = config[
            "sensor"
        ]["fov_up"]

        self.fov_down = config[
            "sensor"
        ]["fov_down"]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        image_path = self.image_files[idx]

        file_id = Path(
            image_path
        ).stem

        lidar_path = os.path.join(
            self.velodyne_dir,
            file_id + ".bin"
        )

        image = cv2.imread(
            image_path
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (640, 192)
        )

        image = (
            image.astype(
                np.float32
            ) / 255.0
        )

        lidar = np.fromfile(
            lidar_path,
            dtype=np.float32
        ).reshape(-1, 4)

        range_image = (
            pointcloud_to_range_image(
                lidar,
                H=self.H,
                W=self.W,
                fov_up=self.fov_up,
                fov_down=self.fov_down,
                max_range=self.max_range
            )
        )

        rgb_tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).permute(
            2,
            0,
            1
        )

        range_tensor = torch.tensor(
            range_image,
            dtype=torch.float32
        ).unsqueeze(0)

        return (
            rgb_tensor,
            range_tensor
        )


# ==========================================================
# TRAIN / VAL SPLIT
# ==========================================================

def build_datasets(
    config
):

    root = config["dataset"]["root"]

    image_dir = os.path.join(
        root,
        config["dataset"]["image_dir"]
    )

    image_files = sorted([
        os.path.join(
            image_dir,
            f
        )
        for f in os.listdir(
            image_dir
        )
        if f.endswith(".png")
    ])

    train_files, val_files = train_test_split(
        image_files,
        test_size=(
            1.0 -
            config["dataset"][
                "train_split"
            ]
        ),
        shuffle=True,
        random_state=config[
            "dataset"
        ]["random_seed"]
    )

    train_dataset = LiDARRangeDataset(
        train_files,
        config
    )

    val_dataset = LiDARRangeDataset(
        val_files,
        config
    )

    return (
        train_dataset,
        val_dataset
    )