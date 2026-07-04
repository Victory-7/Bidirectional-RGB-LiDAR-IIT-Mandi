import os
import glob
import numpy as np
from tqdm import tqdm


# =========================================================
# KITTI VELODYNE PATH
# =========================================================

VELODYNE_DIR = "/DATA/suhani/kitti_object/training/velodyne"

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = "generated_depth_npy/pointcloud"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD ALL BIN FILES
# =========================================================

bin_files = sorted(
    glob.glob(
        os.path.join(
            VELODYNE_DIR,
            "*.bin"
        )
    )
)

print(f"\nFound {len(bin_files)} LiDAR files")


# =========================================================
# CONVERT KITTI BIN -> NPY
# =========================================================

for bin_path in tqdm(bin_files):

    # -----------------------------------------------------
    # LOAD KITTI LIDAR
    # -----------------------------------------------------

    lidar = np.fromfile(
        bin_path,
        dtype=np.float32
    ).reshape(-1, 4)

    # -----------------------------------------------------
    # KEEP XYZ ONLY
    # -----------------------------------------------------

    points = lidar[:, :3]

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    filename = os.path.basename(bin_path)

    filename = filename.replace(
        ".bin",
        ".npy"
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    np.save(save_path, points)

print("\n========================================")
print("POINT CLOUD DATASET CREATED")
print("========================================")

print(f"\nSaved point clouds to:")
print(OUTPUT_DIR)