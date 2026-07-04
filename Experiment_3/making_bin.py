import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm

"""
===========================================================
NPY TO KITTI BIN CONVERTER
===========================================================

PURPOSE:
Convert predicted refined point clouds (.npy)
into KITTI-style .bin files for visualization
and comparison with real KITTI LiDAR.

INPUT:
refined_pointclouds/
    000001.npy
    000002.npy

OUTPUT:
predicted_bin/
    000001.bin
    000002.bin

KITTI BIN FORMAT:
[x, y, z, intensity]
float32

===========================================================
"""


# =========================================================
# PATHS
# =========================================================

NPY_DIR = "refined_pointclouds"

OUTPUT_BIN_DIR = "predicted_bin"

os.makedirs(OUTPUT_BIN_DIR, exist_ok=True)


# =========================================================
# LOAD FILES
# =========================================================

npy_files = sorted(
    glob.glob(os.path.join(NPY_DIR, "*.npy"))
)

print(f"\nFound {len(npy_files)} NPY files")


# =========================================================
# CONVERT NPY -> BIN
# =========================================================

for npy_path in tqdm(npy_files):

    file_id = Path(npy_path).stem

    # -----------------------------------------------------
    # LOAD POINT CLOUD
    # -----------------------------------------------------

    points = np.load(npy_path)

    # -----------------------------------------------------
    # HANDLE SHAPES
    # -----------------------------------------------------

    # If shape is Nx3 -> add intensity
    if points.shape[1] == 3:

        intensity = np.ones(
            (points.shape[0], 1),
            dtype=np.float32
        )

        points = np.hstack((
            points,
            intensity
        ))

    # If shape already Nx4
    elif points.shape[1] == 4:

        pass

    else:

        print(
            f"Skipping {file_id} "
            f"(invalid shape: {points.shape})"
        )

        continue

    # -----------------------------------------------------
    # ENSURE FLOAT32
    # -----------------------------------------------------

    points = points.astype(np.float32)

    # -----------------------------------------------------
    # SAVE BIN FILE
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_BIN_DIR,
        file_id + ".bin"
    )

    points.tofile(output_path)

print("\n================================================")
print("CONVERSION COMPLETED")
print("================================================")

print(f"\nBIN files saved to: {OUTPUT_BIN_DIR}")

print("\nNow you can compare:")

print("Ground Truth:")
print("kitti_object/training/velodyne/*.bin")

print("\nPredicted:")
print("predicted_bin/*.bin")

print("\nYou can load both into:")
print("- Open3D")
print("- KITTI Viewer")
print("- RViz")
print("- MeshLab")
print("- CloudCompare")

print("\nThis helps identify:")
print("- Missing structures")
print("- Sparse regions")
print("- Wrong geometry")
print("- Depth scaling issues")
print("- Noise patterns")
print("- Point distribution problems")