import numpy as np
import glob
import os

pcd_dir = "logs/cam2lidar/2026-06-22T19-39-37_config/samples/00019656/2026-06-23-13-02-26/pcd"

for txt_file in glob.glob(os.path.join(pcd_dir, "*.txt")):

    data = np.loadtxt(txt_file)

    xyz = data[:, :3]

    points = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    points[:, :3] = xyz
    points[:, 3] = 1.0

    bin_file = txt_file.replace(".txt", ".bin")
    points.tofile(bin_file)

    print("Saved:", bin_file)
