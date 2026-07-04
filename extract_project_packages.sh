#!/bin/bash

OUTPUT="/DATA/suhani/project_packages.txt"

# Common packages for KITTI/LiDAR/3D vision projects
cat > "$OUTPUT" << 'PKGS'
numpy
scipy
matplotlib
opencv-python
opencv-contrib-python
pillow
pyyaml
easydict
tqdm
torch
torchvision
pytorch3d
open3d
mayavi
pyqt5
vtk
scikit-learn
scikit-image
pandas
seaborn
jupyter
ipython
tensorboard
tensorflow
keras
laspy
pypcd
python-pcl
mayavi
numba
cython
PKGS

echo "Package list saved to $OUTPUT"
