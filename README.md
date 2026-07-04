# 🚗 RGB-to-LiDAR Generation Framework

> **A complete monocular RGB-to-LiDAR generation pipeline for producing KITTI-compatible LiDAR point clouds using deep metric depth estimation and 3D reconstruction.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Open3D](https://img.shields.io/badge/Open3D-Point%20Clouds-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-KITTI-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

# 📌 Overview

This project presents a complete **RGB-to-LiDAR generation framework** capable of converting a single monocular RGB image into a **KITTI-compatible LiDAR point cloud**.

Unlike traditional approaches that require expensive LiDAR sensors, this framework reconstructs dense 3D geometry using deep monocular depth estimation followed by camera geometry projection and point cloud generation.

The generated outputs can be directly used for:

- Autonomous Driving
- 3D Scene Reconstruction
- LiDAR Dataset Generation
- Sensor Failure Recovery
- Diffusion Model Training
- Robotics Perception

---

# 🎯 Project Goals

The primary objective is to design an end-to-end pipeline that converts:

```
RGB Image
      │
      ▼
Metric3D
      │
      ▼
Metric Depth Map
      │
      ▼
3D Geometry Reconstruction
      │
      ▼
Dense Point Cloud
      │
      ▼
KITTI Compatible LiDAR (.bin)
```

---

# ✨ Features

- Monocular RGB to LiDAR conversion
- Metric depth estimation using Metric3D
- Camera intrinsic calibration support
- Dense point cloud reconstruction
- Sparse Velodyne-like LiDAR generation
- KITTI compatible output format
- GPU accelerated inference
- Automatic batch processing
- Multiple visualization outputs
- Complete dataset generation pipeline

---

# 🏗 Pipeline Architecture

```
RGB Image
      │
      ▼
Preprocessing
 • Resize
 • Padding
 • Normalization
 • Intrinsic Scaling
      │
      ▼
Metric3D Network
      │
      ▼
Metric Depth Prediction
      │
      ▼
Depth Post-processing
      │
      ▼
Camera Projection
      │
      ▼
Dense 3D Point Cloud
      │
      ▼
LiDAR Conversion
      │
      ├────────► Dense BIN
      ├────────► Sparse BIN
      ├────────► PLY
      ├────────► NPY
      ├────────► PNG
      └────────► Heatmaps
```

---

# 📂 Dataset

The project uses the **KITTI Object Detection Dataset**.

## Dataset Statistics

| Item | Count |
|------|-------|
| RGB Images | 7,481 |
| LiDAR Point Clouds | 7,481 |
| Calibration Files | 7,481 |
| Labels | 7,481 |
| Dataset Size | 19.1 GB |

Supported object classes include:

- Car
- Pedestrian
- Cyclist
- Van
- DontCare

---

# ⚙ Technical Contributions

The framework includes more than **15 engineering contributions**, including:

- Metric3D integration for monocular metric depth estimation
- KITTI calibration parser
- Camera intrinsic scaling
- Image preprocessing pipeline
- GPU-based depth prediction
- Scale correction
- Dense 3D reconstruction
- KITTI compatible Velodyne export
- Dense LiDAR generation
- Sparse LiDAR simulation
- Multi-format output generation
- Batch processing for all KITTI images
- CUDA acceleration
- Vectorized processing
- Robust error handling

---

# 📁 Outputs

The pipeline generates multiple output formats:

```
outputs/
│
├── depth_png/
├── depth_npy/
├── heatmaps/
├── pointcloud_ply/
├── dense_bin/
├── sparse_bin/
└── visualization/
```

Generated formats include:

- PNG
- NPY
- PLY
- BIN
- Depth Heatmaps

---

# 🚀 Processing Flow

```
RGB Image
      │
      ▼
Load Camera Calibration
      │
      ▼
Image Preprocessing
      │
      ▼
Metric3D Inference
      │
      ▼
Metric Depth Map
      │
      ▼
Depth Refinement
      │
      ▼
3D Point Cloud Reconstruction
      │
      ▼
Point Cloud Optimization
      │
      ▼
KITTI LiDAR Generation
      │
      ▼
Save Outputs
```

---

# 📊 Qualitative Results

The framework produces:

- High-quality metric depth maps
- Dense point clouds
- KITTI-compatible LiDAR scans
- Depth heatmaps
- Sparse Velodyne-style point clouds

---

# 💡 Applications

This framework can be applied to:

- Autonomous Driving
- Sensor Fusion
- Robotics
- SLAM
- 3D Reconstruction
- Dataset Generation
- LiDAR Diffusion Models
- Perception Research

---

# 📈 Advantages

- No physical LiDAR required
- Cost-effective data generation
- KITTI compatible
- End-to-end automation
- Fast GPU inference
- High-quality geometry reconstruction
- Easily extensible

---

# 🔮 Future Improvements

Potential future work includes:

- Real-time inference
- Multi-view RGB fusion
- Temporal consistency
- Diffusion-based refinement
- Semantic-aware LiDAR generation
- Domain adaptation
- Outdoor robustness improvements

---

# 🛠 Technologies Used

- Python
- PyTorch
- Metric3D
- OpenCV
- NumPy
- Open3D
- CUDA
- KITTI Dataset

---

# 📚 References

- **Semantics-aware Multi-modal Domain Translation: From LiDAR Point Clouds to Panoramic Color Images**  
  *Tiago Cortinhal, Fatih Kurnaz, Eren Erdal Aksoy* (ICCV Workshop 2021)

- **Towards Realistic Scene Generation with LiDAR Diffusion Models (LiDM)**  
  *CVPR 2024*

- KITTI Object Detection Benchmark

---

# 🎯 Impact

This project bridges the gap between monocular RGB imagery and LiDAR sensing by providing a practical framework for generating realistic, KITTI-compatible point clouds.

The generated LiDAR data enables downstream applications such as autonomous driving perception, dataset augmentation, sensor-failure recovery, and diffusion model training without requiring expensive LiDAR hardware.

---

# 👩‍💻 Author

**Suhani Verma**

Project Mentor: **Tarun Sir**

Instructor: **Dr. Dinesh Singh**

Indian Institute of Technology Mandi

---

# ⭐ If you find this project useful, consider giving it a star!
