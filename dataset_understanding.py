import os
from pathlib import Path
from collections import defaultdict
import json

try:
    import pandas as pd
except ImportError:
    pd = None


class KITTIDatasetExplorer:


    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        self.stats = defaultdict(dict)


    def print_header(self):
        print("=" * 70)
        print("KITTI DATASET EXPLORER")
        print("=" * 70)
        print(f"Dataset Path: {self.dataset_path}")
        print()

    def dataset_overview(self):
        print("\n[1] DATASET OVERVIEW")
        print("-" * 70)

        folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]

        if not folders:
            print("No folders found.")
            return

        print("Top-level folders:")
        for folder in folders:
            print(f"  - {folder.name}")

    # ------------------------------------------------------------
    # FILE COUNTING
    # ------------------------------------------------------------

    def count_files(self, folder):
        total = 0
        extensions = defaultdict(int)

        for root, dirs, files in os.walk(folder):
            for file in files:
                total += 1
                ext = Path(file).suffix.lower()
                extensions[ext] += 1

        return total, dict(extensions)

    def analyze_folders(self):
        print("\n[2] FOLDER ANALYSIS")
        print("-" * 70)

        for folder in self.dataset_path.iterdir():
            if folder.is_dir():
                total, extensions = self.count_files(folder)

                print(f"\nFolder: {folder.name}")
                print(f"Total Files: {total}")
                print("File Types:")

                for ext, count in sorted(extensions.items()):
                    ext_name = ext if ext else "[NO EXTENSION]"
                    print(f"   {ext_name:10s} -> {count}")

                self.stats[folder.name]["total_files"] = total
                self.stats[folder.name]["extensions"] = extensions

    # ------------------------------------------------------------
    # TRAIN / VAL / TEST SPLIT ANALYSIS
    # ------------------------------------------------------------

    def analyze_splits(self):
        print("\n[3] TRAIN / VALIDATION / TEST ANALYSIS")
        print("-" * 70)

        split_keywords = [
            "train",
            "training",
            "val",
            "validation",
            "test",
            "testing"
        ]

        found = False

        for root, dirs, files in os.walk(self.dataset_path):
            current = Path(root)

            for keyword in split_keywords:
                if keyword.lower() in current.name.lower():
                    found = True

                    file_count = len(files)

                    print(f"\nSplit Folder: {current.name}")
                    print(f"Path: {current}")
                    print(f"Number of files: {file_count}")

        if not found:
            print("No explicit train/validation/test folders detected.")
            print("You may need custom split files.")

    # ------------------------------------------------------------
    # IMAGE ANALYSIS
    # ------------------------------------------------------------

    def analyze_images(self):
        print("\n[4] IMAGE ANALYSIS")
        print("-" * 70)

        image_extensions = [".png", ".jpg", ".jpeg"]
        image_count = 0

        sample_images = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                ext = Path(file).suffix.lower()

                if ext in image_extensions:
                    image_count += 1

                    if len(sample_images) < 5:
                        sample_images.append(os.path.join(root, file))

        print(f"Total image files: {image_count}")

        print("\nSample image paths:")
        for img in sample_images:
            print(f"  {img}")

    # ------------------------------------------------------------
    # LIDAR ANALYSIS
    # ------------------------------------------------------------

    def analyze_lidar(self):
        print("\n[5] LIDAR / POINT CLOUD ANALYSIS")
        print("-" * 70)

        lidar_extensions = [".bin", ".pcd"]
        lidar_count = 0

        sample_lidar = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                ext = Path(file).suffix.lower()

                if ext in lidar_extensions:
                    lidar_count += 1

                    if len(sample_lidar) < 5:
                        sample_lidar.append(os.path.join(root, file))

        print(f"Total LiDAR files: {lidar_count}")

        print("\nSample LiDAR paths:")
        for lidar in sample_lidar:
            print(f"  {lidar}")

    # ------------------------------------------------------------
    # LABEL ANALYSIS
    # ------------------------------------------------------------

    def analyze_labels(self):
        print("\n[6] LABEL ANALYSIS")
        print("-" * 70)

        label_files = []
        object_classes = defaultdict(int)

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if "label" in root.lower() and file.endswith(".txt"):
                    path = os.path.join(root, file)
                    label_files.append(path)

        print(f"Total label files: {len(label_files)}")

        for label_file in label_files[:50]:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()

                    if len(parts) > 0:
                        obj_class = parts[0]
                        object_classes[obj_class] += 1

            except Exception:
                pass

        if object_classes:
            print("\nDetected Object Classes:")
            for cls, count in sorted(object_classes.items()):
                print(f"  {cls:15s} -> {count}")
        else:
            print("No label objects detected.")

    # ------------------------------------------------------------
    # CALIBRATION ANALYSIS
    # ------------------------------------------------------------

    def analyze_calibration(self):
        print("\n[7] CALIBRATION FILE ANALYSIS")
        print("-" * 70)

        calib_files = []

        for root, dirs, files in os.walk(self.dataset_path):
            if "calib" in root.lower():
                for file in files:
                    calib_files.append(os.path.join(root, file))

        print(f"Total calibration files: {len(calib_files)}")

        if calib_files:
            print("\nSample calibration files:")
            for file in calib_files[:5]:
                print(f"  {file}")

    # ------------------------------------------------------------
    # DATASET SIZE
    # ------------------------------------------------------------

    def get_dataset_size(self):
        print("\n[8] DATASET SIZE ANALYSIS")
        print("-" * 70)

        total_size = 0

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(path)
                except Exception:
                    pass

        gb = total_size / (1024 ** 3)
        print(f"Approximate dataset size: {gb:.2f} GB")

    # ------------------------------------------------------------
    # KITTI-SPECIFIC STRUCTURE DETECTION
    # ------------------------------------------------------------

    def detect_kitti_structure(self):
        print("\n[9] KITTI STRUCTURE DETECTION")
        print("-" * 70)

        common_kitti_folders = [
            "image_2",
            "image_3",
            "velodyne",
            "label_2",
            "calib"
        ]

        found = []

        for root, dirs, files in os.walk(self.dataset_path):
            for directory in dirs:
                if directory in common_kitti_folders:
                    found.append(os.path.join(root, directory))

        if found:
            print("Detected KITTI-style folders:")
            for f in found:
                print(f"  {f}")
        else:
            print("Standard KITTI structure not fully detected.")

    # ------------------------------------------------------------
    # SUMMARY REPORT
    # ------------------------------------------------------------

    def generate_summary(self):
        print("\n[10] FINAL SUMMARY")
        print("-" * 70)

        print("This dataset appears to contain:")
        print("  - RGB image data")
        print("  - LiDAR point cloud data")
        print("  - Calibration files")
        print("  - Annotation labels")
        print("  - Training/testing split structure")

        print("\nPotential uses:")
        print("  - Depth estimation")
        print("  - Pseudo-LiDAR generation")
        print("  - Object detection")
        print("  - Sensor fusion")
        print("  - Autonomous navigation")
        print("  - Swarm intelligence research")

    # ------------------------------------------------------------
    # EXPORT REPORT
    # ------------------------------------------------------------

    def export_report(self, output_file="kitti_dataset_report.json"):
        print("\n[11] EXPORTING REPORT")
        print("-" * 70)

        try:
            with open(output_file, 'w') as f:
                json.dump(self.stats, f, indent=4)

            print(f"Report exported to: {output_file}")

        except Exception as e:
            print(f"Export failed: {e}")

    # ------------------------------------------------------------
    # RUN COMPLETE ANALYSIS
    # ------------------------------------------------------------

    def run_all(self):
        self.print_header()
        self.dataset_overview()
        self.analyze_folders()
        self.analyze_splits()
        self.analyze_images()
        self.analyze_lidar()
        self.analyze_labels()
        self.analyze_calibration()
        self.get_dataset_size()
        self.detect_kitti_structure()
        self.generate_summary()
        self.export_report()


# ==================================================================
# MAIN EXECUTION
# ==================================================================

if __name__ == "__main__":

    print("\nEnter your KITTI dataset path.")
    print("Example:")
    print("  D:/KITTI")
    print("  /home/user/KITTI")
    print()

    # Your KITTI dataset path
    dataset_path = "/DATA/suhani/kitti_object/training"

    explorer = KITTIDatasetExplorer(dataset_path)
    explorer.run_all()
