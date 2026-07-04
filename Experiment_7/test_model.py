import yaml
import torch

from project.Experiment_7.dataset import build_datasets
from project.Experiment_7.model import build_model

# ==========================
# LOAD CONFIG
# ==========================

with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

# ==========================
# DATASET
# ==========================

train_dataset, val_dataset = build_datasets(config)

print("Train Samples:",len(train_dataset))
print("Val Samples:",len(val_dataset))

# ==========================
# SAMPLE
# ==========================

rgb, range_gt = train_dataset[0]

print("\nRGB Shape")
print(rgb.shape)

print("\nRange Shape")
print(range_gt.shape)

# ==========================
# MODEL
# ==========================

model = build_model(config)

model.eval()

rgb = rgb.unsqueeze(0)

with torch.no_grad():

    pred = model(rgb)

print("\nPrediction Shape")
print(pred.shape)