from project.Experiment_7.losses import CombinedLoss
import torch
import yaml

with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

criterion = CombinedLoss(cfg)

pred = torch.rand(
    2,
    1,
    64,
    2048
)

gt = torch.rand(
    2,
    1,
    64,
    2048
)

loss, metrics = criterion(
    pred,
    gt
)

print(loss)
print(metrics)