import torch
import inspect

model = torch.hub.load(
    "yvanyin/metric3d",
    "metric3d_vit_small",
    pretrain=True
)

print("\n=== FORWARD SIGNATURE ===")
print(inspect.signature(model.forward))