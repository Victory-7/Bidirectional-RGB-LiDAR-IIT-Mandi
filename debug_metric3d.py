import torch
import inspect

model = torch.hub.load(
    "yvanyin/metric3d",
    "metric3d_vit_small",
    pretrain=True
)

print("\n=== INFERENCE SIGNATURE ===")
print(inspect.signature(model.inference))

print("\n=== INFERENCE SOURCE ===")
try:
    print(inspect.getsource(model.inference))
except Exception as e:
    print("Could not get source:", e)

print("\n=== MODEL CLASS ===")
print(type(model))