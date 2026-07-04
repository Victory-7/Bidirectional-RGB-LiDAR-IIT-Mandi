import torch
import cv2

model = torch.hub.load(
    "yvanyin/metric3d",
    "metric3d_vit_small",
    pretrain=True
)

print("MODEL LOADED")

img = cv2.imread(
    "/home/teaching/Suhani/project/kitti_object/training/image_2/000000.png"
)

print(img.shape)

print(dir(model))
