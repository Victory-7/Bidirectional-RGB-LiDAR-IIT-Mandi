import torch
import cv2
import numpy as np

model = torch.hub.load(
    "yvanyin/metric3d",
    "metric3d_vit_small",
    pretrain=True
)

model.cuda()
model.eval()

img_path = "/home/teaching/Suhani/project/kitti_object/training/image_2/000000.png"

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Image Shape:", img.shape)

try:
    depth = model.inference({"input": img})
    print("Inference Success")
    print(type(depth))
except Exception as e:
    print("ERROR:")
    print(e)