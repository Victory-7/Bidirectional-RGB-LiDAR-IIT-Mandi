import os
import glob
import argparse

import cv2
import yaml
import torch
import numpy as np

from omegaconf import OmegaConf

from lidm.utils.misc_utils import instantiate_from_config
from lidm.utils.lidar_utils import range2pcd


def load_model(config_path, ckpt_path):
    cfg = OmegaConf.load(config_path)

    model = instantiate_from_config(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    model.cuda().eval()

    return model, cfg


def preprocess(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (1024,64))

    img = img.astype(np.float32)/255.0

    img = torch.from_numpy(img).permute(2,0,1)

    splits = torch.chunk(img,4,dim=2)

    splits = [s.unsqueeze(0).cuda() for s in splits]

    return splits


@torch.no_grad()
def infer(model,cfg,image_dir,out_dir):

    os.makedirs(out_dir,exist_ok=True)

    images = sorted(glob.glob(os.path.join(image_dir,"*.png")))

    print("Found",len(images),"images")

    for image_path in images:

        camera = preprocess(image_path)

        cond = model.get_learned_conditioning(camera)

        latent,_ = model.sample_log(
            cond=cond,
            batch_size=1,
            ddim=True,
            ddim_steps=50,
            eta=1.0
        )

        decoded = model.decode_first_stage(latent)

        range_img = decoded[0,0].cpu().numpy()

        pcd,_,_ = range2pcd(
            range_img,
            fov=cfg.data.params.dataset.fov,
            depth_range=cfg.data.params.dataset.depth_range,
            depth_scale=cfg.data.params.dataset.depth_scale,
            log_scale=cfg.data.params.dataset.log_scale
        )

        intensity = np.zeros((pcd.shape[0],1),dtype=np.float32)

        bin_points = np.concatenate(
            [pcd.astype(np.float32),intensity],
            axis=1
        )

        out_name = os.path.basename(image_path).replace(".png",".bin")

        bin_points.tofile(
            os.path.join(out_dir,out_name)
        )

        print(out_name,"saved")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",required=True)
    parser.add_argument("--ckpt",required=True)
    parser.add_argument("--images",required=True)
    parser.add_argument("--out",required=True)

    args=parser.parse_args()

    model,cfg = load_model(args.config,args.ckpt)

    infer(
        model,
        cfg,
        args.images,
        args.out
    )