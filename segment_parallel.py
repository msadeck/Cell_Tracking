from imaging_functions import *
from aicsimageio import AICSImage
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
from skimage.measure import regionprops
from skimage.color import label2rgb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from skimage import exposure
from skimage.transform import rescale
from cellpose import plot
from skimage.measure import regionprops
from skimage.exposure import equalize_adapthist
from skimage import img_as_ubyte
device = torch.device("cuda:0")
import tifffile
from tqdm import tqdm
from skimage.io import imread, imsave

import argparse

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--stride", type=int, required=True)
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
start = args.start
stride = args.stride
scale_factor = 1.0

os.makedirs(output_dir, exist_ok=True)

# === Load image files ===
image_files = sorted([
    fname for fname in os.listdir(input_dir)
    if fname.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
])

# === Initialize Cellpose model ===
model = models.CellposeModel(device=device)

# === Process assigned frames ===
for idx in range(start, len(image_files), stride):
    fname = image_files[idx]
    image_path = os.path.join(input_dir, fname)
    base_name = os.path.splitext(fname)[0]

    img_rgb = imread(image_path).astype(np.float32) / 255.0
    red_channel = img_rgb[..., 0]
    red_eq = equalize_adapthist(red_channel, clip_limit=0.01)
    blended = 0.8 * red_eq + 4.0 * img_rgb.mean(axis=-1)
    channel_img = np.clip(blended, 0, 1)
    scaled_img = rescale(channel_img, scale=scale_factor, preserve_range=True, anti_aliasing=True)
    scaled_img = img_as_ubyte(scaled_img)

    masks_list, flows_list, _ = model.eval(
        [scaled_img],
        diameter=None,
        augment=True,
        tile_overlap=0.5,
        progress=False,
        cellprob_threshold=-4.0,
        flow_threshold=0.0,
    )

    masks = masks_list[0]
    flow_rgb = flows_list[0][0]
    #flow_yx  = flows_list[0][1]
    #cellprob = flows_list[0][2]

    imsave(os.path.join(output_dir, f"{base_name}_mask.tif"), masks.astype(np.uint16))
    #cellprob = cellprob[None, ...]
    #flow_stack = np.concatenate([flow_yx, cellprob], axis=0).astype(np.float32)
    #tifffile.imwrite(os.path.join(output_dir, f"{base_name}_flows.tif"), flow_stack)
    imsave(os.path.join(output_dir, f"{base_name}_flow_rgb.png"), flow_rgb)
