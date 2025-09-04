from imaging_functions import *
from aicsimageio import AICSImage
import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from skimage.io import imread
from skimage.measure import regionprops
from skimage.color import label2rgb
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import gc

# === Parse arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--beg", type=int, required=True)
parser.add_argument("--end", type=int, required=True)
parser.add_argument("--start", type=int, required=True)
parser.add_argument("--stride", type=int, required=True)
args = parser.parse_args()

start = args.start
stride = args.stride

# === Set directory range here ===
start_idx = args.beg # start index (inclusive, 0-based)
end_idx   = args.end # exclusive → processes dirs [10, ..., 25]

# === Set main path ===
base_dir = '/mnt/storage_sda/Projects/imaging_data/'
output_subfolder_name = 'enhanced_rgb_frames'
cellpose_output_name = 'cellpose_outputs'

# === Get list of sample directories and slice range ===
all_dirs = sorted([
    os.path.join(base_dir, d) for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d))
])
selected_dirs = all_dirs[start_idx:end_idx]

print(f"Processing sample folders {start_idx} to {end_idx-1} out of {len(all_dirs)} total folders.")

for sample_path in tqdm(selected_dirs, desc="Processing sample folders"):
    # === Initialize Cellpose model ===
    model = models.CellposeModel(device=device)
    
    input_dir = os.path.join(sample_path, output_subfolder_name)
    output_dir = os.path.join(sample_path, cellpose_output_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get frame files
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))
    ])

    for fname in tqdm(image_files[start::stride], desc=f"Segmenting {os.path.basename(sample_path)}", leave=False):
        frame_path = os.path.join(input_dir, fname)
        base_name = os.path.splitext(fname)[0]

        # === Preprocess ===
        img_rgb = imread(frame_path).astype(np.float32) / 255.0
        red_channel = img_rgb[..., 0]
        red_eq = equalize_adapthist(red_channel, clip_limit=0.01)
        blended = 0.8 * red_eq + 4.0 * img_rgb.mean(axis=-1)
        channel_img = np.clip(blended, 0, 1)
        scaled_img = rescale(channel_img, scale=1.0, preserve_range=True, anti_aliasing=True)
        scaled_img = img_as_ubyte(channel_img)

        # === Cellpose inference ===
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

        # === Save outputs ===
        imsave(os.path.join(output_dir, f"{base_name}_mask.tif"), masks.astype(np.uint16))
        imsave(os.path.join(output_dir, f"{base_name}_flow_rgb.png"), flow_rgb)

    del model
    gc.collect()
    torch.cuda.empty_cache()
