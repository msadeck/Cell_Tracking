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


# === Set file range ===
start_idx = int(sys.argv[1])  # start index (inclusive, 0-based)
end_idx   = int(sys.argv[2])  # end index (exclusive): will process files [1, 2, ..., 20]

# === Paths ===
current_directory = '/home/mars/Data/snyder_colab/'
data_subdirectory = 'data_062725/8may/'
input_dir = os.path.join(current_directory, data_subdirectory)
output_base = '/home/mars/mnt/storage_sda/Projects/imaging_data/'

# === Get and slice file list ===
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.czi')])
czi_files = all_files[start_idx:end_idx]  # <--- slice range

print(f"Processing files {start_idx} to {end_idx-1} out of {len(all_files)} total .czi files")

# === Process each CZI file ===
for filename in tqdm(czi_files, desc="Processing selected CZI files"):
    czi_path = os.path.join(input_dir, filename)

    base_name = os.path.splitext(filename)[0]
    output_dir = os.path.join(output_base, base_name, 'enhanced_rgb_frames')
    os.makedirs(output_dir, exist_ok=True)

    img = AICSImage(czi_path)
    num_timepoints = img.dims.T

    for t in range(num_timepoints):
        try:
            blue_raw  = img.get_image_data("YX", T=t, C=0, Z=0)
            green_raw = img.get_image_data("YX", T=t, C=1, Z=0)
            red_raw   = img.get_image_data("YX", T=t, C=2, Z=0)

            blue  = normalize_to_8bit(blue_raw, gamma=1.2, boost=1.5)
            green = normalize_to_8bit(green_raw, gamma=1.0, boost=1.0)
            red   = normalize_to_8bit(red_raw, gamma=0.8, boost=2.5)

            overlap_mask = red > 50
            green[overlap_mask] = green[overlap_mask] * 0.4

            composite_intensity = (red.astype(float) + green.astype(float) + blue.astype(float)) / 3
            background_mask = composite_intensity < 15
            red[background_mask] = green[background_mask] = blue[background_mask] = 0

            rgb = np.stack([red, green, blue], axis=-1)
            Image.fromarray(rgb).save(f"{output_dir}/frame_{t:03d}.png")

        except Exception as e:
            print(f"⚠️ Skipped frame {t} in {filename} due to error: {e}")