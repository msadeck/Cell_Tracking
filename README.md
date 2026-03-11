# Cell_Tracking

This repository provides a full workflow for generating nuclei tracks from time-lapse microscopy data. It integrates image processing, segmentation (via Cellpose), feature extraction, filtering and trajectory (via Trackpy). 

# Full method pipeline: 
- Image loading and Enhancement
    scripts: imaging_functions.py and image_save.py
- Nuclei segmentation with Cellpose
    script: cellpose_gen.py
- Extraction of nuclei properties 
    notebook: trackpy_filtered2.ipynb
- Outlier filtering 
    notebook: trackpy_filtered2.ipynb
- Nuclei tracking with Trackpy 
    notebook: trackpy_filtered2.ipynb

# File Descriptions: 

imaging_functions.py: Defines functions that are used to process images in image_save.py 

image_save.py: 
This script processes a selected range of .czi microscopy image files. It extracts RGB channels for each timepoint, applies normalization and enhancement, and saves the result as enhanced .png images in a structured output directory. The file accepts two command line arguments (start idx and end idx). This allows parallel/chunked processing for efficiency. The script uses AICSImage from aicsimageio to read to CZI file and extracts three color channels (CFP/Blue, GFP/Green and RFP/Red). The image enhancement step normalizes the image to 8 bit, supresses green channel intensity in places where red dominates, removes background noise and stacks channels into a single RBG image. Each frame is saved as frame_XXX.png in /mnt/storage_sda/Projects/imaging_data/<base_filename>/enhanced_rgb_frames/. 

cellpose_gen.py: 
This script performs segmentation of microscopy time-lapse images (preprpsessed as enghanced RBG frames in image_save.py) using Cellpose. It take command line inputs of a range of sample directories and a start/stride to allow parellel processing and subsetting across GPUs. For each selected sample folder (/mnt/storage_sda/Projects/imaging_data/<base_filename>/enhanced_rgb_frames/) a new Cellpose model is initialized per sample folder to avoid memory build up. The images are preprocessed to enhances contrast and visibility of cells to improve Cellpose segmentation accuracy. The script saves each frame mask as .tif and each fram flow visualization as .png in directory /mnt/storage_sda/Projects/imaging_data/<base_filename>/cellpose_outputs. We use Cellpose Version 4.0.5, Python Version 3.10.13 and Torch Version 2.5.1.

trackpy_filtered2.ipynb: This notebook extracts cell centroids from Cellpose output masks and builds cell tracks using Trackpy. It outputs a csv file and a movie for unfiltered cell tracks as well as filtered cell tracks. The steps of this notebook, as well as the logic are detailed below:
    1. Cell masks/images are loaded and properties are extracted using regionprops(). A wide variety of properties (centroid, area, solidity, diameter, intensity, orientation, perimeter etc.) are stored so they can be used for outlier detection if necessary. These data are saved in data frame, full_df. 
    2. Frame particle counts are visualized throughout filtering process in order to understand noise. 
    3. Data frame filtering (pre Trackpy): The entire data frame is filtered using cKDTree. cDKTree allows for nearest-neighbor searches in k-dimensional space. By setting max_distance to 15, it ensures that each detection in a given fram has a nearby detection in the previous frame. This removes isolated noise and spurious detections that do not persist across time. Without this step, trackpy will throw a SubnetOversizeException. This step outputs a filtered dataframe, filtered_df. For some data, disconnected clusters of detections persist in a few frames and prevent Trackpy from running. In order to bypass this problem, the frames are filtered again by percentage change of particle number. If the quanity of particles in a frame changes by more than 25, that frame is removed. 
    4. Trackpy: Links are created using the filtered_df with a search range of 15 and a memory of 3. The tracks are saved in a .csv file, /mnt/storage_sda/Projects/imaging_data/<base_filename>/tracks/linked_tracks.csv. We use Trackpy version 0.6.4.
    5. Optional link filtering: this step removes tracks that are likely due to noise by filter by track duration. 
        - min_duration: minimum number of frames a particle must appear in to be considered a valid track. Tracks with very short durations are often due to false positives (e.g. debris or noise). We select a 10 frame minimum.
    6. Visualization: Images and movies of both the link filtered and link unfiltered tracks are produced and saved. 

# Directory Structure:
Projects/
└── imaging_data/
    └── <base_filename>/                        # e.g. mix_dilution_FibronectinPDK_24hr_5x_1x_JDF_9may2025-Scene-45-F10-F10
        ├── enhanced_rgb_frames/                # Output of image_save.py
        │   └── frame_000.png, frame_001.png, ...   
        ├── cellpose_outputs/                   # Output of cellpose_gen.py
        │   ├── frame_000_mask.tif
        │   ├── frame_000_flow_rgb.png
        │   └── ...
        └── tracks/                             # Output of trackpy_filtered2.ipynb
            ├── linked_tracks.csv               # All tracks before duration filtering
            ├── linked_tracks_filtered.csv      # Tracks filtered by min duration
            ├── track_overlay.mp4               # Movie of all tracks over original frames
            ├── track_overlay_filtered.mp4      # Movie of filtered tracks
            ├── track_overlay_unfiltered.mp4    # (optional) unfiltered overlay if both saved
            ├── track_frames_filtered/          # Per-frame images of filtered tracks
            │   └── track_000.png, ...
            ├── track_frames_unfiltered/        # Per-frame images of unfiltered tracks
            │   └── track_000.png, ...
            └── rgb_track_frames/               # RGB images used as overlay background
                └── frame_000.png, ...
