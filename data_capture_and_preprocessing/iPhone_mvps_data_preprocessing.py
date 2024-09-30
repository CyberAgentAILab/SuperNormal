import rawpy, os
from glob import glob
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./flower_girl")
parser.add_argument("--num_img_per_view", type=int, default=13)
arg = parser.parse_args()

dng_list = glob(os.path.join(arg.data_dir, "RAW", "*.DNG"))
dng_list.sort()
num_image_per_view = arg.num_img_per_view
num_view = len(dng_list) // num_image_per_view

resize_factor = 1  # resize the png image to 1/2, 1/4, or 1

if resize_factor == 1/2:
    sfm_data_dir = os.path.join(arg.data_dir, "sfm_png_half")
    mvps_data_dir = os.path.join(arg.data_dir, "mvps_png_half")
elif resize_factor == 1/4:
    sfm_data_dir = os.path.join(arg.data_dir, "sfm_png_quarter")
    mvps_data_dir = os.path.join(arg.data_dir, "mvps_png_quarter")
elif resize_factor == 1:
    mvps_data_dir = os.path.join(arg.data_dir, "mvps_png_full")
    sfm_data_dir = os.path.join(arg.data_dir, "sfm_png_full")

os.makedirs(sfm_data_dir, exist_ok=True)
os.makedirs(mvps_data_dir, exist_ok=True)

for view_idx in tqdm(range(num_view)):
    view_dir = os.path.join(mvps_data_dir, f"view_{view_idx:02d}.data")
    if os.path.exists(view_dir):
        continue
    os.makedirs(view_dir, exist_ok=True)
    view_dng_list = dng_list[view_idx * num_image_per_view: (view_idx + 1) * num_image_per_view]

    for dng_idx, dng_path in enumerate(view_dng_list):
        with rawpy.imread(dng_path) as raw:
            rgb = raw.postprocess(no_auto_bright=True, output_bps=16)[..., ::-1].astype(np.float32)
            rgb = rgb.astype(np.uint16)
        rgb_resized = cv2.resize(rgb, (0, 0), fx=resize_factor, fy=resize_factor)

        # choose the first image in each view for SfM
        if dng_idx == 0:
            cv2.imwrite(os.path.join(sfm_data_dir, f"{view_idx:02d}.png"), rgb_resized)

        cv2.imwrite(os.path.join(view_dir, f"L{dng_idx:02d}.png"), rgb_resized)


