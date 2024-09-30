from glob import glob
import os
import numpy as np
import cv2
from bs4 import BeautifulSoup
from metashape2neus import normalize_camera, make4x4
import json
import argparse

def create_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="./flower_girl")
arg = parser.parse_args()

data_dir = os.path.join(arg.data_dir, "sfm_png_full")
mask_dir = os.path.join(arg.data_dir, "mask")
xml_path = os.path.join(arg.data_dir, "cameras.xml")
obj_name = os.path.basename(arg.data_dir)

target_dir = os.path.join(arg.data_dir, "neus2_input", "images")
os.makedirs(target_dir, exist_ok=True)

# load images and masks and save them as rgba images
img_list = glob(os.path.join(data_dir, "*.png"))
img_list.sort()
num_view = len(img_list)
print(num_view)
img_h, img_w = cv2.imread(img_list[0]).shape[:2]

for i in range(num_view):
    img_path = img_list[i]
    mask_path = os.path.join(mask_dir, f"{i:02d}.png")
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[..., 3] = mask[..., 0]
    new_img_path = os.path.join(target_dir, f"{i:02d}.png")
    cv2.imwrite(new_img_path, img)
    print(f"Saved {new_img_path}")

data = {
    "from_na": True,
    "w": img_w,
    "h": img_h,
    "aabb_scale": 1.0,
    "frames": [],
    "scale": 1,
    "offset": [1, 1, 1],
}

with open(xml_path, "r") as f:
    xml_data = f.read()
bs_data = BeautifulSoup(xml_data, "xml")
c_unique = bs_data.find_all('resolution')
img_width = int(c_unique[0].get("width"))
img_height = int(c_unique[0].get("height"))
c_intrinsics = bs_data.find_all('calibration')
f = float(c_intrinsics[0].find("f").text)
cx_offset = float(c_intrinsics[0].find("cx").text)
cy_offset = float(c_intrinsics[0].find("cy").text)
K = np.array([[f, 0, (img_width - 1) / 2 + cx_offset],
              [0, f, (img_height - 1) / 2 + cy_offset],
              [0, 0, 1]])

b_unique = bs_data.find_all('camera')
R_list = []
t_list = []
C2W_list = []
camera_sphere = dict()
for tag in b_unique:
    img_name = tag.get("label")
    view_idx = int(img_name.split("_")[-1])
    # camera to world transform
    C2W = np.array([float(i) for i in tag.find("transform").text.split(" ")]).reshape((4, 4))
    C2W_list.append(C2W)

    print(img_name, view_idx)
    W2C = np.linalg.inv(C2W)
    R_list.append(W2C[:3, :3])
    t_list.append(W2C[:3, 3])

    camera_sphere[f"world_mat_{view_idx}"] = make4x4(K) @ W2C
    print(img_name)
    data["frames"].append({
        "file_path": f"images/{img_name}.png",
        "transform_matrix": C2W.tolist(),
        "intrinsic_matrix": make4x4(K).tolist()
    })

offset, scale = normalize_camera(R_list, t_list, camera2object_ratio=10)
data["scale"] = scale
data["offset"] = list((-offset*scale + 0.5))


create_json_file(data, os.path.join(arg.data_dir, "neus2_input", 'transform.json'))