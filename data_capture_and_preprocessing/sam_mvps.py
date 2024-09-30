import os.path
from glob import glob
import argparse
import torch.cuda
from segment_anything import SamPredictor, sam_model_registry

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--data_dir", type=str, default="./")
args = parser.parse_args()

sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
sam.to(device="cuda")
predictor = SamPredictor(sam)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

obj_dir = os.listdir(args.data_dir)
obj_dir = [os.path.join(args.data_dir, obj) for obj in obj_dir if ".data" in obj]
mask_dir = os.path.join(os.path.dirname(os.path.dirname(args.data_dir)), "mask")
os.makedirs(mask_dir, exist_ok=True)

def pick_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'You selected point ({x}, {y})')
        points.append(np.array([[x, y]]))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



for obj_dir_path in obj_dir:
    mask_path = os.path.join(obj_dir_path, "mask.png")
    if os.path.exists(mask_path):
        continue
    # randomly pick an image from the object directory
    img_list = glob(os.path.join(obj_dir_path, "*.png")) + glob(os.path.join(obj_dir_path, "*.jpg"))
    img_test_path = img_list[0]
    img_test = cv2.imread(img_test_path)

    predictor.set_image(img_test)
    torch.cuda.synchronize()

    points = []

    while True:
        # Create a window
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        # Bind the callback function to the window
        cv2.setMouseCallback('image', pick_point)

        while(1):
            cv2.imshow('image', img_test)
            if cv2.waitKey(20) & 0xFF == 27:  # Break the loop when 'ESC' is pressed
                break

        cv2.destroyAllWindows()
        print(f'Selected points: {points}')

        input_point = np.concatenate(points, axis=0).reshape(-1, 2)
        input_label = np.ones(input_point.shape[0], dtype=np.int64)
        print(f'Input point: {input_point}')

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(img_test[:, :, ::-1])
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        value = input("Press enter to save the mask, or c to continue selecting points: ")
        if value == "c":
            continue
        elif value == "":
            break

    # save the mask
    base_dir = os.path.dirname(img_test_path)
    view_idx = int(base_dir.split("/")[-1].split(".")[0].split("_")[-1])
    mask_path1 = os.path.join(base_dir, "mask.png")
    mask_path2 = os.path.join(mask_dir, f"{view_idx:02d}.png")
    cv2.imwrite(mask_path1, mask.astype(np.uint8) * 255)
    cv2.imwrite(mask_path2, mask.astype(np.uint8) * 255)
    print(f"Mask saved at {mask_path1} and {mask_path2}")

