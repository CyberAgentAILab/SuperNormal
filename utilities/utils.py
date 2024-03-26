import numpy as np
import cv2
from PIL import Image, ImageChops
import os
import time
import torch
from PIL import Image, ImageDraw, ImageFont

exp_time = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def crop_a_set_of_images(*image_path):
    from PIL import ImageChops, Image
    imgs = []
    bboxes = []
    for im_path in image_path:
        im = Image.open(im_path)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -5)
        bbox = diff.getbbox()

        imgs.append(im)
        bboxes.append(bbox)
    bbox_aggre = np.asarray(bboxes)
    bbox_min = np.min(bbox_aggre, 0)
    bbox_max = np.max(bbox_aggre, 0)
    bbox_common = (bbox_min[0], bbox_min[1], bbox_max[2], bbox_max[3])
    for idx, img in enumerate(imgs):
        img = img.crop(bbox_common)
        img.save(image_path[idx])
    pass


def crop_image_based_on_ref_image(ref_img_path, *img_path):
    from PIL import ImageChops, Image
    ref_im = Image.open(ref_img_path)
    bg = Image.new(ref_im.mode, ref_im.size, ref_im.getpixel((0, 0)))
    diff = ImageChops.difference(ref_im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -5)
    bbox = diff.getbbox()

    for idx, im_path in enumerate(img_path):
        img = Image.open(im_path)
        img = img.crop(bbox)
        img.save(im_path)


def angular_error_map(N1, N2):
    dot = np.sum(np.multiply(N1, N2), axis=-1)
    dot = np.clip(dot, -1., 1.)
    return np.rad2deg(np.arccos(dot))


def crop_mask(mask):
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8) * 255
    im = Image.fromarray(mask)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    return bbox


def crop_image_by_mask(img, mask):
    bbox = crop_mask(mask)
    try:
        crop_img = img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    except:
        crop_img = img.copy()
    return crop_img


def save_video(vpath, images, fps):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(vpath, fourcc, fps, (width, height))
    for image in images:
        video.write(image)
    cv2.destroyAllWindows()
    video.release()


def toRGBA(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:, :, 3] = (mask.astype(bool)*255).astype(np.uint8)
    return img
