import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
import pyexr
import open3d as o3d
import time


def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # pose is world2camera


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        normal_dir = conf.get_string('normal_dir')

        self.data_dir = conf.get_string('data_dir')
        self.cameras_name = conf.get_string('cameras_name')
        self.exclude_view_list = conf['exclude_views']  # list of views to exclude from training. Used in novel-view normal synthesis evaluation.
        self.upsample_factor = conf.get_int('upsample_factor', default=1)
        ic(self.exclude_view_list)

        # load the GT mesh for evaluation if any
        mesh_path = os.path.join(self.data_dir, 'mesh_Gt.ply')
        if os.path.exists(mesh_path):
            self.mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
        else:
            self.mesh_gt = None
        self.points_gt = None  # will be computed from the mesh at evaluation time

        camera_dict = np.load(os.path.join(self.data_dir, self.cameras_name))
        self.camera_dict = camera_dict
        self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
        self.n_images = len(self.normal_lis)
        self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)
        self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in self.normal_lis]

        print("loading normal maps...")
        self.normal_np = np.stack([pyexr.read(im_name)[..., :3] for im_name in self.normal_lis])
        if self.upsample_factor > 1:
            # resize normal maps
            self.normal_np = F.interpolate(torch.from_numpy(self.normal_np).permute(0, 3, 1, 2), scale_factor=self.upsample_factor, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).numpy()
        self.normals = torch.from_numpy(self.normal_np.astype(np.float32)).to(self.device)  # [n_images, H, W, 3]
        print("loading normal maps done.")

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 255.0

        if self.upsample_factor > 1:
            # resize mask
            self.masks_np = F.interpolate(torch.from_numpy(self.masks_np).permute(0, 3, 1, 2), scale_factor=self.upsample_factor, mode='nearest').permute(0, 2, 3, 1).numpy()
        self.masks_np = self.masks_np[..., 0]
        self.total_pixel = np.sum(self.masks_np)

        # set background of normal map to 0
        self.normal_np[self.masks_np == 0] = 0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]

        self.intrinsics_all = []
        self.pose_all = []
        self.V_inverse_all = []

        self.H, self.W = self.normal_np.shape[1], self.normal_np.shape[2]
        for scale_mat, world_mat, normal_map, mask in zip(self.scale_mats_np, self.world_mats_np, self.normals, self.masks_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            if self.upsample_factor > 1:
                # resize intrinsics
                intrinsics[0, 0] *= self.upsample_factor
                intrinsics[1, 1] *= self.upsample_factor
                intrinsics[0, 2] *= self.upsample_factor
                intrinsics[1, 2] *= self.upsample_factor
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            intrinsics_inverse = torch.inverse(torch.from_numpy(intrinsics).float())
            pose = torch.from_numpy(pose).float()
            # compute the V_inverse
            tx = torch.linspace(0, self.W - 1, int(self.W))
            ty = torch.linspace(0, self.H - 1, int(self.H))
            pixels_x, pixels_y = torch.meshgrid(tx, ty)
            p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(intrinsics_inverse.device)  # W, H, 3
            p = torch.matmul(intrinsics_inverse[None, None, :3, :3],
                             p[:, :, :, None]).squeeze()  # W, H, 3
            rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
            rays_v = torch.matmul(pose[None, None, :3, :3],
                                  rays_v[:, :, :, None]).squeeze()  # W, H, 3
            rays_v = rays_v.transpose(0, 1).to(self.device) # H, W, 3

            # the axis direction of the camera coordinate system in the world coordinate system
            rays_right = pose[None, :3, 0].expand(rays_v.shape).to(self.device)  # H, W, 3
            rays_down = pose[None, :3, 1].expand(rays_v.shape).to(self.device)  # H, W, 3

            V_concat = torch.cat([rays_v[..., None, :],
                                  rays_right[..., None, :],
                                  rays_down[..., None, :]], dim=-2)  # (H, W, 3, 3)

            # computing the inverse may take a while if the resolution is high
            # For 512x612, it takes about 0.8ms
            V_inverse = torch.inverse(V_concat)  # (H, W, 3, 3)
            self.V_inverse_all.append(V_inverse)

        self.masks = torch.from_numpy(self.masks_np.astype(np.float32)).to(self.device) # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal_length = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.image_pixels = self.H * self.W
        self.V_inverse_all = torch.stack(self.V_inverse_all).to(self.device)  # [n_images, H, W, 3, 3]

        # for mesh extraction
        self.object_bbox_min = np.array([-1., -1., -1.])
        self.object_bbox_max = np.array([1.,  1.,  1.])
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, within_mask=False):
        """
        Generate all rays at world space from one camera.
        """
        mask_np = self.masks_np[img_idx].astype(bool)
        # resize the mask using resolution_level
        mask_np = cv.resize(mask_np.astype(np.uint8)*255, (int(self.W // resolution_level), int(self.H // resolution_level)), interpolation=cv.INTER_NEAREST).astype(bool)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l))
        ty = torch.linspace(0, self.H - 1, int(self.H // l))
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        rays_o = rays_o.transpose(0, 1)
        rays_v = rays_v.transpose(0, 1)

        if within_mask:
            return rays_o[mask_np], rays_v[mask_np]
        else:
            return rays_o, rays_v

    def gen_patches_at(self, img_idx, resolution_level=1, patch_H=3, patch_W=3):
        tx = torch.linspace(0, self.W - 1, int(self.W // resolution_level))
        ty = torch.linspace(0, self.H - 1, int(self.H // resolution_level))
        pixels_y, pixels_x = torch.meshgrid(ty, tx)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # H, W, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, :3, :3], p[..., None]).squeeze()  # H, W, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, :3, :3], rays_v[:, :, :, None]).squeeze()  # H, W, 3

        # split rays_v into non-overlapping patches
        height, width, _ = rays_v.shape
        horizontal_num_patch = width // patch_W
        vertical_num_patch = height // patch_H
        rays_v_patches_all = []
        rays_V_inverse_patches_all = []
        rays_ez_patches_all = []
        mask_value = []
        for i in range(0, height-patch_H//2-1, patch_H):
            for j in range(0, width-patch_W//2-1, patch_W):
                rays_v_patch = rays_v[i:i + patch_H, j:j + patch_W]
                rays_v_patches_all.append(rays_v_patch)

                rays_V_inverse_patch = self.V_inverse_all[img_idx][i:i + patch_H, j:j + patch_W]
                rays_V_inverse_patches_all.append(rays_V_inverse_patch)

                rays_ez_patch = self.normals[img_idx][i + patch_H//2, j + patch_W//2]
                rays_ez_patches_all.append(rays_ez_patch)

                mask_value.append(self.masks_np[img_idx][i + patch_H//2, j + patch_W//2].astype(bool))
        rays_v_patches_all = torch.stack(rays_v_patches_all, dim=0)  # (num_patch, patch_H, patch_W, 3)
        rays_V_inverse_patches_all = torch.stack(rays_V_inverse_patches_all, dim=0)  # (num_patch, patch_H, patch_W, 3, 3)
        rays_o_patches_all = self.pose_all[img_idx, :3, 3].expand(rays_v_patches_all.shape)  # (num_patch, patch_H, patch_W, 3)

        rays_o_patch_center = rays_o_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_v_patches_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        marching_plane_normal_patches_all = self.pose_all[img_idx, :3, 2].expand(rays_d_patch_center.shape)  # (num_patch, 3)

        return rays_o_patch_center, \
                rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            marching_plane_normal_patches_all, \
            rays_V_inverse_patches_all, horizontal_num_patch, vertical_num_patch

    def gen_random_patches(self, num_patch, patch_H=3, patch_W=3):
        """
        Generate random patches of rays at world space from all viewpoints.
        X-axis right, Y-axis down

        Parameters:
        num_patch (int): The number of patches to generate.
        patch_H (int, optional): The height of the patches. Default is 3.
        patch_W (int, optional): The width of the patches. Default is 3.

        Returns:
        rays_o_patch_all (torch.Tensor): The origins of the rays in each patch. A tensor of shape (num_patch, patch_H, patch_W, 3).
        rays_d_patch_all (torch.Tensor): The directions of the rays in each patch. A tensor of shape (num_patch, patch_H, patch_W, 3).
        marching_plane_normal (torch.Tensor): The normal direction of the image/marching plane.
                Since we randomly sample patches from all viewpoints, this normal is only identical for each patch. A tensor of shape (num_patch, 3).
        V_inverse_patch_all (torch.Tensor): The inverse of the V matrix at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 3, 3).
        normal (torch.Tensor): The normals at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 3).
        mask (torch.Tensor): The mask values at patches of pixels. A tensor of shape (num_patch, patch_H, patch_W, 1).
        """
        # randomly sample center pixel locations of patches
        # assume all images have the same resolution
        patch_center_x = torch.randint(low=0+patch_W//2, high=self.W-1-patch_W//2, size=[num_patch], device=self.device)  # (num_patch, )
        patch_center_y = torch.randint(low=0+patch_H//2, high=self.H-1-patch_H//2, size=[num_patch], device=self.device)  # (num_patch, )

        # compute all pixel locations within the patches given patch size (patch_H, patch_W)
        patch_center_x_all = patch_center_x[:, None, None] + torch.arange(-patch_W//2+1, patch_W//2+1, device=self.device).repeat(patch_H, 1)   # (num_patch, patch_H, patch_W)
        patch_center_y_all = patch_center_y[:, None, None] + torch.arange(-patch_H//2+1, patch_H//2+1, device=self.device).reshape(-1, 1).repeat(1, patch_W)   # (num_patch, patch_H, patch_W)

        # randomly sample viewpoints
        img_idx = np.random.choice(list(self.train_images), size=[num_patch])  # (num_patch, )
        img_idx = torch.tensor(img_idx, device=self.device)
        img_idx_expand = img_idx.view(-1, 1, 1).expand_as(patch_center_x_all)  # (num_patch, patch_H, patch_W)

        # input normals and mask values for supervision
        normal = self.normals[img_idx_expand, patch_center_y_all, patch_center_x_all]  # (num_patch, patch_H, patch_W, 3)
        V_inverse_patch_all = self.V_inverse_all[img_idx_expand, patch_center_y_all, patch_center_x_all]  # (num_patch, patch_H, patch_W, 3, 3)
        mask = self.masks[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1)#[..., :1]     # (num_patch, patch_H, patch_W)

        # compute all ray directions within patches
        p_all = torch.stack([patch_center_x_all, patch_center_y_all, torch.ones_like(patch_center_y_all)], dim=-1).float().to(self.device)  # (num_patch, patch_H, patch_W, 3)
        p_all = torch.matmul(self.intrinsics_all_inv[img_idx_expand, :3, :3], p_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        p_norm_all = torch.linalg.norm(p_all, ord=2, dim=-1, keepdim=True)  # (num_patch, patch_H, patch_W, 1)
        rays_d_patch_all = p_all / p_norm_all  # (num_patch, patch_H, patch_W, 3)
        rays_d_patch_all = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_d_patch_all[..., None])[..., 0]  # (num_patch, patch_H, patch_W, 3)
        rays_o_patch_all = self.pose_all[img_idx, None, None, :3, 3].expand(rays_d_patch_all.shape)  # (num_patch, patch_H, patch_W, 3)

        # the normal direction of the image/marching plane is the 3rd column of world2camera transformation
        marching_plane_normal = self.pose_all[img_idx, :3, 2].expand((num_patch, 3))  # (num_patch, 3)

        return rays_o_patch_all, \
                rays_d_patch_all, \
                marching_plane_normal, \
                V_inverse_patch_all, \
                normal,\
                mask

    def near_far_from_sphere(self, rays_o, rays_d):
        """
        This function calculates the near and far intersection points of rays with a unit sphere.

        Parameters:
        rays_o (torch.Tensor): Origin of the rays. A tensor of shape (N, 3) where N is the number of rays.
        rays_d (torch.Tensor): Direction of the rays. A tensor of shape (N, 3) where N is the number of rays.

        Returns:
        near (torch.Tensor): Near intersection points of the rays with the unit sphere. A tensor of shape (N, ).
        far (torch.Tensor): Far intersection points of the rays with the unit sphere. A tensor of shape (N, ).
        """
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        c = torch.sum(rays_o**2, dim=-1, keepdim=True) - 1.0
        mid = 0.5 * (-b) / a
        near = mid - torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        far = mid + torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        return near[..., 0], far[..., 0]

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

