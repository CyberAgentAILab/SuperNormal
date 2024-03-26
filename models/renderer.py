import torch
import numpy as np
import mcubes
from tqdm import tqdm
from nerfacc import ContractionType, OccupancyGrid, ray_marching, \
    render_weight_from_alpha_patch_based, accumulate_along_rays_patch_based, \
    render_weight_from_alpha, accumulate_along_rays

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in tqdm(enumerate(X)):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class NeuSRenderer:
    def __init__(self, sdf_network, deviation_network,
                 gradient_method="dfd"):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network

        # define the occ grid, see NerfAcc for more details
        self.scene_aabb = torch.as_tensor([-1., -1., -1., 1., 1., 1.], dtype=torch.float32)
        # define the contraction_type for scene contraction
        self.contraction_type = ContractionType.AABB
        # create Occupancy Grid
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=128,  # if res is different along different axis, use [256,128,64]
            contraction_type=self.contraction_type).to("cuda")
        self.sampling_step_size = 0.01  # ray marching step size, will be modified during training
        self.gradient_method = gradient_method   # dfd or fd or ad


    def occ_eval_fn(self, x):
        # function for updating the occ grid given the current sdf
        sdf = self.sdf_network(x)[..., :1]
        alpha = torch.sigmoid(- sdf * 80)  # occ grids with alpha below the occ threshold will be set as 0
        return alpha


    def render(self, rays_o_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     rays_d_patch_all,  # (num_patch, patch_H, patch_W, 3)
                     marching_plane_normal,  # (num_patch, 3)
                     near,  # (num_patch,)
                     far,  # (num_patch,)
                     V_inverse_patch_all,  # (num_patch, patch_H, patch_W, 3, 3)
                     val_gradient_method='dfd',
                     mode='train'):
        # patch size, should be odd
        patch_H = rays_o_patch_all.shape[1]
        patch_W = rays_o_patch_all.shape[2]
        num_patch = rays_o_patch_all.shape[0]

        # extract camera location and ray direction of the patches' center pixels
        rays_o_patch_center = rays_o_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)
        rays_d_patch_center = rays_d_patch_all[:, patch_H//2, patch_W//2]  # (num_patch, 3)

        def alpha_fn_patch_center(t_starts, t_ends, ray_indices, ret_sdf=False):
            # the function used in ray marching
            ray_indices = ray_indices.long()
            t_origins = rays_o_patch_center[ray_indices]
            t_dirs = rays_d_patch_center[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return
            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_start[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        with torch.no_grad():
            patch_indices, t_starts_patch_center, t_ends_patch_center = ray_marching(
                rays_o_patch_center, rays_d_patch_center,
                t_min=near,
                t_max=far,
                grid=self.occupancy_grid,
                render_step_size=self.sampling_step_size,
                stratified=True,
                cone_angle=0.0,
                early_stop_eps=1e-8,
                alpha_fn=alpha_fn_patch_center,
            )
            samples_per_ray = patch_indices.shape[0] / num_patch
            if patch_indices.shape[0] == 0:  # all patch center rays are within the zero region of the occ grid. skip this iteration.
                return {
                    "comp_normal": torch.zeros([num_patch, patch_H, patch_W, 3], device=rays_o_patch_center.device)
                }

            num_samples = patch_indices.shape[0]
            patch_indices = patch_indices.long()

            # compute the sampling distance on remaining rays
            t_starts_patch_all = t_starts_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                                 /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]
            t_ends_patch_all = t_ends_patch_center[:, None, None, :] * (rays_d_patch_center * marching_plane_normal).sum(-1, keepdim=True)[patch_indices][:, None, None, :] \
                               /(rays_d_patch_all * marching_plane_normal[:, None, None, :]).sum(-1, keepdim=True)[patch_indices]


            t_starts_patch_center_shift_left = t_starts_patch_center[1:]
            t_starts_patch_center_shift_left = torch.cat([t_starts_patch_center_shift_left, t_starts_patch_center[-1:]], 0)
            diff_mask = ((t_ends_patch_center - t_starts_patch_center_shift_left) != 0)[..., 0]
            positions_starts_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_starts_patch_all
            positions_ends_patch_all = rays_o_patch_all[patch_indices] + rays_d_patch_all[patch_indices] * t_ends_patch_all  # (num_samples, patch_H, patch_W, 3)
            positions_ends_diff = positions_ends_patch_all[diff_mask]
            positions_all = torch.cat([positions_starts_patch_all, positions_ends_diff], 0)
            positions_all_flat = positions_all.reshape(-1, 3)

        sdf_all = self.sdf_network(positions_all_flat)
        sdf_all = sdf_all.reshape(*positions_all.shape[:-1], 1)

        sdf_starts_patch_all = sdf_all[:positions_starts_patch_all.shape[0]]

        sdf_end_diff = sdf_all[positions_starts_patch_all.shape[0]:]
        sdf_ends_patch_all = sdf_starts_patch_all[1:]
        sdf_ends_patch_all = torch.cat([sdf_ends_patch_all, sdf_starts_patch_all[-1:]], 0)
        sdf_ends_patch_all[diff_mask] = sdf_end_diff

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter

        prev_cdf = torch.sigmoid(sdf_starts_patch_all * inv_s)  # (num_samples, patch_H, patch_W, 1)
        next_cdf = torch.sigmoid(sdf_ends_patch_all * inv_s)   # (num_samples, patch_H, patch_W, 1)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)  # (num_samples, patch_H, patch_W, 1)
        weights_cuda = render_weight_from_alpha_patch_based(alpha.reshape(num_samples, patch_H*patch_W, 1), patch_indices)  # (num_samples, patch_H, patch_W, 1)

        if mode == 'train':
            gradient_method = self.gradient_method
        elif mode == 'eval':
            gradient_method = val_gradient_method

        if gradient_method == "dfd":
            with torch.no_grad():
                # distance between neighboring points on the same marching plane
                dist_x = torch.norm(positions_starts_patch_all[:, :, 1:, :] -
                                    positions_starts_patch_all[:, :, :-1, :], dim=-1, keepdim=True)  # (num_samples, patch_H, patch_W-1, 1)
                dist_y = torch.norm(positions_starts_patch_all[:, 1:, :, :] -
                                    positions_starts_patch_all[:, :-1, :, :], dim=-1, keepdim=True)  # (num_samples, patch_H-1, patch_W, 1)

            # directional derivatives along the ray direction
            # forward difference
            df_dt = (sdf_ends_patch_all - sdf_starts_patch_all) / (t_ends_patch_all - t_starts_patch_all)  # (num_samples, patch_H, patch_W, 1)
            # directional derivatives along the image's x-direction
            # central difference
            df_dx = (sdf_starts_patch_all[:, :, 2:] - sdf_starts_patch_all[:, :, :-2]) / (dist_x[:, :, :-1] + dist_x[:, :, 1:] )  # (num_samples, patch_H, patch_W-2, 1)
            # directional derivatives along the image's y-direction
            # central difference
            df_dy = (sdf_starts_patch_all[:, 2:, :] - sdf_starts_patch_all[:, :-2, :]) / (dist_y[:, 1:, :] + dist_y[:, :-1, :])  # (num_samples, patch_H-2, patch_W, 1)

            # for points only have one-side neighbor point,
            # we use forward or backward difference correspondingly
            df_dx_left_boundary = (sdf_starts_patch_all[:, :, 1:2] - sdf_starts_patch_all[:, :, 0:1]) / dist_x[:, :, 0:1]  # (num_samples, patch_H, 1)
            df_dx_right_boundary = (sdf_starts_patch_all[:, :, -1:] - sdf_starts_patch_all[:, :, -2:-1]) / dist_x[:, :, -1:]  # (num_samples, patch_H, 1)
            df_dy_top_boundary = (sdf_starts_patch_all[:, 1:2, :] - sdf_starts_patch_all[:, 0:1, :]) / dist_y[:, 0:1, :]  # (num_samples, 1, patch_W)
            df_dy_bottom_boundary = (sdf_starts_patch_all[:, -1:, :] - sdf_starts_patch_all[:, -2:-1, :]) / dist_y[:, -1:, :]  # (num_samples, 1, patch_W)

            # concat the directional derivatives for boundary points and central points
            df_dx = torch.cat([df_dx_left_boundary, df_dx, df_dx_right_boundary], dim=2)  # (num_samples, patch_H, patch_W, 1)
            df_dy = torch.cat([df_dy_top_boundary, df_dy, df_dy_bottom_boundary], dim=1)  # (num_samples, patch_H, patch_W, 1)

            # concat the directional partial derivatives in three directions
            projected_gradients = torch.cat([df_dt,
                                             df_dx,
                                             df_dy], dim=-1)  # (num_patches, patch_H, patch_W, 3)

            # recover the gradients from directional partial derivatives using the inverse of known directions
            V_inverse = V_inverse_patch_all[patch_indices]  # (num_patches, patch_H, patch_W, 3, 3)
            gradients = (V_inverse @ projected_gradients[..., None])[..., 0]  # (num_samples, patch_H, patch_W, 3)

        elif gradient_method == "ad":
            gradients = self.sdf_network.gradient(positions_starts_patch_all.reshape(-1, 3)).reshape(num_samples, patch_H, patch_W, 3)

        elif gradient_method == "fd":
            # 6-point finite difference
            self.fd_epsilon = 1e-3
            positions_xn = positions_starts_patch_all + torch.tensor([[[[-self.fd_epsilon, 0, 0]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)
            positions_xp = positions_starts_patch_all + torch.tensor([[[[self.fd_epsilon, 0, 0]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)
            positions_yn = positions_starts_patch_all + torch.tensor([[[[0, -self.fd_epsilon, 0]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)
            positions_yp = positions_starts_patch_all + torch.tensor([[[[0, self.fd_epsilon, 0]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)
            positions_zn = positions_starts_patch_all + torch.tensor([[[[0, 0, -self.fd_epsilon]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)
            positions_zp = positions_starts_patch_all + torch.tensor([[[[0, 0, self.fd_epsilon]]]], device=positions_starts_patch_all.device).expand(
                positions_starts_patch_all.shape)

            positions_concat = torch.cat(
                [positions_xn, positions_xp, positions_yn, positions_yp, positions_zn, positions_zp], 0).to(
                torch.float32).reshape(-1, 3)

            sdf_concat = self.sdf_network(positions_concat).reshape(-1, patch_H, patch_W, 1)
            num_samples = positions_starts_patch_all.shape[0]
            sdf_xn = sdf_concat[:num_samples].reshape(num_samples, patch_H, patch_W, 1)
            sdf_xp = sdf_concat[num_samples:2 * num_samples].reshape(num_samples, patch_H, patch_W, 1)
            sdf_yn = sdf_concat[2 * num_samples:3 * num_samples].reshape(num_samples, patch_H, patch_W, 1)
            sdf_yp = sdf_concat[3 * num_samples:4 * num_samples].reshape(num_samples, patch_H, patch_W, 1)
            sdf_zn = sdf_concat[4 * num_samples:5 * num_samples].reshape(num_samples, patch_H, patch_W, 1)
            sdf_zp = sdf_concat[5 * num_samples:].reshape(num_samples, patch_H, patch_W, 1)

            df_dx = (sdf_xp - sdf_xn) / (2 * self.fd_epsilon)
            df_dy = (sdf_yp - sdf_yn) / (2 * self.fd_epsilon)
            df_dz = (sdf_zp - sdf_zn) / (2 * self.fd_epsilon)

            gradients = torch.stack([df_dx, df_dy, df_dz], -1)

        weights_sum_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, n_patches=num_patch)  # (num_samples, patch_H, patch_W, 1)
        weights_sum = weights_sum_cuda.reshape(num_patch, patch_H, patch_W, 1)

        comp_normals_cuda = accumulate_along_rays_patch_based(weights_cuda, patch_indices, values=gradients.reshape(num_samples,patch_H * patch_W, 3),n_patches=num_patch)  # (num_samples, patch_H, patch_W, 3)
        comp_normal = comp_normals_cuda.reshape(num_patch, patch_H, patch_W, 3)
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter

        return {
            's_val': 1/inv_s,
            'weight_sum': weights_sum,
            'gradients': gradients,
            "comp_normal": comp_normal,
            "samples_per_ray": samples_per_ray,
        }

    @torch.no_grad()
    def render_normal_pixel_based(self, rays_o, rays_d, near, far):
        def alpha_fn(t_starts, t_ends, ray_indices, ret_sdf=False):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = t_starts[1:]
            # attach the last element of t_ends to the end of t_starts_shift_left
            t_starts_shift_left = torch.cat([t_starts_shift_left, t_starts[-1:]], 0)

            # compute the diff mask between t_ends and t_starts_shift_left
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze()
            # if the diff maks is empty, return

            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            # ic(diff_mask.shape, positions_ends_diff.shape, positions_starts.shape)
            positions_all = torch.cat([positions_starts, positions_ends_diff], 0)

            sdf_all = self.sdf_network(positions_all)
            sdf_start = sdf_all[:positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0]:]

            sdf_start_shift_left = sdf_start[1:]
            sdf_start_shift_left = torch.cat([sdf_start_shift_left, sdf_start[-1:]], 0)

            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
            alpha = alpha.reshape(-1, 1)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            else:
                return alpha

        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d,
            t_min=near.squeeze(),
            t_max=far.squeeze(),
            grid=self.occupancy_grid,
            render_step_size=self.sampling_step_size,
            stratified=True,
            cone_angle=0.0,
            alpha_thre=0.0,
            early_stop_eps=1e-3,
            alpha_fn=alpha_fn,
        )

        alpha = alpha_fn(t_starts, t_ends, ray_indices)

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        gradients = self.sdf_network.gradient(positions).reshape(-1, 3)

        n_rays = rays_o.shape[0]
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)  # [n_samples, 1]
        comp_normal = accumulate_along_rays(weights, ray_indices, values=gradients, n_rays=n_rays)
        comp_depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        return comp_normal, comp_depth

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
