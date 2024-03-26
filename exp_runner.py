import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm.auto import tqdm
from pyhocon import ConfigFactory
from models.fields import SDFNetwork, SingleVarianceNetwork

import pyexr
import time
from utilities.utils import crop_image_by_mask, toRGBA

import open3d as o3d
import pyvista as pv
pv.set_plot_theme("document")
pv.global_theme.transparent_background = True
from models.cd_and_fscore import chamfer_distance_and_f1_score
import csv
from collections import OrderedDict


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

class Runner:
    def __init__(self, conf_text, mode='train', is_continue=False, datadir=None):
        self.device = torch.device('cuda')
        self.conf_text = conf_text

        if not is_continue:
            exp_time = str(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
            exp_time_dir = f"exp_{exp_time}"

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = os.path.join(self.conf['general.base_exp_dir'], exp_time_dir)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = get_class(self.conf['general.dataset_class'])(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.patch_size = self.conf.get_int('train.patch_size', default=3)

        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

        self.loss_type = self.conf.get('train.loss_type', 'l1')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        self.eikonal_weight = self.conf.get_float('train.eikonal_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')

        self.increase_bindwidth_every = self.conf.get_int('train.increase_bindwidth_every', default=350)

        # validation parameters
        self.val_normal_freq = self.conf.get_int('val.val_normal_freq')
        self.val_normal_resolution_level = self.conf.get_int('val.val_normal_resolution_level')
        self.val_gradient_method = self.conf.get('val.gradient_method', 'dfd')

        self.val_mesh_freq = self.conf.get_int('val.val_mesh_freq')
        self.val_mesh_res = self.conf.get_int('val.val_mesh_res')

        self.eval_metric_freq = self.conf.get_int('val.eval_metric_freq')
        self.report_freq = self.conf.get_int('val.report_freq')
        self.save_freq = self.conf.get_int('val.save_freq')

        # Ray marching parameters
        self.start_step_size = self.conf.get_float('model.ray_marching.start_step_size', default=1e-2)
        self.end_step_size = self.conf.get_float('model.ray_marching.end_step_size', default=5e-4)
        self.slop_step = (np.log10(self.start_step_size) - np.log10(self.end_step_size)) / self.end_iter

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'], encoding_config=self.conf['model.encoding']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)

        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())

        self.renderer = get_class(self.conf['general.renderer_class'])(self.sdf_network,
                                                                       self.deviation_network,
                                                                       self.conf["train"]["gradient_method"])

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.is_continue = is_continue
        self.mode = mode

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        print("Start training...")
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.writer.add_graph(self.sdf_network, verbose=False, input_to_model=torch.randn(1, 3))
        self.update_learning_rate()

        # create a csv file to save the evaluation metrics
        csv_file_name = f"eval_metrics.csv"
        csv_file_path = os.path.join(self.base_exp_dir, csv_file_name)
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w') as f:
                writer = csv.writer(f)
                if len(self.dataset.exclude_view_list)>0:
                    writer.writerow(['iter',
                                     'mae_all_view',
                                     'mae_test_view',
                                     'CD',
                                     'fscore'])
                else:
                    writer.writerow(['iter',
                                     'mae_all_view',
                                     'CD',
                                     'fscore'])

        res_step = self.end_iter - self.iter_step
        pbar = tqdm(range(res_step))
        for iter_i in pbar:
            # update ray marching step size
            self.renderer.sampling_step_size = 10 ** (np.log10(self.start_step_size) - self.slop_step*iter_i)

            # update occupancy grid
            self.renderer.occupancy_grid.every_n_step(step=iter_i,
                                                      occ_eval_fn=self.renderer.occ_eval_fn,
                                                      occ_thre=self.conf["model.ray_marching"]["occ_threshold"],
                                                      n=self.conf["model.ray_marching"]["occ_update_freq"])

            # following neuralangelo, gradually increase ingp bandwidth
            if self.iter_step % self.increase_bindwidth_every == 0:
                self.renderer.sdf_network.increase_bandwidth()

            # sample patches of pixels for training
            rays_o_patch_all, rays_d_patch_all, marching_plane_normal, V_inverse_patch_all, true_normal, mask = \
                self.dataset.gen_random_patches(self.batch_size, patch_H=self.patch_size, patch_W=self.patch_size)

            rays_o_patch_center = rays_o_patch_all[:, self.patch_size // 2, self.patch_size // 2]  # (num_patch, 3)
            rays_d_patch_center = rays_d_patch_all[:, self.patch_size // 2, self.patch_size// 2]  # (num_patch, 3)
            near, far = self.dataset.near_far_from_sphere(rays_o_patch_center, rays_d_patch_center)

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            mask_sum = mask.sum() + 1e-5

            # forward rendering
            render_out = self.renderer.render(rays_o_patch_all,
                                              rays_d_patch_all,
                                              marching_plane_normal,
                                              near, far, V_inverse_patch_all)

            if render_out['gradients'] is None:  # all rays are in the zero region of the occupancy grid
                self.update_learning_rate()
                continue

            comp_normal = render_out['comp_normal']  # rendered normal at pixels
            gradients = render_out['gradients']  # gradients at all sampled 3D points
            comp_mask = render_out['weight_sum']  # rendered occupancy at pixels
            samples_per_ray = render_out['samples_per_ray']

            normal_error = (comp_normal - true_normal) * mask
            if self.loss_type == 'l1':
                normal_loss = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
            elif self.loss_type == 'l2':
                normal_loss = F.mse_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum

            gradients_norm = torch.linalg.norm(gradients, ord=2, dim=-1)
            eikonal_loss = F.mse_loss(gradients_norm, torch.ones_like(gradients_norm), reduction='mean')
            mask_loss = F.binary_cross_entropy(comp_mask.clip(1e-5, 1.0 - 1e-5), mask)

            loss = self.normal_weight * normal_loss + \
                   self.mask_weight * mask_loss + \
                   self.eikonal_weight * eikonal_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            self.update_learning_rate()

            if self.iter_step % self.report_freq == 0:
                message_postfix = OrderedDict(loss=f"{loss:.3e}",
                                              s=f"{self.deviation_network.variance.item():.3e}",
                                              rm_step=f"{self.renderer.sampling_step_size.item():.3e}",
                                              samples_per_ray=f"{samples_per_ray:.1f}")
                pbar.set_postfix(ordered_dict=message_postfix)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(resolution=self.val_mesh_res)

            if self.iter_step % self.val_normal_freq == 0:
                for val_idx in range(self.dataset.n_images):
                    self.validate_normal_patch_based(idx=val_idx, resolution_level=self.val_normal_resolution_level,
                                                     gradient_method=self.val_gradient_method)

            if self.iter_step % self.eval_metric_freq == 0:
                # no gt mesh, skip the evaluation
                if self.dataset.mesh_gt is None:
                    continue

                # remove invisible faces in the gt mesh
                if self.dataset.mesh_gt is not None and self.dataset.points_gt is None:
                    self.dataset.mesh_gt.vertices = o3d.utility.Vector3dVector(
                        (np.asarray(self.dataset.mesh_gt.vertices) -
                         self.dataset.scale_mats_np[0][:3, 3][None]) /
                        self.dataset.scale_mats_np[0][0, 0])
                    mesh = trimesh.Trimesh(np.asarray(self.dataset.mesh_gt.vertices),
                                           np.asarray(self.dataset.mesh_gt.triangles), process=False)
                    self.dataset.points_gt = self.find_visible_points(mesh) * self.dataset.scale_mats_np[0][0, 0] + \
                                             self.dataset.scale_mats_np[0][:3, 3][None]

                cd, fscore = self.eval_geo(resolution=512)
                print(f'iter: {self.iter_step} cd: {cd:.3e}, fscore: {fscore:.3e}')
                if len(self.dataset.exclude_view_list)>0:
                    mae_allview, mae_test_view = self.eval_mae(gradient_method=self.val_gradient_method)

                    print('MAE (all views) {0}: {1:.5f}'.format(self.val_gradient_method, mae_allview))
                    print('MAE (test views) {0}: {1:.5f}'.format(self.val_gradient_method, mae_test_view))

                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.iter_step,
                                         mae_allview,
                                         mae_test_view,
                                         cd, fscore])

                else:
                    mae_allview = self.eval_mae(gradient_method="dfd")
                    # write to csv file
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.iter_step,
                                         mae_allview,
                                         cd, fscore])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        try:
            copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        except:
            # save conf_text into a txt file
            with open(os.path.join(self.base_exp_dir, 'recording', 'config.conf'), 'w') as f:
                f.write(self.conf_text)

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_normal_pixel_based(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level, within_mask=False)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(8192)
        rays_d = rays_d.reshape(-1, 3).split(8192)

        out_normal_fine = []
        out_depth_fine = []

        mask_np = self.dataset.masks_np[idx].astype(bool)[..., 0]
        mask_np = cv.resize(mask_np.astype(np.uint8),
                            ((int(W), int(H))),
                            interpolation=cv.INTER_NEAREST).astype(bool)

        for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            # background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            batch_normal, batch_depth = self.renderer.render_normal_pixel_based(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far)

            out_normal_fine.append(batch_normal.detach().cpu().numpy())
            out_depth_fine.append(batch_depth.detach().cpu().numpy())

        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())  # W2C rotation
            # normal_img_world = (normal_img.reshape([H, W, 3]) * 128 + 128).clip(0, 255)
            normal_img = np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1])
            normal_img[:,:, [1, 2]] *= -1
            normal_img_norm = np.linalg.norm(np.squeeze(normal_img), axis=2, keepdims=True)
            normal_img_normalized = np.squeeze(normal_img) / (normal_img_norm+1e-7)

            # normal_img = ((np.squeeze(normal_img)/normal_img_norm) * 128 + 128).clip(0, 255)
            normal_img = (np.squeeze(normal_img) * 128 + 128).clip(0, 255)
            normal_img_normalized = (np.squeeze(normal_img_normalized) * 128 + 128).clip(0, 255)


            depth_img = np.concatenate(out_depth_fine, axis=0).reshape([H, W])

        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, "depth"), exist_ok=True)

        normal_img_norm[~mask_np] = np.nan
        depth_img[~mask_np] = np.nan

        normal_img_norm = np.squeeze(normal_img_norm.clip(0.8, 1.2))
        normal_img_norm = (normal_img_norm - np.nanmin(normal_img_norm)) / (np.nanmax(normal_img_norm) - np.nanmin(normal_img_norm))
        normal_img_norm = np.nan_to_num(normal_img_norm)
        normal_img_norm = (normal_img_norm * 255).astype(np.uint8)
        normal_img_norm = cv.applyColorMap(normal_img_norm, cv.COLORMAP_JET)
        normal_img_norm[~mask_np] = 0
        cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}_norm.png'.format(self.iter_step, 0, idx)),
                           normal_img_norm[..., ::-1])

        cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, 0, idx)),
                           normal_img[..., ::-1])
        cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}_normalized.png'.format(self.iter_step, 0, idx)),
                            normal_img_normalized[..., ::-1])
        np.save(os.path.join(self.base_exp_dir,
                                'depth',
                                '{:0>8d}_{}_{}.npy'.format(self.iter_step, 0, idx)),
                    depth_img)
        return idx, (normal_img - 128) / 128.

    def validate_normal_patch_based(self, idx=-1, resolution_level=-1, gradient_method="dfd"):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Rendering normal maps...  iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o_patch_center, \
            rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            rays_ez, \
            rays_A_inverse, horizontal_num_patch, vertical_num_patch = self.dataset.gen_patches_at(idx, resolution_level=resolution_level,
                                                                                                   patch_H=self.patch_size,
                                                                                                   patch_W=self.patch_size)
        mask_np = self.dataset.masks_np[idx].astype(bool)  # (H, W)

        img_w = horizontal_num_patch * self.patch_size
        img_h = vertical_num_patch * self.patch_size
        # resize mask to the size of the image
        mask_np = cv.resize(mask_np.astype(np.uint8),
                            ((int(img_w), int(img_h))),
                            interpolation=cv.INTER_NEAREST).astype(bool)

        num_patches = rays_o_patches_all.shape[0]
        eval_patch_size = 1024
        comp_normal_map = np.zeros([img_h, img_w, 3])
        comp_normal_list = []

        for patch_idx in range(0, num_patches, eval_patch_size):
            rays_o_patch_center_batch = rays_o_patch_center[patch_idx:patch_idx+eval_patch_size]
            rays_d_patch_center_batch = rays_d_patch_center[patch_idx:patch_idx+eval_patch_size]
            rays_o_patches_all_batch = rays_o_patches_all[patch_idx:patch_idx+eval_patch_size]
            rays_v_patches_all_batch = rays_v_patches_all[patch_idx:patch_idx+eval_patch_size]
            rays_ez_batch = rays_ez[patch_idx:patch_idx+eval_patch_size]
            rays_A_inverse_batch = rays_A_inverse[patch_idx:patch_idx+eval_patch_size]

            near, far = self.dataset.near_far_from_sphere(rays_o_patch_center_batch,
                                                          rays_d_patch_center_batch)
            render_out = self.renderer.render(rays_o_patches_all_batch,
                                                    rays_v_patches_all_batch,
                                                    rays_ez_batch,
                                                    near, far,
                                                    rays_A_inverse_batch, gradient_method, mode='eval')

            comp_normal = render_out['comp_normal']
            comp_normal = comp_normal.detach().cpu().numpy()
            comp_normal_list.append(comp_normal)

        comp_normal_list = np.concatenate(comp_normal_list, axis=0)

        count = 0
        for i in range(0, img_h, self.patch_size):
            for j in range(0, img_w, self.patch_size):
                comp_normal_map[i:i+self.patch_size, j:j+self.patch_size] = comp_normal_list[count]
                count += 1
        normal_img_world = comp_normal_map

        rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())  # W2C rotation

        normal_img = np.matmul(rot, normal_img_world[..., None]).squeeze()
        normal_img[..., [1, 2]] *= -1
        normal_img_png = (np.squeeze(normal_img) * 128 + 128).clip(0, 255)
        normal_img_norm = np.linalg.norm(np.squeeze(normal_img), axis=2, keepdims=True)
        normal_dir = os.path.join(self.base_exp_dir, f'normals_validation_{gradient_method}', 'iter_{:0>6d}'.format(self.iter_step))
        os.makedirs(normal_dir, exist_ok=True)

        normal_img_normalized = np.squeeze(normal_img) / (normal_img_norm + 1e-7)
        normal_img_normalized = (np.squeeze(normal_img_normalized) * 128 + 128).clip(0, 255)

        normal_eval = np.zeros((img_h, img_w, 3))
        normal_eval[:normal_img_png.shape[0], :normal_img_png.shape[1]] = normal_img_png

        normal_eval_normalized = np.zeros((img_h, img_w, 3))
        normal_eval_normalized[:normal_img_normalized.shape[0], :normal_img_normalized.shape[1]] = normal_img_normalized

        normal_img_normalized = crop_image_by_mask(toRGBA(normal_eval_normalized.astype(np.uint8)[...,::-1], mask_np), mask_np)

        cv.imwrite(os.path.join(normal_dir, '{:0>8d}_{}_{}_rendered.png'.format(self.iter_step, 0, idx)),
                           normal_eval[..., ::-1])

        cv.imwrite(os.path.join(normal_dir, '{:0>8d}_{}_{}_normalized.png'.format(self.iter_step, 0, idx)),
                            normal_img_normalized)
        return normal_img_world, normal_dir

    def validate_mesh(self, world_space=True, resolution=256, threshold=0.0):
        print('Extracting mesh...  iter: {}'.format(self.iter_step))
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)

        mesh = trimesh.Trimesh(vertices, triangles)
        vertices, triangles = mesh.vertices, mesh.faces

        save_dir = os.path.join(self.base_exp_dir, 'meshes_validation')
        os.makedirs(save_dir, exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        self.writer.add_mesh('mesh_eval', vertices=vertices[None,...], faces=triangles[None,...], global_step=self.iter_step)

        mesh = self.remove_isolated_clusters(trimesh.Trimesh(vertices, triangles))
        mesh_path = os.path.join(save_dir, 'iter_{:0>8d}.ply'.format(self.iter_step))
        o3d.io.write_triangle_mesh((mesh_path), mesh)

        print(f'Mesh saved at {mesh_path}')

    def remove_isolated_clusters(self, mesh):
        # cleaning the marching cube extracted mesh
        import copy
        mesh = mesh.as_open3d
        # with o3d.utility.VerbosityContextManager(
        #         o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        mesh_eval = copy.deepcopy(mesh)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_eval.remove_triangles_by_mask(triangles_to_remove)
        mesh_eval.remove_unreferenced_vertices()
        return mesh_eval

    @torch.no_grad()
    def eval_mae(self, gradient_method):
        print("Computing mean angular errors...")
        normal_gt_dir = os.path.join(self.dataset.data_dir, "normal_world_space_GT")

        ae_map_list = []
        normal_map_eval_list = []
        ae_map_eval_list = []
        ae_map_test_list = []
        for idx in range(self.dataset.n_images):
            normal_gt = pyexr.read(os.path.join(normal_gt_dir, "{:02d}.exr".format(idx)))[..., :3]

            mask_np = self.dataset.masks_np[idx].astype(bool)

            normal_map_world, save_dir = self.validate_normal_patch_based(idx, resolution_level=self.val_normal_resolution_level, gradient_method=gradient_method)

            normal_map_world = normal_map_world / (1e-10 + np.linalg.norm(normal_map_world, axis=-1, keepdims=True))

            normal_eval = np.zeros((self.dataset.H, self.dataset.W, 3))
            normal_eval[:normal_map_world.shape[0], :normal_map_world.shape[1]] = normal_map_world
            normal_eval[~mask_np] = np.nan
            normal_map_eval_list.append(normal_eval)
            # self.writer.add_image(step=self.iter_step, data=(normal_eval + 1) / 2, name=("normal_eval_{:02d}".format(idx)))
            # pyexr.write(os.path.join(normal_save_dir, "{:02d}.exr".format(idx)), normal_img)

            angular_error_map = np.rad2deg(np.arccos(np.clip(np.sum(normal_gt * normal_eval, axis=-1), -1, 1)))
            # save angular error map

            ae_map_list.append(angular_error_map.copy())
            if idx in self.dataset.exclude_view_list:
                ae_map_test_list.append(angular_error_map.copy())

            # apply jet to angular error map
            angular_error_map[~mask_np] = 0
            angular_error_map_jet = cv.applyColorMap((angular_error_map / 20 * 255).clip(0, 255).astype(np.uint8),
                                                     cv.COLORMAP_JET)
            angular_error_map_jet[~mask_np] = 255
            angular_error_map_jet = crop_image_by_mask(toRGBA(angular_error_map_jet, mask_np), mask_np)
            cv.imwrite(os.path.join(save_dir, '{:0>8d}_{}_{}_ae_up_{}.png'.format(self.iter_step, 0, idx, 20)), angular_error_map_jet)


            ae_map_eval_list.append(angular_error_map_jet)

        mae = np.nanmean(np.stack(ae_map_list, axis=0))
        self.writer.add_scalar('Statistics/mae_allview', mae, self.iter_step)

        if len(ae_map_test_list) > 0:
            mae_test = np.nanmean(np.stack(ae_map_test_list, axis=0))
            self.writer.add_scalar('Statistics/mae_testview', mae_test, self.iter_step)
            return mae, mae_test

        return mae

    @torch.no_grad()
    def eval_geo(self, resolution=1024):
        # save the mesh
        save_dir = os.path.join(self.base_exp_dir, 'points_val')
        os.makedirs(save_dir, exist_ok=True)

        # save gt points
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(self.dataset.points_gt)
        if not os.path.exists(os.path.join(save_dir, f"pcd_gt.ply")):
            o3d.io.write_point_cloud(os.path.join(save_dir, f"pcd_gt.ply"), pcd_gt)

        # marching cubes
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=0)

        # vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh = trimesh.Trimesh(np.asarray(vertices), np.asarray(triangles), process=False)
        vertices_world = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh_world = trimesh.Trimesh(np.asarray(vertices_world), np.asarray(triangles), process=False)
        mesh_world_path = os.path.join(save_dir, f"{self.iter_step}_world.obj")
        mesh_world.export(mesh_world_path)

        points_eval = self.find_visible_points(mesh)*self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        # save the sampled points
        sampled_points_path = os.path.join(save_dir, f"{self.iter_step}_points_eval.ply")
        pcd_eval = o3d.geometry.PointCloud()
        pcd_eval.points = o3d.utility.Vector3dVector(points_eval)
        o3d.io.write_point_cloud(sampled_points_path, pcd_eval)

        cd, fscore = chamfer_distance_and_f1_score(points_eval, self.dataset.points_gt)
        self.writer.add_scalar('Statistics/cd', cd, self.iter_step)
        self.writer.add_scalar('Statistics/fscore', fscore, self.iter_step)
        return cd, fscore

    def find_visible_points(self, mesh):
        num_view = self.dataset.n_images
        points_list = []
        for view_idx in range(num_view):
            rays_o, rays_v = self.dataset.gen_rays_at(view_idx, resolution_level=1, within_mask=True)
            rays_o, rays_v = rays_o.cpu().detach().numpy(), rays_v.cpu().detach().numpy()
            rays_v = rays_v / np.linalg.norm(rays_v, axis=-1, keepdims=True)
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=rays_o,
                ray_directions=rays_v,
                multiple_hits=False)
            points_list.append(locations)
        return np.concatenate(points_list, axis=0)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='eval_normal')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--obj_name', type=str, default='')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    print(f'Running on the object: {args.obj_name}')

    f = open(args.conf)
    conf_text = f.read()
    conf_text = conf_text.replace('CASE_NAME', args.obj_name)

    runner = Runner(conf_text, args.mode, args.is_continue)
    runner.train()

