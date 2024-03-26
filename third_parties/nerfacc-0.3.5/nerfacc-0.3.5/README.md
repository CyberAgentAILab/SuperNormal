<p>
  <!-- pypi-strip -->
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png" />
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" />
  <!-- /pypi-strip -->
  <img alt="nerfacc logo" src="https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png" width="350px" />
  <!-- pypi-strip -->
  </picture>
  <!-- /pypi-strip -->
</p>

[![Core Tests.](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml/badge.svg)](https://github.com/KAIR-BAIR/nerfacc/actions/workflows/code_checks.yml)
[![Documentation Status](https://readthedocs.com/projects/plenoptix-nerfacc/badge/?version=latest)](https://www.nerfacc.com/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/nerfacc)](https://pepy.tech/project/nerfacc)

https://www.nerfacc.com/

NerfAcc is a PyTorch Nerf acceleration toolbox for both training and inference. It focuses on efficient volumetric rendering of radiance fields, which is universal and plug-and-play for most of the NeRFs.

Using NerfAcc, 

- The `vanilla NeRF` model with 8-layer MLPs can be trained to *better quality* (+~0.5 PNSR)
  in *1 hour* rather than *days* as in the paper.
- The `Instant-NGP NeRF` model can be trained to *equal quality* in *4.5 minutes*,
  comparing to the official pure-CUDA implementation.
- The `D-NeRF` model for *dynamic* objects can also be trained in *1 hour*
  rather than *2 days* as in the paper, and with *better quality* (+~2.5 PSNR).
- Both *bounded* and *unbounded* scenes are supported.

**And it is a pure Python interface with flexible APIs!**

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easist way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).
```
pip install nerfacc
```

Or install from source. In this way it will build the CUDA code during installation.
```
pip install git+https://github.com/KAIR-BAIR/nerfacc.git
```

We also provide pre-built wheels covering major combinations of Pytorch + CUDA supported by [official Pytorch](https://pytorch.org/get-started/previous-versions/).

```
# e.g., torch 1.13.0 + cu117
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu117.html
```

| Windows & Linux | `cu102` | `cu113` | `cu116` | `cu117` |
|-----------------|---------|---------|---------|---------|
| torch 1.10.0    | ✅      | ✅      |         |         |
| torch 1.11.0    | ✅*     | ✅      |         |         |
| torch 1.12.0    | ✅*     | ✅      | ✅      |         |
| torch 1.13.0    |         |         | ✅      | ✅      |

\* Pytorch does not support Windows pre-built wheels for those combinations thus we do not support as well.

## Usage

The idea of NerfAcc is to perform efficient ray marching and volumetric rendering. So NerfAcc can work with any user-defined radiance field. To plug the NerfAcc rendering pipeline into your code and enjoy the acceleration, you only need to define two functions with your radiance field.
- `sigma_fn`: Compute density at each sample. It will be used by `nerfacc.ray_marching()` to skip the empty and occluded space during ray marching, which is where the major speedup comes from. 
- `rgb_sigma_fn`: Compute color and density at each sample. It will be used by `nerfacc.rendering()` to conduct differentiable volumetric rendering. This function will receive gradients to update your network.

A simple example is like this:

``` python
import torch
from torch import Tensor
import nerfacc 

radiance_field = ...  # network: a NeRF model
rays_o: Tensor = ...  # ray origins. (n_rays, 3)
rays_d: Tensor = ...  # ray normalized directions. (n_rays, 3)
optimizer = ...  # optimizer

def sigma_fn(
    t_starts: Tensor, t_ends:Tensor, ray_indices: Tensor
) -> Tensor:
    """ Query density values from a user-defined radiance field.
    :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
    :params t_ends: End of the sample interval along the ray. (n_samples, 1).
    :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
    :returns The post-activation density values. (n_samples, 1).
    """
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
    sigmas = radiance_field.query_density(positions) 
    return sigmas  # (n_samples, 1)

def rgb_sigma_fn(
    t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
) -> Tuple[Tensor, Tensor]:
    """ Query rgb and density values from a user-defined radiance field.
    :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
    :params t_ends: End of the sample interval along the ray. (n_samples, 1).
    :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
    :returns The post-activation rgb and density values. 
        (n_samples, 3), (n_samples, 1).
    """
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
    rgbs, sigmas = radiance_field(positions, condition=t_dirs)  
    return rgbs, sigmas  # (n_samples, 3), (n_samples, 1)

# Efficient Raymarching: Skip empty and occluded space, pack samples from all rays.
# ray_indices: (n_samples,). t_starts: (n_samples, 1). t_ends: (n_samples, 1).
with torch.no_grad():
    ray_indices, t_starts, t_ends = nerfacc.ray_marching(
        rays_o, rays_d, sigma_fn=sigma_fn, near_plane=0.2, far_plane=1.0, 
        early_stop_eps=1e-4, alpha_thre=1e-2, 
    )

# Differentiable Volumetric Rendering.
# colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
color, opacity, depth = nerfacc.rendering(
    t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
)

# Optimize: Both the network and rays will receive gradients
optimizer.zero_grad()
loss = F.mse_loss(color, color_gt)
loss.backward()
optimizer.step()
```

## Examples: 

Before running those example scripts, please check the script about which dataset it is needed, and download the dataset first.

```bash
# clone the repo with submodules.
git clone --recursive git://github.com/KAIR-BAIR/nerfacc/
```

``` bash
# Instant-NGP NeRF in 4.5 minutes with reproduced performance!
# See results at here: https://www.nerfacc.com/en/latest/examples/ngp.html
python examples/train_ngp_nerf.py --train_split train --scene lego
```

``` bash
# Vanilla MLP NeRF in 1 hour with better performance!
# See results at here: https://www.nerfacc.com/en/latest/examples/vanilla.html
python examples/train_mlp_nerf.py --train_split train --scene lego
```

```bash
# D-NeRF for Dynamic objects in 1 hour with better performance!
# See results at here: https://www.nerfacc.com/en/latest/examples/dnerf.html
python examples/train_mlp_dnerf.py --train_split train --scene lego
```

```bash
# Instant-NGP on unbounded scenes in 20 minutes!
# See results at here: https://www.nerfacc.com/en/latest/examples/unbounded.html
python examples/train_ngp_nerf.py --train_split train --scene garden --auto_aabb --unbounded --cone_angle=0.004
```

Used by:
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio): A collaboration friendly studio for NeRFs.
- [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl): NeuS in 10 minutes.


## Common Installation Issues


<details>
    <summary>ImportError: .../csrc.so: undefined symbol</summary>
    If you are installing a pre-built wheel, make sure the Pytorch and CUDA version matchs with the nerfacc version (nerfacc.__version__).
</details>

## Citation

```bibtex
@article{li2022nerfacc,
  title={NerfAcc: A General NeRF Accleration Toolbox.},
  author={Li, Ruilong and Tancik, Matthew and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2210.04847},
  year={2022}
}
```
