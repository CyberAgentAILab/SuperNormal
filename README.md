<h2 align="center">SuperNormal: Neural Surface Reconstruction via Multi-View Normal Integration</h2>
<h4 align="center">
    <a href="https://xucao-42.github.io/homepage/"><strong>Xu Cao</strong></a>
    ·
    <a href="https://taketomitakafumi.sakura.ne.jp/web/en/"><strong>Takafumi Taketomi</strong></a>
<br>
CyberAgent </h4>
<h4 align="center"><a href="https://cvpr.thecvf.com/">CVPR 2024 </a></h3>
<p align="center">
  <br>
    <a href="https://arxiv.org/abs/2312.04803">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>

[//]: # (    <a href='https://xucao-42.github.io/mvas_homepage/'>)

[//]: # (      <img src='https://img.shields.io/badge/MVAS-Project Page-5468FF?style=for-the-badge' alt='Project Page'></a>)
</p>

<div align="center">
<img src="./media/teaser.png" alt="Teaser" width="100%">
Fast and fine-grained 3D reconstruction from multi-view surface normal maps. 
</div>

### Quick Start
Code was tested on Ubuntu 18.04 (WSL2) using Python 3.8, PyTorch 2.1.0, and CUDA 11.8 on an Nvidia RTX4090Ti (24GB). 

**Before started, please ensure CUDA is installed in your environment ([11.8 can be found here](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)).**
It is required by [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

<details><summary> You should see something like the following after typing `nvcc --version` </summary>

```commandline
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
</details>

Clone the repository and prepare the conda environment:
```commandline
git clone https://github.com/CyberAgentAILab/SuperNormal.git
cd SuperNormal
./create_env.sh
```

Download data (~1.8GB):
```commandline
./download_data.sh
```

Run on the DiLiGenT-MV benchmark objects or on our captured objects:
```commandline
./run_diligent.sh  # Training should take about 50 seconds per object
```
or 
```commandline
./run_own_object.sh  # Training should take about 5 minutes per object
```
Results are saved under `./exp`.

NOTE: If RuntimeError like below occurs, `apt install ninja-build` may resolve the error.
```
RuntimeError: Ninja is required to load C++ extensions
```

### Hyperparameter tuning tips
Training hyperparameters are defined in `./configs/*.conf`.
Some important hyperparameters are:
- `dataset.normal_dir`: You can choose normal maps estimated by different methods as input for DiLiGenT-MV benchmark objects.
- `train.end_iter`: The number of iterations for training. Should be adjusted according to the number of views and normal map resolutions.
- `train.increase_bindwidth_every`: A strategy used in [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/) to progressively activate finer hash grid during training. Less than `end_iter`/`model.encoding.n_levels` should be fine.
- `train.batch_size`: Number of patches in each batch for training. Should be adjusted according to the GPU memory.
- `train.patch_size`: Better to be fixed to 3, i.e., each patch is 3x3. Large patch size will cause inaccurate volume rendering results for boudary pixels in a patch.

### Modifications to NerfAcc
We add several functions to the original [NerfAcc](https://www.nerfacc.com) to adapt it to patch-based volume rendering.
The key new functions (which are indicated by `patch_based` in function name) are in 
[third_parties/nerfacc-0.3.5/nerfacc-0.3.5/nerfacc/cuda/csrc/render_weight.cu/](./third_parties/nerfacc-0.3.5/nerfacc-0.3.5/nerfacc/cuda/csrc/render_weight.cu) 
and [third_parties/nerfacc-0.3.5/nerfacc-0.3.5/nerfacc/vol_rendering.py](./third_parties/nerfacc-0.3.5/nerfacc-0.3.5/nerfacc/vol_rendering.py).


### Acknowledgement
This repo is built up on [NeuS](https://github.com/Totoro97/NeuS) and benefits from the amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [NerfAcc](https://www.nerfacc.com).
We also learned a lot from [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl).

### Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{supernormal2024cao,
  title={SuperNormal: {N}eural Surface Reconstruction via Multi-View Normal Integration},
  author={Cao Xu and Taketomi Takafumi},
  booktitle={CVPR},
  year={2024}
}
```


