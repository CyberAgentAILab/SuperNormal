conda deactivate
conda remove -y -n sn --all
conda create -y -n sn python=3.8
conda activate sn

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install tiny-cuda-nn
export PATH="/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -e ./third_parties/nerfacc-0.3.5/nerfacc-0.3.5/
pip install opencv-python==4.8.1.78 trimesh==3.23.5 open3d==0.17 pyvista==0.42.3 scipy==1.10.1 scikit-image==0.21.0 pyhocon==0.3.59 pyexr==0.3.10 tensorboard==2.14.0 icecream==2.1.3 PyMCubes==0.1.4 pyembree==0.2.11