conda deactivate
conda remove -y -n sn --all
conda create -y -n sn python=3.8
eval "$(conda shell.bash hook)"
conda activate sn

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install tiny-cuda-nn
export PATH="/usr/local/cuda/bin:$PATH"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install -e ./third_parties/nerfacc-0.3.5/nerfacc-0.3.5/
pip install opencv-python trimesh open3d pyvista scipy scikit-image pyhocon==0.3.59 pyexr tensorboard icecream PyMCubes pyembree