conda create --name mvdecor
conda activate mvdecor

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install trimesh==3.9.29
pip install open3d=0.10.0.0
pip install h5py
pip install pyrender==0.1.45
pip install pyrr==0.10.3
conda install scikit-learn==0.24.1
pip install rtree==0.9.7
pip install pyembree
pip install tensorboard==2.8.0