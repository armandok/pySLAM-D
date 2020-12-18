# pySLAM-D

pySLAM-D is a SLAM code for **RGB-D** images in *python* that computes the camera trajectory and yileds a 3D reconstruction of the scene. This code is suitable for Reinforcement Learning purposes and utilizes existing C++ libraries for fast visual odometry, loop-closure detection, and pose graph optimization. We have tested this code for visual SLAM in the [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) environement.



## Installation
We have tested this library in **Ubuntu 18.04** and **CentOS 7**.

### 1. Creating the conda environment 
We suggest using a conda environment.
```
conda update -q conda
conda create -q -n pyslamd python=3.7 opencv=3.4.* numpy boost py-boost cmake eigen
conda activate pyslamd
```
### 2. GTSAM
The [gtsam](https://github.com/borglab/gtsam) library is used for pose graph optimization. Make sure to build it with python bindings.

### 3. Installing dependencies and third party libraries
We use Open3D for 3D data handling.
```
conda install -c open3d-admin open3d
```

We make use of the [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus) library for a robust and certifiable front-end.
For loop-closure detection, we use the [FBOW](https://github.com/rmsalinas/fbow) library and a modified version of [pyfbow](https://github.com/vik748/pyfbow) for python bindings. These two libraries are included in the Thirdpary folder. To install them, follow these instructions:

```
cd Thirdparty
chmod +x build.sh
./build.sh
```

### 4. Using with Habitat-Sim
run habitat.py

