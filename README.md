# semantic_hazard_cloud
Semantically identifiy hazard and create a 3D color coded cloud

# Installation
This package is tested with TensorFlow 2.0.0, CUDA 10.1. It's higly recommended to run keras-segmentation using GPU to have higher execution speed.

Some required packges/modules before installing image-segmentation-kerasL
```sh
pip install launchpadlib==1.10.6
pip install setuptools==41.0.0
pip install tensorflow==2.0.0
pip install cntk
pip install 'scikit-image<0.15'
pip install opencv-python

pip install Theano
cd ~
git clone https://github.com/Theano/libgpuarray.git
cd libgpuarray
git checkout tags/v0.6.5 -b v0.6.9
mkdir Build
cd Build
cmake .. -DCMAKE_BUILD_TYPE=Release # or Debug if you are investigating a crash
make
sudo make install
cd ..
# for pygpu
# This must be done after libgpuarray is installed as per instructions above.
python setup.py build
python setup.py install
sudo ldconfig

sudo apt-get install -y python-mako
apt-get install -y libsm6 libxext6 libxrender-dev
```

Setup `keras-segmentation`

```sh
cd ~
git clone https://github.com/divamgupta/image-segmentation-keras
```

In [predict.py](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/predict.py), change the line with `return pr` to `return seg_img`

Build and install

```sh
cd image-segmentation-keras
python setup.py install
```

Build this package
```sh
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel

cd ~/catkin_ws/src
# You need access to clone this package
git clone https://github.com/kucars/semantic_hazard_cloud.git

cd ..
catkin build
```
