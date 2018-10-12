#! /bin/bash

# drivers
wget http://us.download.nvidia.com/tesla/375.66/nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
dpkg -i nvidia-diag-driver-local-repo-ubuntu1604_375.66-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get update && sudo apt-get -y upgrade

# python
sudo apt-get install unzip
sudo apt-get --assume-yes install python3-tk
sudo apt-get --assume-yes install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv numpy scipy matplotlib

# virtualenv
mkdir envs
cd envs
virtualenv --system-site-packages deepL

# pytorch
source ~/envs/deepL/bin/activate
pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install torchvision tqdm

sudo reboot
