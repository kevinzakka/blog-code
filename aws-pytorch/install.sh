######################################################################
# Driver part is based off Michael Dietz's gist which you can view @
# https://gist.github.com/mjdietzx/fda9535e3246f0db39b0da80403265d1
######################################################################

# drivers
wget http://us.download.nvidia.com/tesla/375.51/nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo dpkg -i nvidia-driver-local-repo-ubuntu1604_375.51-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
sudo apt-get update && sudo apt-get -y upgrade

# python3 misc
sudo apt-get --assume-yes install python3-tk
sudo apt-get --assume-yes install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv numpy scipy matplotlib

# virtualenv
mkdir envs
cd envs
virtualenv --system-site-packages deepL

# install pytorch
source ~/envs/deepL/bin/activate
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl
pip install torchvision tensorboard_logger tqdm

echo "Reboot required."
