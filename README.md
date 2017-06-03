# geweldig
CS231N Project - Rijksmuseum classification
Dream Team: Tara, Sarah, Emily

# Downloading the dataset
The dataset exists in a CSV file, which was created as follows:
`python download_dataset.py > museum_data.csv`
Since the dataset may be too large (with 112,000+ images), we wrote a script to download images from the dataset.
The script reads the CSV file and then uses a threadpool (using multiprocessing) to parallelize the image downloads. 

## CSV file
The columns in the museum_data.csv are
id, image_url, principalOrFirstMaker, title, longTitle, width, height

## To download images
In the python script download_images.py, make sure the `image_dir` variable in line 12 is updated to your own path. 
It will exist in /data/images, so make sure the `images` directory exists (if not just mkdir it). 
Then run `python download_images.py`

# Installing CMAKE
In order to extract image features (SURF) we need to install OpenCV/CMAKE
https://geeksww.com/tutorials/operating_systems/linux/installation/downloading_compiling_and_installing_cmake_on_linux.php

If this doesn't work try:
`sudo apt-get install python-opencv`
# Installing homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# Installing cv2 with SURF
Brew may depend on curl.h
`sudo apt-get install libcurl4-openssl-dev`

`brew install git cmake pkg-config jpeg libpng libtiff openexr eigen tbb`

`conda install -c menpo opencv3`

`conda update hdf5`

# Installing correct tensorflow version on GPU:
Check python version: 
$ python -V

If python 2.7, use
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc0-cp27-none-linux_x86_64.whl

If python 3.5 us
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.0rc0-cp35-cp35m-linux_x86_64.whl

# Tensorboard

See here for more information https://www.tensorflow.org/get_started/summaries_and_tensorboard

Make sure you set static IP for the GPU instance, and then run the following:
tensorboard --logdir=train --port=7000

# Installing PyTorch

# 1. Install Conda
`wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh`
`bash Miniconda2-latest-Linux-x86_64.sh`
Type `:q` to leave the License agreement, and make sure you type "yes" to adding to .bashrc!

# 2. Install pytorch
From http://pytorch.org/
`conda install pytorch torchvision cuda80 -c soumith`

If it says conda isn't found, try this to double check conda:
`export PATH=/home/emjtang/miniconda2/bin:$PATH`
`conda --version` 



