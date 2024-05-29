# 配置说明
## Installation

First, use pip to install `omniglue`:

```sh
conda create -n omniglue pip
conda activate omniglue

git clone https://github.com/google-research/omniglue.git
cd omniglue
pip install -r requirements.txt
```

Then, download the following models to `./models/`

```sh
# Download to ./models/ dir.
mkdir models
cd models

# SuperPoint.
git clone https://github.com/rpautrat/SuperPoint.git
mv SuperPoint/pretrained_models/sp_v6.tgz . && rm -rf SuperPoint
tar zxvf sp_v6.tgz && rm sp_v6.tgz

# DINOv2 - vit-b14.
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# OmniGlue.
wget https://storage.googleapis.com/omniglue/og_export.zip
unzip og_export.zip && rm og_export.zip
```

Direct download links:

-   [[SuperPoint weights]](https://github.com/rpautrat/SuperPoint/tree/master/pretrained_models): from [github.com/rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint)
-   [[DINOv2 weights]](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth): from [github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) (ViT-B/14 distilled backbone without register).
-   [[OmniGlue weights]](https://storage.googleapis.com/omniglue/og_export.zip)

## Usage
The code snippet below outlines how you can perform OmniGlue inference in your
own python codebase.

```py

import omniglue

image0 = ... # load images from file into np.array
image1 = ...

og = omniglue.OmniGlue(
  og_export='./models/og_export',
  sp_export='./models/sp_v6',
  dino_export='./models/dinov2_vitb14_pretrain.pth',
)

match_kp0s, match_kp1s, match_confidences = og.FindMatches(image0, image1)
# Output:
#   match_kp0: (N, 2) array of (x,y) coordinates in image0.
#   match_kp1: (N, 2) array of (x,y) coordinates in image1.
#   match_confidences: N-dim array of each of the N match confidence scores.
```

## Demo

`demo.py` contains example usage of the `omniglue` module. To try with your own
images, replace `./res/demo1.jpg` and `./res/demo2.jpg` with your own
filepaths.

```sh
conda activate omniglue
python demo.py ./res/demo1.jpg ./res/demo2.jpg
# <see output in './demo_output.png'>
```

Expected output:
![demo_output.png](res/demo_output.png "demo_output.png")
