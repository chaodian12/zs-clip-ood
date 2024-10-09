# zs-clip-ood
Zero-Shot Out-of-Distribution Detection by Co-generating Fine-grained Descriptions
![main_structure](main.png)
## Installation
The project is based on PyTorch.
Below are quick steps for installation:
```shell
conda create -n open-mmlab python=3.8 pytorch==1.10 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
cd NegLabel
mim install -e .
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
## Dataset Preparation
### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data (not necessary) and validation data in
`./data/id_data/imagenet_train` and  `./data/id_data/imagenet_val`, respectively.

### Out-of-distribution dataset

We use the following 4 OOD datasets for evaluation: [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf).

Please refer to [MOS](https://github.com/deeplearning-wisc/large_scale_ood), download OOD datasets and put them into `./data/ood_data/`.
