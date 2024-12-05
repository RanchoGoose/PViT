# PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection

This repository contains the source code for the paper titled [PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection](https://arxiv.org/abs/2410.20631). 

Note: We apologize for any inconvenience caused by the current state of the code. Due to ongoing improvements, the training and testing scripts are located in separate folders. We are working on consolidating the code into a single repository, which will be released upon the acceptance of the paper.

### Setup
Install the `./requirements.txt`.

Then install the required packages by
```
chmod +x install_packages.sh
./install_packages.sh
```

### Prepare Dataset
To set up the dataset folder structures, please refer to the `README.md` file located in the `./dataloaders` directory.

#### 1. Download ImageNet-1k:
Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from the official ImageNet [website](). And use `./dataloaders/assets/extract_ILSVRC.sh` to unzip the zip files.

#### 2. Download iNaturalist, SUN, Places, Textures, OpenImage-O OOD datasets:
To download iNaturalist, SUN, and Places
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
Download Textures from the official [website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
Download OpenImage-O from the official [website](https://github.com/haoqiwang/vim/tree/master/datalists).

#### 3. Additional Datasets
For other datasets, please refer to the [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main)

#### 4. Pretrained models
Download `resnet50-supcon.pt` from the [link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and put it in the directory `pretrained_models` as `./pretrained_models/resnet50-supcon.py`.

Other prior models will be automatically downloaded from Huggingface or Pytorch.

# Train PViT

For instructions on training the PViT model, please refer to the `README.md` in the folder `./train`


To train the PViT model with different prior models, modify the `--prior_model_name` argument. The available options are:

- `vit_imagenet` for Google ViT
- `vit-b-16` for DeiT
- `vit-lp` for ViT-LP
- `resnet50-supcon` for ResNet50-SupCon
- `regnet-y-16gf-swag-e2e-v1` for RegNet
- `vit-b16-swag-e2e-v1` for ViT-Swag

# Evaluate PViT

To run experiments, run
```
python main.py --model_name pvit --id_data_name imagenet1k --ood_data_name inaturalist --ood_detectors pvit --batch_size 512 --num_workers 1 --prior_model vit-lp --pvit --score cross_entropy
```
modify the `--score` argument. The available options are:

- `cross_entropy` for CE
- `KL` for KL divergence
- `dis` for ED

modify the `--prior_model` argument to evaluate with different prior models.

# Reproduce the results in the paper
To simply reproduce the results presented in the paper:
1. Download the [saved model outputs](https://drive.google.com/file/d/170lh8DJLK3uPScxDbvwqbmOriHODM5gT/view?usp=sharing) . If you do this, you do not need to download the dataset. 
2. Place the unzipped folder in the root directory.

