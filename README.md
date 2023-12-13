
# PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection

This repository contains the source code for the paper "PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection." PViT introduces a novel approach to enhance Vision Transformers with priors for improved out-of-distribution detection.

## Getting Started

### Dependencies

1. **Create a Conda Environment**:
   ```bash
   conda create -n pvit python=3.8
   conda activate pvit
   ```

2. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

- Download the OOD datasets as per instructions in the [Open-OOD](https://github.com/Jingkang50/OpenOOD) repository.
- Update the file paths in `dataloader.py` and `utils.py` to reflect your local dataset paths.

## Training PViT Models

### CIFAR-10

- To train on CIFAR-10 using a pre-trained ViT as the prior model, run the following command:
  ```bash
  python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit --num_epochs 10 --dataset cifar10 --alpha_weight 1 --prior_token_position all
  ```

### CIFAR-100

- For CIFAR-100, modify the `--dataset` argument:
  ```bash
  python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit --num_epochs 10 --dataset cifar100 --alpha_weight 1 --prior_token_position all
  ```

### IMAGENET-100

- To train on IMAGENET-100 using a pre-trained ViT as the prior model, run the following command:
  ```bash
  python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit_imagenet --num_epochs 10 --dataset imagenet --alpha_weight 1 --prior_token_position all
  ```

## Evaluation

- Evaluate the trained model on CIFAR-10:
  ```bash
  python test.py --model_size small --batch_size 256 --prior_model_name vit --num_epochs 10 --num_workers 16 --dataset cifar10 --score cross_entropy --alpha_weight 1 --prior_token_position all
  ```

## Contributing

