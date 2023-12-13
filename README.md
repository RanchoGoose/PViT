# PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection
This is the source code for the paper [PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection](). 

# Dependencies
create your conda environment and install the dependencies with:
```conda create -n pvit```
```pip install -r requirements.txt```

# Dataset Preparation
To download the OOD datasets please refer to the [Open-OOD](https://github.com/Jingkang50/OpenOOD) repository.
Remember to modify the path in the dataloader.py and the utils.py files to your own path.

# PViT
To simply train a PViT model on CIFAR-10, run:
```python python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit --num_epochs 10  --dataset cifar10 --alpha_weight 1 --prior_token_position all```
To train on CIFAR-100, modify the --dataset to CIFAR100.

To train on IMAGENET-100, run:
```python python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit_imagenet --num_epochs 10  --dataset imagenet --alpha_weight 1 --prior_token_position all```

To evaluate the trained model, run:
```python test.py --model_size small --batch_size 256 --prior_model_name vit --num_epochs 10 --num_workers 16 --dataset cifar --score cross_entropy --alpha_weight 1 --prior_token_position all```
