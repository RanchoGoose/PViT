# PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection

This is the source code for the paper [PViT: Prior-augmented Vision Transformer for Out-of-distribution Detection](). Implementation details will be published on Github soon.
# PViT
To simply train a PViT model on CIFAR-10, run:
```python python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit --num_epochs 10  --dataset cifar10 --alpha_weight 1 --prior_token_position all
```
modify the --dataset to CIFAR100 to train on CIFAR-100.

To train on IMAGENET-100, run:
```python python train.py --model_size small --batch_size -1 --num_workers 16 --prior_model_name vit_imagenet --num_epochs 10  --dataset imagenet --alpha_weight 1 --prior_token_position all
```

To evaluate the trained model, run:
```python test.py --model_size small --batch_size 256 --prior_model_name vit --num_epochs 10 --num_workers 16 --dataset cifar --score cross_entropy --alpha_weight 1 --prior_token_position all
```
