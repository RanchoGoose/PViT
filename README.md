# PViT: Prior-Augmented Vision Transformer for Out-of-Distribution Detection

Official implementation of [PViT: Prior-Augmented Vision Transformer for Out-of-Distribution Detection](https://arxiv.org/abs/2410.20631).

## Overview

PViT is a novel framework that enhances Vision Transformer robustness for image Out-of-Distribution (OOD) detection. It trains a ViT to predict class labels using both image tokens and prior class logits from a pretrained model, then identifies OOD samples by measuring the divergence between predicted and prior logits. PViT achieves state-of-the-art performance without requiring additional data modeling, generation methods, or structural modifications.

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Prepare Datasets

#### 1. ImageNet-1K
Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from the official [ImageNet website](https://www.image-net.org/). Use the provided script to extract:
```bash
bash ./dataloaders/assets/extract_ILSVRC.sh
```
For dataset folder structure details, see `./dataloaders/README.md`.

#### 2. OOD Datasets
Download iNaturalist, SUN, and Places:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
- **Textures**: Download from the [official website](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- **OpenImage-O**: Download from [ViM repo](https://github.com/haoqiwang/vim/tree/master/datalists)
- **Other datasets**: Refer to [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main)

#### 3. Pretrained Models
Download `resnet50-supcon.pt` from [this link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and place it in `./pretrained_models/`. Other prior models are automatically downloaded from HuggingFace or PyTorch Hub.

## Training

Train PViT on ImageNet-1K:
```bash
python main.py --model_name pvit --id_data_name imagenet1k --ood_data_name inaturalist \
    --ood_detectors pvit --pvit --batch_size 256 --num_workers 1 \
    --prior_model vit-b-16 --score cross_entropy --seed 0 --train
```

Train PViT on CIFAR-100:
```bash
python main.py --model_name pvit --id_data_name cifar100 --ood_data_name cifar10 \
    --ood_detectors pvit --pvit --batch_size 512 --num_workers 1 \
    --prior_model vit_cifar100 --score cross_entropy --seed 0 --train
```

### Available Prior Models
| Argument | Model |
|----------|-------|
| `vit_imagenet` | Google ViT |
| `vit-b-16` | DeiT |
| `vit-lp` | ViT-LP |
| `resnet50-supcon` | ResNet50-SupCon |
| `regnet-y-16gf-swag-e2e-v1` | RegNet |
| `vit-b16-swag-e2e-v1` | ViT-SWAG |

## Evaluation

```bash
python main.py --model_name pvit --id_data_name imagenet1k --ood_data_name inaturalist \
    --ood_detectors pvit --batch_size 512 --num_workers 1 \
    --prior_model vit-lp --pvit --score cross_entropy
```

### OOD Scoring Functions
| Argument | Method |
|----------|--------|
| `cross_entropy` | PGE with Cross-Entropy (CE) guidance |
| `KL` | PGE with KL-Divergence guidance |
| `dis` | PGE with Euclidean-Distance (ED) guidance |
| `guidance_only` | CE guidance term alone (ablation) |
| `additive` | Additive combination `S_base - G` (ablation) |

### Score Orientation Convention

All detectors return **confidence scores**: the higher the score, the more
ID-like the input. The evaluation harness (`main.py` +
`eval_assets.compute_ood_performances`) labels ID samples as the positive
class and consumes scores unmodified.

The paper's PGE score `S_PGE = S_base * G` (Eq. 12) is an *OOD-ness* score:
the guidance term `G` is a divergence between the prior distribution and
PViT's prediction, so `S_PGE` is small for ID and large for OOD. The
detector therefore returns `-S_PGE`, which is equivalent to the paper's
decision rule "**x is ID iff `S_PGE < γ`**" (Eq. 13). This is the
orientation under which all results in the paper were computed.

## Reproduce Paper Results

To reproduce the results without training:
1. Download the [saved model outputs](https://drive.google.com/file/d/170lh8DJLK3uPScxDbvwqbmOriHODM5gT/view?usp=sharing)
2. Place the unzipped folder in the root directory
3. Run the evaluation command above with the desired configuration

## Citation

```bibtex
@article{zhang2024pvit,
  title={PViT: Prior-Augmented Vision Transformer for Out-of-Distribution Detection},
  author={Zhang, Tianhao and Chen, Zhixiang and Mihaylova, Lyudmila S.},
  journal={arXiv preprint arXiv:2410.20631},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
