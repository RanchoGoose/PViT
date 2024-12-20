import math
import argparse
import random
import csv
import os
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image

from backbones.pvit_backbone import PViT
# from model_engines.resnet18_32x32 import ResNet18_32x32 
from model_engines.resnet50_supcon import ResNetSupCon
from model_engines.mobilenet_v2 import MobileNet
from model_engines.vit_b16_swag_e2e_v1 import ViT
from model_engines.resnet50_react import ResNet
from model_engines.regnet import RegNet
from model_engines.swin_t import Swin_T
from torch.hub import load_state_dict_from_url
from torchvision.models import vit_b_16, ViT_B_16_Weights, Swin_T_Weights

import timm
import xml.etree.ElementTree as ET
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch.distributed as dist
import detectors

def get_transformer_config(model_size, image_size=224, patch_size=16, dropout=0.1, emb_dropout=0.1):
    # Define a dictionary to map model_size to parameters
    model_params = {
        'tiny': {'dim': 192, 'depth': 12, 'heads': 3, 'mlp_dim': 384},
        'small': {'dim': 384, 'depth': 12, 'heads': 6, 'mlp_dim': 768},
        'medium': {'dim': 768, 'depth': 12, 'heads': 8, 'mlp_dim': 1536},
        'big': {'dim': 1024, 'depth': 24, 'heads': 16, 'mlp_dim': 2048},
        'huge': {'dim': 1280, 'depth': 36, 'heads': 20, 'mlp_dim': 2560}
    }

    # Get the parameters for the given model_size
    params = model_params.get(model_size, {})

    config = {
        'image_size': image_size,
        'patch_size': patch_size,
        'dim': params.get('dim', 0),  # Default to 0 if model_size is invalid
        'depth': params.get('depth', 0),
        'heads': params.get('heads', 0),
        'mlp_dim': params.get('mlp_dim', 0),
        'dropout': dropout,
        'emb_dropout': emb_dropout
    }

    return config

def get_transformer_model(args, model_spec, num_classes):
    model_cls = PViT
    if args.prior_model in ["resnet18_32x32", "resnet18_cifar100"]:
        image_size = 32 
        patch_size = 2
    elif args.prior_model == "vit-b16-swag-e2e-v1":
        image_size = 384
        patch_size = 16
    else:
        image_size = model_spec['image_size']
        patch_size = model_spec['patch_size']
    
    model = model_cls(args=args,
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = model_spec['dim'],
        depth = model_spec['depth'],
        heads = model_spec['heads'],
        mlp_dim = model_spec['mlp_dim'],
        dropout = model_spec['dropout'],
        emb_dropout = model_spec['emb_dropout']
        )
    return model

def get_openai_lr(transformer_model):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)

def load_prior_model(args, device, num_outputs):
    try:
        state_dict = torch.load(
            './pretrained_models/resnet50-supcon.pt',
            map_location='cpu',
            weights_only=True,
            pickle_module=torch.serialization.safe_pickle
        )['model_state_dict']
    except Exception as e:
        print(f"Warning: Failed to load with weights_only=True, falling back to default loading for compatibility: {e}")
        state_dict = torch.load(
            './pretrained_models/resnet50-supcon.pt',
            map_location='cpu'
        )['model_state_dict']

    if args.prior_model == 'vgg16':
        net = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    elif args.prior_model == 'resnet18':
        net = timm.create_model("resnet18_cifar10", pretrained=True)
    elif args.prior_model == 'resnet50':
        net = timm.create_model("resnet50_cifar10", pretrained=True)
    elif args.prior_model == 'wrn':
        net = WideResNet(layers, num_outputs, widen_factor=2, dropRate=0.3)
        model_path = os.path.join(os.path.join(args.load_prior_path), 'cifar10_wrn_pretrained_epoch_99.pt')
        net.load_state_dict(torch.load(model_path))       
    elif args.prior_model == 'densenet':
        net = timm.create_model("densenet121_cifar10", pretrained=True)
        # net = DenseNet3(100, num_outputs, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None,
        #                 k=None, info=None)
        # model_path = os.path.join(os.path.join(args.load_prior_path), '.pt')
        # net.load_state_dict(torch.load(model_path))
    elif args.prior_model == 'vit':
        net = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    elif args.prior_model == 'BEiT':
        net = AutoModelForImageClassification.from_pretrained("jadohu/BEiT-finetuned")
    elif args.prior_model == 'vit_imagenet':
        net = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    elif args.prior_model == 'resnet18_imagenet':
        net = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
    elif args.prior_model == 'resnet50_imagenet':
        net = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    elif args.prior_model == 'resnet101_imagenet':
        net = AutoModelForImageClassification.from_pretrained("microsoft/resnet-101")
    elif args.prior_model == 'resnet18_cifar100':
        net = timm.create_model("resnet18_cifar100", pretrained=True)  
    elif args.prior_model == 'resnet50_cifar100':
        net = timm.create_model("resnet50_cifar100", pretrained=True)
    elif args.prior_model == 'vit_cifar100':    
        net = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100") 
    # elif args.prior_model == 'resnet18_imagenet200':
    #     net = ResNet18_224x224()
    #     model_path = os.path.join(os.path.join(args.load_prior_path), 'imagenet200_resnet18_224x224_base_e90_lr0.1_default/s2/best_epoch88_acc0.8480.ckpt')
    #     net.load_state_dict(torch.load(model_path)) 
    # elif args.prior_model == 'resnet18_32x32':    
    #     net = ResNet18_32x32(num_classes=100)
    #     net.load_state_dict(
    #     torch.load(os.path.join('/mnt/parscratch/users/coq20tz/OpenOOD/scripts/best.ckpt'), map_location='cpu'))
    elif args.prior_model == "resnet50":
        net = ResNet()
    elif args.prior_model == "resnet50-supcon":
        net = ResNetSupCon()
        net.load_state_dict(state_dict, strict=False) 
    # elif args.prior_model == "vit-b16-swag-e2e-v1":
    #     net = ViT(model_name='vit-lp')s
    elif args.prior_model == "vit-b16-swag-e2e-v1":
        net = ViT(model_name='vit-b16-swag-e2e-v1')
    elif args.prior_model == "vit-b-16":
        net = ViT(model_name='vit-lp')
    elif args.prior_model == "vit-lp":
        net = ViT(model_name='vit-lp')
    elif args.prior_model == "regnet-y-16gf-swag-e2e-v1":
        net = RegNet(model_name=args.prior_model)
    elif args.prior_model == "mobilenet-v2":
        net = MobileNet()
    elif args.prior_model == 'swin-t':
        net = ViT(model_name='vit-b-16')
        # net = Swin_T()
        # weights = eval(f'Swin_T_Weights.IMAGENET1K_V1')
        # net.load_state_dict(load_state_dict_from_url(weights.url))

    net.to(device)
    net.eval()
    return net

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    a warmup period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Custom transform that returns both 32x32 and 224x224 resolutions
class DualResolutionTransform:
    def __init__(self, mean, std):
        self.transform_32 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.transform_224 = transforms.Compose([
            transforms.Resize(256),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, image):
        return {
                'image_32': self.transform_32(image),
                'image_224': self.transform_224(image)
                }

# Custom dataset wrapper
class DualResolutionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image['image_32'], image['image_224'], label

def get_batch_size(args, num_gpu):
    if args.model_size not in ['tiny', 'small', 'medium', 'big', 'huge']:
        raise ValueError(f'Invalid model size: {args.model_size}')
    
    # Check if the model name matches any of the given names
    is_vit_based = args.prior_model in ['vit', 'BEiT']
    
    # base_batch_size = 128 if is_vit_based else 256
    base_batch_size = 128 
    # Adjust base batch size based on dataset
    if args.dataset == 'CIFAR10':
        base_batch_size *= 2   
    if args.model_size in ['big', 'huge']:
        base_batch_size //= 2
    
    return base_batch_size * num_gpu

# def load_model(args, model):
#     # unwrap module if model was wrapped in DataParallel
#     start_epoch = 0
    
#     if isinstance(model, nn.DataParallel):
#         model = model.module

#     for i in range(50 - 1, -1, -1):
#         if args.id_data_name == 'cifar100':
#             model_name = os.path.join(os.path.join("/mnt/parscratch/users/coq20tz/Bayesiantransformer/snapshots"), f'all_True_cifar100_{args.alpha_weight}_small_{args.prior_model}_{(i)}.pt')
#         else:
#             model_name = os.path.join(os.path.join("/mnt/parscratch/users/coq20tz/Bayesiantransformer/snapshots"), f'all_True_imagenet_{args.alpha_weight}_small_{args.prior_model}_{(i)}.pt')

#         if os.path.isfile(model_name):
#                 # load state_dict
#                 model.load_state_dict(torch.load(model_name))
#                 print('Model restored! Epoch:', i)
#                 start_epoch = i + 1
#                 break
#     if start_epoch == 0:
#         assert False, "could not resume "+ model_name

#     return model

def load_model(args, model):
    # unwrap module if model was wrapped in DataParallel
    start_epoch = 0
    pathname = f'pvit_{args.id_data_name}_{args.alpha_weight}_{args.prior_model}'
    
    if isinstance(model, nn.DataParallel):
        model = model.module

    for i in range(50 - 1, -1, -1):
        model_save_path = os.path.join(args.model_save_path, f'{pathname}_{(i)}.pt')

        if os.path.isfile(model_name):
                # load state_dict
                model.load_state_dict(torch.load(model_name))
                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
            
    if start_epoch == 0:
        assert False, "could not resume "+ model_name

    return model

def build_dataset(args, batch_size):
    # mean and standard deviation
    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    batch_size = batch_size
    
    if args.dataset == 'CIFAR10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        if any(keyword in args.prior_model for keyword in ["vit", "BEiT"]):
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            dataset_train = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            transform_dual = DualResolutionTransform(mean, std)
            cifar_train_dataset_dual = datasets.CIFAR10(root='./data', train=True, transform=transform_dual, download=False)
            train_dataset = DualResolutionDataset(cifar_train_dataset_dual)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            cifar_test_dataset_dual = datasets.CIFAR10(root='./data', train=False, transform=transform_dual, download=False)
            test_dataset = DualResolutionDataset(cifar_test_dataset_dual)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        if any(keyword in args.prior_model for keyword in ["vit", "BEiT"]):
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            dataset_train = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            transform_dual = DualResolutionTransform(mean, std)
            cifar_train_dataset_dual = datasets.CIFAR100(root='./data', train=True, transform=transform_dual, download=True)
            train_dataset = DualResolutionDataset(cifar_train_dataset_dual)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            cifar_test_dataset_dual = datasets.CIFAR100(root='./data', train=False, transform=transform_dual, download=True)
            test_dataset = DualResolutionDataset(cifar_test_dataset_dual)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        num_classes = 100
    elif args.dataset == 'IMAGENET1K':
        # num_tasks = dist.get_world_size()
        # global_rank = dist.get_rank()
    
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        dataset_train = ImageNet1K(root="/mnt/parscratch/users/coq20tz/data/imagenet/", split='train', transform=transform)
        # train_dataset = datasets.ImageFolder(root='/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/CLS-LOC/train', transform=transform)
        # sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset,  num_replicas=num_tasks, rank=global_rank)      
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)

        dataset_val = ImageNet1K(root="/mnt/parscratch/users/coq20tz/data/imagenet/", split='val', transform=transform)
        # test_dataset = datasets.ImageFolder(root='/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform)
        # sampler_val = torch.utils.data.distributed.DistributedSampler(
        #     test_dataset, shuffle=False)
        test_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)
        num_classes = 1000
    elif args.dataset == 'IMAGENET100':
        id_train_loader, ood_train_loader, val_loader = create_imagenet_dataloaders(batch_size = batch_size, root="/mnt/parscratch/users/coq20tz/data/imagenet/")
        train_loader = id_train_loader
        test_loader = val_loader
        num_classes = 100
    return train_loader, test_loader, num_classes
    
class ImageNet1K(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(ImageNet1K, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.split = split
        self.images = []
        self.labels = []
        
        # Load label from CSV for val and test
        if self.split in ['val', 'test']:
            csv_file = os.path.join(root, f"LOC_{self.split}_solution.csv")
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    image_id, prediction_string = row
                    self.images.append(os.path.join(root, "ILSVRC", "Data", "CLS-LOC", self.split, f"{image_id}.JPEG"))
                    label, _ = prediction_string.split(' ', 1)
                    self.labels.append(label)
        
        # Load image paths and labels for train
        if self.split == 'train':
            for synset in os.listdir(os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "train")):
                for image_name in os.listdir(os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "train", synset)):
                    self.images.append(os.path.join(root, "ILSVRC", "Data", "CLS-LOC", "train", synset, image_name))
                    self.labels.append(synset)
        
        # Create a mapping from synset IDs to integer labels
        self.synset_to_int = {synset: i for i, synset in enumerate(sorted(set(self.labels)))}
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        
        label = self.synset_to_int[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_imagenet_dataloaders(batch_size=256, root="/mnt/parscratch/users/coq20tz/data/imagenet/"):
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    # Define transform for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    
    # Load full training dataset to get all unique classes
    full_train_dataset = ImageNet1K(root=root, split='train', transform=transform)
    all_classes = list(full_train_dataset.synset_to_int.keys())
    
    saved_classes_file = 'sampled_classes.txt'
    
    # Check if classes were saved previously
    if os.path.exists(saved_classes_file):
        with open(saved_classes_file, 'r') as f:
            lines = f.readlines()
            id_classes = [line.strip() for line in lines if "ID" not in line and line.strip() != ""][:100]
            ood_classes = [line.strip() for line in lines if "OOD" in line][1:101]
    else:
        # Randomly sample 100 classes for ID and 100 different classes for OOD
        np.random.shuffle(all_classes)
        id_classes = all_classes[:100]
        ood_classes = all_classes[100:200]
        
        # Store the sampled classes
        with open(saved_classes_file, 'w') as f:
            f.write("ID Classes:\n")
            for cls in id_classes:
                f.write(cls + '\n')
            f.write("\nOOD Classes:\n")
            for cls in ood_classes:
                f.write(cls + '\n')
    
    # Create dataloaders based on the sampled classes
    id_indices = [i for i, label in enumerate(full_train_dataset.labels) if label in id_classes]
    ood_indices = [i for i, label in enumerate(full_train_dataset.labels) if label in ood_classes]
    
    id_train_sampler = torch.utils.data.SubsetRandomSampler(id_indices)
    ood_train_sampler = torch.utils.data.SubsetRandomSampler(ood_indices)
    
    id_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=id_train_sampler, pin_memory=True)
    ood_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, sampler=ood_train_sampler, pin_memory=True)
    
    # Load validation dataset and its loader (can be used for both ID and OOD evaluation)
    val_dataset = ImageNet1K(root=root, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print("Sampled ID Classes:", id_classes)
    print("Sampled OOD Classes:", ood_classes)
    
    return id_train_loader, ood_train_loader, val_loader


# Function to load checkpoint
def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'", flush=True)
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})", flush=True)
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'. Starting from scratch.")
        return 0

# Function to save checkpoint
def save_checkpoint(state, filename):
    torch.save(state, filename)

def save_scores_to_csv(in_scores, out_scores, save_path):
    with open(save_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['In-Score', 'Out-Score'])
        for in_s, out_s in zip(in_scores, out_scores):
            writer.writerow([in_s, out_s])
            


def apply_react(model, dataloader_train, device, react_percentile=0.95):
    
    model.eval()
    model = model.to(device)
    
    feas = [[]] * len(dataloader_train)
    for i, labeled_data in tqdm(enumerate(dataloader_train), desc=f"{apply_react.__name__}"):
        _x = labeled_data[0].to(device)

        with torch.no_grad():
            _feas, _ = model(_x)

        feas[i] = _feas.cpu()

    feas = torch.cat(feas, dim=0).numpy()
    c = np.quantile(feas, react_percentile)
    print(f"{((feas < c).mean()*100).round(2)}% of the units of train features are less than {c}")

    print(f"ReAct c = {c}")
    model.encoder = torch.nn.Sequential(model.encoder, ReAct(c))

    return model

class ReAct(torch.nn.Module):
    def __init__(self, c=1.0):
        super(ReAct, self).__init__()
        self.c = c

    def forward(self, x):
        return x.clip(max=self.c)
