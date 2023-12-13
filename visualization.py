import numpy as np
import sys
import os

import pickle
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as trn
import matplotlib.pyplot as plt
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from utils import get_transformer_config, get_transformer_model, get_openai_lr, load_prior_model, get_cosine_schedule_with_warmup, DualResolutionTransform, DualResolutionDataset, get_batch_size, load_model, TestTransform, get_transformer_model_temp, build_dataset, save_scores_to_csv
# from visualization import plot_histogram, plot_cpvr
import ood_utils.score_calculation as lib
from ood_utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
import ood_utils.svhn_loader as svhn
import ood_utils.lsun_loader as lsun_loader
    
parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_size', default='small', type=str)
parser.add_argument('--prior_model_name', default='resnet18', type=str)
parser.add_argument('--load_prior_path', '-l', type=str, default='./priors_model/', help='Prior model path.')
parser.add_argument('--model_save_path', default='./snapshots', type=str)
parser.add_argument('--batch_size', default=-1, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--warmup_epochs', default=10, type=int)
parser.add_argument('--seed', default=777, type=int)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--score', default='energy', type=str, help='score options: MSP|energy')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--alpha_weight', default=0.1, type=float)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
batch_size = get_batch_size(args, torch.cuda.device_count()) if args.batch_size == -1 else args.batch_size

num_epochs = args.num_epochs
num_classes = args.num_classes
# Load the pre-trained model
prior_model = load_prior_model(args, device=device, num_outputs=num_classes)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

transform_32 = DualResolutionTransform(mean, std)
transform_224 = trn.Compose([
            trn.Resize((224, 224)),  # Resize images to fit the input size of ViT
            trn.ToTensor(),
            trn.Normalize(mean, std)
        ])

transform_dual = TestTransform(mean, std)
cifar_test_dataset_dual = datasets.CIFAR10(root='./data', train=False, transform=transform_dual, download=False)
test_dataset = DualResolutionDataset(cifar_test_dataset_dual)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
dataset_test = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader_32 = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

train_loader, test_loader, num_classes = build_dataset(args, batch_size)
# # Use the custom dataset
# train_dataset = Prior_Dataloader(model=prior_model, root='./data', train=True, transform=transform, download=False)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_spec = get_transformer_config(args.model_size)

# Define the Vision Transformer model
# model = get_transformer_model_temp(model_spec, num_classes)
model = get_transformer_model(args, model_spec, num_classes)
model = load_model(args, model)
model = model.to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!", flush=True)
    model = nn.DataParallel(model)
    
model.eval()

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_dataset) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_dataset))

concat = lambda x: np.concatenate(x, axis=0)
# to_np = lambda x: x.data.cpu().numpy()

def to_np(x):
    if isinstance(x, float):
        return np.array([x])
    else:
        return x.data.cpu().numpy()
    
def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= ood_num_examples // batch_size and not in_dist:
                break
            
            if any(keyword in args.prior_model_name for keyword in ["vit", "imagenet", "BEiT"]):
                images, target = batch
                images = images.cuda()
                priors = prior_model(images).logits
                output = model(images, priors)
            else:
                images_32, images_224, target = batch
                images_32, images_224 = images_32.cuda(), images_224.cuda()
                output = 0
                priors = prior_model(images_32)
                output = model(images_224, priors) 
                 
            smax = to_np(F.softmax(output, dim=1))
            smax_priors = to_np(F.softmax(priors, dim=1))
            
            if args.use_xent:
                _score.append(to_np((priors.mean(1) - torch.logsumexp(priors, dim=1))))
            elif args.score == 'energy':
                _score.append(-to_np((args.T*torch.logsumexp(priors / args.T, dim=1))))
            elif args.score == 'double_energy':
                prior_energy = abs(to_np((args.T*torch.logsumexp(priors / args.T, dim=1))))
                true_energy = abs(to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
                _score.append(abs(prior_energy-true_energy))                    
            elif args.score == 'cosine':
                _score.append(-F.cosine_similarity(priors, output).cpu().numpy())
            elif args.score == 'cross_entropy':
                    # Cross-Entropy
                p = F.softmax(priors, dim=1)
                cross_entropy_score = F.cross_entropy(output, p.argmax(dim=1), reduction='none').cpu().numpy()
                _score.append(cross_entropy_score)
            elif args.score == 'Euclidean_distance':
                _score.append(to_np(torch.norm(priors - output, dim=1)))
            elif args.score == 'KL':
                p = F.softmax(priors, dim=1)
                q = F.softmax(output, dim=1)
                kl_values = F.kl_div(torch.log(q), p, reduction='none').sum(dim=1)  # Sum over classes for each image
                _score.extend(to_np(kl_values))
            elif args.score == 'difference':    
                p = F.softmax(priors, dim=1)
                diff_classes_score = (p.argmax(dim=1) != output.argmax(dim=1)).float().cpu().numpy()
                _score.append(diff_classes_score)
            else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                _score.append(-np.max(smax_priors, axis=1))
                    
        if in_dist:
            if args.score == 'cross_entropy':                        
                preds = np.argmax(smax, axis=1)
            else:
                preds = np.argmax(smax_priors, axis=1)
            # preds_prior = np.argmax(smax_priors, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            # right_indices_prior = preds_prior == targets
            wrong_indices = np.invert(right_indices)
            # wrong_indices_prior = np.invert(right_indices_prior)

            if args.use_xent:
                _right_score.append(to_np((priors.mean(1) - torch.logsumexp(priors, dim=1)))[right_indices])
                _wrong_score.append(to_np((priors.mean(1) - torch.logsumexp(priors, dim=1)))[wrong_indices])
            elif args.score == 'cosine':
                cosine_scores = F.cosine_similarity(priors, output).cpu().numpy()  # This will be an array with a score for each sample in the batch
                _right_score.append(-cosine_scores[right_indices])
                _wrong_score.append(-cosine_scores[wrong_indices])
            # elif args.score == 'cross_entropy':
            #     # p = F.softmax(priors, dim=1)
            #     # cross_entropy_score = F.cross_entropy(output, p.argmax(dim=1), reduction='none').cpu().numpy()
            #     # Ensure the indexed result is always a 1D array
            #     _right_score.append(-np.max(smax[right_indices], axis=1))
            #     _wrong_score.append(-np.max(smax[wrong_indices], axis=1)) 
            else:
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))   
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()
    
if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(args, test_loader_32, net=prior_model, bs=batch_size, ood_num_examples=ood_num_examples, T=args.T, noise=args.noise, in_dist=True)
elif args.score == 'M':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    if 'CIFAR10' in args.dataset:
        train_data = datasets.CIFAR10('./data', train=True, transform=transform)
    else:
        train_data = datasets.CIFAR100('./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, 
                                          num_workers=args.num_workers, pin_memory=True)
    num_batches = ood_num_examples // batch_size

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = prior_model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    sample_mean, precision = lib.sample_estimator(prior_model, num_classes, feature_list, train_loader) 
    in_score = lib.get_Mahalanobis_score(prior_model, test_loader_32, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True)
    print(in_score[-3:], in_score[-103:-100])
else:
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.prior_model_name)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, prior_model, batch_size, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score = lib.get_Mahalanobis_score(prior_model, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)
        else:
            out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
            csv_path = f'./results/{dataset_name}_{args.dataset}_{args.alpha_weight}_{args.model_size}_{args.prior_model_name}_{args.score}_{args.num_epochs}_scores.csv'
            save_scores_to_csv(-in_score, -out_score, save_path=csv_path)
            # plot_cpvr(-in_score, -out_score, save_path=f'./results/{dataset_name}_{args.dataset}_{args.alpha_weight}_{args.model_size}_{args.prior_model_name}_{(args.num_epochs)}_histogram.png')
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:6], out_score[:6])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.prior_model_name + args.score)
    else:
        print_measures(auroc, aupr, fpr, args.prior_model_name + args.score)

def visualize_samples(id_loader, ood_loader, prior_model, model, args, ood_num_examples=100, batch_size=10, save_path='results/visualization.png'):
    fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))


    score = get_ood_scores(ood_loader, in_dist=False)

    # Get CIFAR10 samples
    id_images, id_labels, _, _ = get_data(id_loader)
    id_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Get OOD samples
    ood_images, _, _, _ = get_data(ood_loader, in_dist=False)

    # Display ID samples
    for i in range(batch_size):
        axes[0, i].imshow(np.transpose(id_images[i].numpy(), (1, 2, 0)))
        axes[0, i].set_title(id_classes[id_labels[i]])
        axes[0, i].axis('off')
    
    # Display OOD samples
    for i in range(batch_size):
        axes[1, i].imshow(np.transpose(ood_images[i].numpy(), (1, 2, 0)))
        axes[1, i].set_title("OOD")
        axes[1, i].axis('off')
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path)
    
# Define datasets and paths
datasets_info = {
    'Texture': '/mnt/parscratch/users/coq20tz/data/dtd/images',
    # 'SVHN': '/mnt/parscratch/users/coq20tz/data/SVHN/test/',
    # 'Places365': '/mnt/parscratch/users/coq20tz/data/places365/',
    # 'LSUN_C': '/mnt/parscratch/users/coq20tz/data/LSUN/lsun-master',
    # 'LSUN_Resize': '/mnt/parscratch/users/coq20tz/data/LSUN/LSUN-R/LSUN_resize',
    # 'iSUN': '/mnt/parscratch/users/coq20tz/data/ISUN/',
}

# Determine if using dual transform or single transform
use_dual = not any(keyword in args.prior_model_name for keyword in ["vit", "imagenet", "BEiT"])
transform_to_use = transform_32 if use_dual else transform_224

# Loop over datasets
for dataset_name, dataset_path in datasets_info.items():
    # Choose the appropriate dataset loading method
    if dataset_name == "SVHN":
        ood_data_base = svhn.SVHN(root=dataset_path, split="test", transform=transform_to_use)
    else:
        ood_data_base = datasets.ImageFolder(root=dataset_path, transform=transform_to_use)

    # Apply DualResolution if necessary
    ood_data = DualResolutionDataset(ood_data_base) if use_dual else ood_data_base
    
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    visualize_samples(test_loader, ood_loader, num_samples=5)
    # print(f'\n\n{dataset_name} Detection')
    # get_and_print_results(ood_loader)









