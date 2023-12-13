import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import random
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_transformer_config, get_transformer_model, load_prior_model, get_batch_size, load_model
from torchvision.transforms.functional import to_pil_image
from dataloader import build_dataset, idx_to_label

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_size', default='small', type=str)
parser.add_argument('--prior_model_name', default='resnet18', type=str)
parser.add_argument('--load_prior_path', '-l', type=str, default='./priors_model/', help='Prior model path.')
parser.add_argument('--model_save_path', default='./snapshots', type=str)
parser.add_argument('--batch_size', default=-1, type=int)
parser.add_argument('--num_epochs', default=20, type=int)
parser.add_argument('--warmup_epochs', default=5, type=int)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--weight_decay', default=0.01, type=int)
parser.add_argument('--momentum', default=0.9, type=int)
parser.add_argument('--dropout', default=0.1, type=int)
parser.add_argument('--alpha_weight', default=0.01, type=float)
parser.add_argument('--prior_token_position', default='end', type=str)
parser.add_argument('--token', default= True, type=bool)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--draw', default= True, type=bool)
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(args.seed)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
idx_to_label = idx_to_label(args)
batch_size = get_batch_size(args, torch.cuda.device_count()) if args.batch_size == -1 else args.batch_size
print(f"Batch size: {batch_size}", flush=True)
num_epochs = args.num_epochs

train_loader, test_loader, num_classes = build_dataset(args, batch_size)

if args.dataset == 'imagenet200' or args.dataset == 'imagenet':
    ood_near = {'ssb_hard', 'ninco'}
    ood_far = {'inaturalist', 'textures', 'openimage_o'}
    ood_name = 'openimage_o'
    # ood_loader = get_test_far_ood_loader(args, ood_name=ood_name,batch_size=batch_size)

# Load the pre-trained model
prior_model = load_prior_model(args, device=device, num_outputs=num_classes)

model_spec = get_transformer_config(args.model_size, dropout=args.dropout, emb_dropout=args.dropout)

# Define the Vision Transformer model
model = get_transformer_model(args, model_spec, num_classes)
model = load_model(args, model)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!", flush=True)
    model = nn.DataParallel(model)
    
model = model.to(device)
# Loss and optimizer
    
def visualize_attention_with_prior(image, attention_weights, predicted_label, prior_label, idx_to_label, image_size=224, patch_size=16, save_path=None):
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numeric label index to string name
    prior_label = idx_to_label[prior_label]
    predicted_label = idx_to_label[predicted_label]
    
    # Calculate the number of patches per side
    num_patches_per_side = image_size // patch_size

    # Move the tensor to CPU and convert to NumPy array
    attention_weights = attention_weights.cpu().numpy()

    # Extract attention for class token and prior
    # class_token_attention = attention_weights[0, 1]  # Class token attention
    if args.prior_token_position == 'end':
        prior_token_attention = attention_weights[0, -1]
        image_patch_attention = attention_weights[0, 1:-1]
        # print(len(image_patch_attention), len(image_patch_attention[0]))
    elif args.prior_token_position == 'start':
        prior_token_attention = attention_weights[0, 1]  # Prior token attention
        image_patch_attention = attention_weights[0, 2:]
    elif args.prior_token_position == 'second':
        prior_token_attention = attention_weights[0, 2]
        image_patch_attention = attention_weights[0, 2:]
    elif args.prior_token_position == 'all':
        prior_token_attention = attention_weights[0, -1]
        image_patch_attention = attention_weights[0, 1:-1]
        # print(len(image_patch_attention), len(image_patch_attention[0]))
    else:
        raise ValueError("Invalid prior token position")
        
    # Reshape the attention for the image patches to a square
    reshaped_image_attention = image_patch_attention.reshape(num_patches_per_side, num_patches_per_side)

    # Apply a Gaussian filter to smooth the attention maps
    # smoothed_prior_attention = gaussian_filter(prior_token_attention, sigma=1)
    smoothed_image_attention = gaussian_filter(reshaped_image_attention, sigma=1)

    # # Check the range of the smoothed attention
    attention_min = np.min(smoothed_image_attention)
    attention_max = np.max(smoothed_image_attention)
    print(f"Attention range: {attention_min} to {attention_max}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Only two axes now, one for the image and one for the attention map

    # Assuming image is your tensor and is on CUDA
    image_to_visualize = image.detach()  # Get the image and detach it from the graph

    # Revert the normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image_to_visualize = image_to_visualize * std[:, None, None] + mean[:, None, None]

    # Clamp the values to ensure they are in the [0, 1] range after un-normalization
    image_to_visualize = torch.clamp(image_to_visualize, 0, 1)

    # Move the tensor to CPU before converting it to byte type
    image_to_visualize = image_to_visualize.cpu()

    # Convert to [0, 255] and byte type
    image_to_visualize = (image_to_visualize * 255).byte()

    # Convert to PIL image
    original_img = to_pil_image(image_to_visualize)

    plt.rcParams.update({'font.size': 18})
     # Now you can plot the original image with the labels
    axes[0].imshow(original_img)
    axes[0].set_title(f'{prior_label} | {predicted_label}')
    axes[0].axis('off')    
    
    im = axes[1].imshow(smoothed_image_attention, cmap='viridis', aspect='equal')
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    # attention_min = attention_weights.min()s
    # attention_max = attention_weights.max()

    # Normalize the prior token attention value to the range of the color bar
    # and then clamp it to a certain range to ensure visibility
    norm_prior_attention = (prior_token_attention - attention_min) / (attention_max - attention_min)
    norm_prior_attention = np.clip(norm_prior_attention, 0.02, 0.98)  # Clamp to the range [0.05, 0.95]
    
    # Plot the color bar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical')
    label_str = f'Prior token: {prior_token_attention:.2e}'
    cbar.set_label(label_str)
    
    # If the attention map has variability and you want to use the actual range:
    im.set_clim(attention_min, attention_max)

    # Set the format of the color bar tick labels to plain decimal numbers
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.2e}'))

    # Mark the normalized and clamped prior attention value on the color bar
    # The y-coordinate for axhline is given in the scale of [0, 1], so we use the normalized value
    if prior_token_attention >= attention_max:
        prior_token_attention = attention_max* 0.98
    elif prior_token_attention <= attention_min:
        prior_token_attention = attention_min* 1.02
        
    cbar.ax.axhline(prior_token_attention, color='red', linewidth=4)

    # # Add a label next to the marked line if desired
    # cbar.ax.text(1.05, norm_prior_attention, f'{prior_token_attention:.2e}', color='red', va='center', ha='left', transform=cbar.ax.transAxes)
    
    print(attention_max, attention_min, prior_token_attention)

    # Save the figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

# def visualize_fig(args, image, image_size=224, patch_size=16, save_path=None):
    

model.eval()
with torch.no_grad():
    # for ood eva     
    # test_loader = ood_loader
    
    random_indices = np.random.choice(len(test_loader.dataset), size=5, replace=False)
    # Create a sampler and loader for these indices
    random_sampler = torch.utils.data.SubsetRandomSampler(random_indices)
    random_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=1, sampler=random_sampler)
    
    # count = 0
    if any(keyword in args.prior_model_name for keyword in ["vit", "imagenet", "BEiT"]):
        for images, labels in random_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            priors = prior_model(images)
            logits = priors.logits
            cls_outputs, attention_weights = model(images, logits)  
            
            _, prior_label = torch.max(logits, 1)
            _, predicted = torch.max(cls_outputs, 1)
                  
            attention_map = attention_weights[-1][0][0]  # Last layer, first head
            
            for img, label, pred_label in zip(images, prior_label, predicted):
                prior_label = label.item()  # Or your mapping, e.g., idx_to_class[label.item()]
                predicted_label = pred_label.item()  # Or your mapping, e.g., idx_to_class[pred_label.item()]

                # Visualize the attention along with the labels
                visualize_attention_with_prior(
                    image=img.cpu(),
                    attention_weights=attention_map.cpu(),
                    predicted_label=predicted_label,
                    prior_label=prior_label,
                    idx_to_label=idx_to_label,
                    image_size=224,  # Or the size of your images
                    patch_size=16,  # Or the size of your patches
                    save_path=f'./fig/{args.prior_token_position}_{args.token}_{args.dataset}_{args.alpha_weight}_{args.model_size}_{args.prior_model_name}_attention_map_{label.item()}.png')
                    
            attention_map = []      
             # Break after processing 5 images
            # count += 1
            # if count >= 1:  # Since i starts at 0 for the first image
            #     break                                   
    else:
        for images_32, images_224, labels in random_loader:
            images_32 = images_32.to(device)
            images_224 = images_224.to(device)
            labels = labels.to(device)

            priors = prior_model(images_32)
            cls_outputs, attention_weights = model(images_224, priors)
            
            _, prior_label = torch.max(priors, 1)
            _, predicted = torch.max(cls_outputs, 1)
            
            attention_map = attention_weights[-1][0][0]  # Last layer, first head
            # visualize_attention()
            for img, label, pred_label in zip(images_32, prior_label, predicted):
                actual_label = label.item()  # Or your mapping, e.g., idx_to_class[label.item()]
                predicted_label = pred_label.item()  # Or your mapping, e.g., idx_to_class[pred_label.item()]

                # Visualize the attention along with the labels
                visualize_attention_with_prior(
                    image=img.cpu(),
                    attention_weights=attention_map.cpu(),
                    predicted_label=predicted_label,
                    actual_label=actual_label,
                    idx_to_label=idx_to_label,
                    image_size=224,  # Or the size of your images
                    patch_size=16,  # Or the size of your patches
                    save_path=f'./fig/{args.prior_token_position}_{args.token}_{args.dataset}_{args.alpha_weight}_{args.model_size}_{args.prior_model_name}_attention_map_{label.item()}.png')                 
                
            attention_map = [] 

               


# # Assume `image_patch_attention` contains the attention for each image patch
# # And `prior_token_attention` contains the attention for the prior token
# # We first ensure that the image_patch_attention is a 1D array
# image_patch_attention = image_patch_attention.flatten()

# # Now, we append the prior_token_attention to the end of the image_patch_attention
# combined_attention = np.append(image_patch_attention, prior_token_attention)

# # Normalize the combined attention for better visualization
# combined_attention = combined_attention / np.max(combined_attention)

# # Now create a new figure or use your existing axes to plot
# fig, ax = plt.subplots(figsize=(12, 5))

# # Create a range for the x-axis that corresponds to the number of patches plus one for the prior token
# x_range = np.arange(len(combined_attention))

# # Create the bar plot
# ax.bar(x_range[:-1], combined_attention[:-1], label='Image Patch Attention')
# ax.bar(x_range[-1], combined_attention[-1], color='r', label='Prior Token Attention')

# # Add labels and title
# ax.set_xlabel('Patch Number')
# ax.set_ylabel('Attention Weight')
# ax.set_title('Attention Weights for Image Patches and Prior Token')

# # Add a legend to distinguish between image patches and the prior token
# ax.legend()

# # Show the plot
# plt.show()
