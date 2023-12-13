import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utils import get_transformer_config, get_transformer_model, get_openai_lr, load_prior_model, get_cosine_schedule_with_warmup, DualResolutionTransform, DualResolutionDataset # Import your ViT model

def find_max_batch_size(model, dataset):
    max_batch_size = 1
    while True:
        print(f'Testing batch size: {max_batch_size}')
        dataloader = DataLoader(dataset, batch_size=max_batch_size, shuffle=True)
        data, targets = next(iter(dataloader))
        data, targets = data.cuda(), targets.cuda()
        try:
            with torch.no_grad():
                prior = torch.randn(max_batch_size, 10).cuda()
                outputs = model(data, prior)
            print(f'Batch size {max_batch_size} fits in memory')
            max_batch_size *= 2  # Double the batch size for the next iteration
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f'Maximum batch size: {max_batch_size // 2}')
                break
            else:
                raise e  # Re-raise any other exception

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    model_spec = get_transformer_config('huge')
    num_classes = 10
    model = get_transformer_model(model_spec, num_classes)
    model.cuda()  # Assume your model is defined in the YourViTModel class
    find_max_batch_size(model, dataset)
