import numpy as np
import os
import argparse
import time
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from utils import get_transformer_config, get_transformer_model, get_openai_lr, load_prior_model, get_batch_size, save_checkpoint, load_checkpoint, get_linear_schedule_with_warmup
from dataloader import build_dataset

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
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--weight_decay', default=0.001, type=int)
parser.add_argument('--momentum', default=0.9, type=int)
parser.add_argument('--dropout', default=0.1, type=int)
parser.add_argument('--alpha_weight', default=0.01, type=float)
parser.add_argument('--prior_token_position', default='end', type=str)
parser.add_argument('--token', default=True, type=bool)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--draw', default= False, type=bool)
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
if torch.cuda.device_count() > 1:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

batch_size = get_batch_size(args, torch.cuda.device_count()) if args.batch_size == -1 else args.batch_size
print(f"Batch size: {batch_size}", flush=True)
num_epochs = args.num_epochs

train_loader, test_loader, num_classes = build_dataset(args, batch_size)

# Load the pre-trained model
prior_model = load_prior_model(args, device=device, num_outputs=num_classes)

model_spec = get_transformer_config(args.model_size, dropout=args.dropout, emb_dropout=args.dropout)

# Define the Vision Transformer model
model = get_transformer_model(args, model_spec, num_classes)

# Define the common path prefix
pathname = f'{args.prior_token_position}_{args.token}_{args.dataset}_{args.alpha_weight}_{args.model_size}_{args.prior_model_name}'


if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!", flush=True)
    model = nn.DataParallel(model)
    
torch.cuda.manual_seed(args.seed)
model = model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()

learning_rate = get_openai_lr(model)

# learning_rate = 0.001
print(f"Initial Learning rate: {learning_rate}", flush=True)
optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
# Define the scheduler
# scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.num_epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_epochs, args.num_epochs)
total_batches = len(train_loader)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
        
# Train the model
def train(epoch):
    start_time = time.time()  
    running_loss = 0.0
    correct = 0
    total = 0
    if any(keyword in args.prior_model_name for keyword in ["vit", "imagenet", "BEiT"]):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            priors = prior_model(images)                     
            # Forward pass
            if args.prior_model_name == 'resnet18_imagenet200':
                outputs = model(images, priors)
            else:
                logits = priors.logits             
                outputs = model(images, logits)
            cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]           
            cls_outputs = cls_outputs.squeeze(1)
            loss = criterion(cls_outputs, labels)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            # average_gradients(model) 
            optimizer.step()           
            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(cls_outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)
    else:
        for i, (images_32, images_224, labels) in enumerate(train_loader):
            images_32 = images_32.to(device)
            images_224 = images_224.to(device)
            labels = labels.to(device)

            priors = prior_model(images_32)           
            # Forward pass
            outputs = model(images_224, priors)
            cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]            
            cls_outputs = cls_outputs.squeeze(1)
            loss = criterion(cls_outputs, labels)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(cls_outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)
    
    # Step the scheduler at the end of each epoch
    scheduler.step()
    
     # Print epoch-level information
    state['epoch_loss'] = running_loss / total_batches
    epoch_accuracy = 100.0 * correct / total
    elapsed_time = time.time() - start_time

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {state['epoch_loss']:.4f}, Accuracy: {epoch_accuracy:.2f}%, Time: {elapsed_time:.2f}s", flush=True)

# Test the model
def test():
    model.eval()
    with torch.no_grad():
        correct_prior = 0
        correct = 0
        total = 0       
        if any(keyword in args.prior_model_name for keyword in ["vit", "imagenet", "BEiT"]):
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                priors = prior_model(images)
                if args.prior_model_name == 'resnet18_imagenet200':
                    outputs = model(images, priors)
                    _, prior_predicted = torch.max(priors.data, 1)
                else:
                    logits = priors.logits             
                    outputs = model(images, logits)
                    prior_outputs = logits.squeeze(1)
                    _, prior_predicted = torch.max(prior_outputs.data, 1)
                cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
                cls_outputs = cls_outputs.squeeze(1)
                _, predicted = torch.max(cls_outputs.data, 1)
                
                total += labels.size(0)
                correct_prior += (prior_predicted == labels).sum().item()
                correct += (predicted == labels).sum().item()                           
        else:
            for images_32, images_224, labels in test_loader:
                images_32 = images_32.to(device)
                images_224 = images_224.to(device)
                labels = labels.to(device)

                priors = prior_model(images_32)
                outputs = model(images_224, priors)
                cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
                
                prior_outputs = priors.squeeze(1)
                cls_outputs = cls_outputs.squeeze(1)
                
                _, prior_predicted = torch.max(prior_outputs.data, 1)
                _, predicted = torch.max(cls_outputs.data, 1)
                
                total += labels.size(0)
                correct_prior += (prior_predicted == labels).sum().item()
                correct += (predicted == labels).sum().item()
            
        state['test_accuracy'] = correct / total
        state['prior_test_accuracy'] = correct_prior / total
    print(f"Accuracy of the model on the test images: {100 * correct / total} %", flush=True)
    print(f"Accuracy of the prior model on the test images: {100 * correct_prior / total} %", flush=True)
    
# Make save directory
if not os.path.exists(args.model_save_path):
     os.makedirs(args.model_save_path)
if not os.path.isdir(args.model_save_path):
     raise Exception('%s is not a dir' % args.model_save_path)
 
 # Writing to the training results CSV file
results_csv_path = os.path.join(args.model_save_path, f'{pathname}_{args.num_epochs}_training_results.csv')

with open(results_csv_path, 'w') as f:
    f.write("epoch,time(s),train_loss,test_error(%),prior_test_error(%) \n")
 
print('Beginning Training\n')
# Main loop
# Load checkpoint if one exists
# Check for the latest checkpoint and load it

# Find the latest checkpoint and determine the start epoch
checkpoint_pattern = f'./checkpoints/{pathname}_checkpoint_*.pth.tar'
list_of_files = glob.glob(checkpoint_pattern)
latest_checkpoint = max(list_of_files, key=os.path.getctime) if list_of_files else None
start_epoch = load_checkpoint(latest_checkpoint, model, optimizer) if latest_checkpoint else 0


previous_checkpoint = None  # Initialize this outside of the epoch loop

for epoch in range(num_epochs):   
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning rate: {current_lr}", flush=True)
    begin_epoch = time.time()
    train(epoch)
    test()
    
    # Path for the current checkpoint
    current_checkpoint = f'./checkpoints/{pathname}_checkpoint_{epoch + 1}.pth.tar'
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=current_checkpoint)
    
    # If a previous checkpoint exists, delete it
    if previous_checkpoint and os.path.exists(previous_checkpoint):
        os.remove(previous_checkpoint)
    
    # Update the previous checkpoint name
    previous_checkpoint = current_checkpoint
    
    if (epoch + 1) % 5 == 0:
        model_save_path = os.path.join(args.model_save_path, f'{pathname}_{epoch}.pt')
        torch.save(model.state_dict(), model_save_path)
    
   # Delete the model snapshot from 10 epochs ago to save space
    if epoch >= 10:
        old_model_path = os.path.join(args.model_save_path, f'{pathname}_{epoch - 5}.pt')
        if os.path.exists(old_model_path):
            os.remove(old_model_path)
    
    # Use it for opening the file and appending results
    with open(results_csv_path, 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.2f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['epoch_loss'],
            100 - 100. * state['test_accuracy'],
            100 - 100. * state['prior_test_accuracy'],
        ))
    
    

    