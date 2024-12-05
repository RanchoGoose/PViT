from typing import Dict
import torch
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import ViT_B_16_Weights

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
        
from model_engines.interface import ModelEngine
from model_engines.assets import extract_features_pvit, extract_features_pvit_ablation
from utils import get_transformer_config, get_transformer_model, load_model
from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class PViTModelEngine(ModelEngine):
    def set_model(self, args, prior_model):
        super().set_model(args)
        self._args = args
        model_spec = get_transformer_config(model_size='small')
        num_classes = 100 if args.id_data_name == 'cifar100' else 1000
        model = get_transformer_model(args, model_spec, num_classes=num_classes)
        if self._args.run_ablation:
            self._model = ViT_Imagenet()
        else:
            self._model = model
            if not args.train:
                self._model = load_model(args, self._model)
        
        if args.prior_model == "vit-b16-swag-e2e-v1":
            self._data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif args.prior_model == "resnet18_cifar100":
            self._data_transform = DATA_TRANSFORM_32
        else:           
            self._data_transform = DATA_TRANSFORM

        self._react_percentile = 0.9

        self._model.to(self._device)
        self._model.eval()
        self.prior_model = prior_model
    
    def get_data_transform(self):
        return self._data_transform
    
    def set_dataloaders(self):
        self._dataloaders = {}
        self._dataloaders['train'] = get_train_dataloader(self._data_root_path, 
                                                         self._train_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)

        self._dataloaders['id'] = get_id_dataloader(self._data_root_path, 
                                                         self._id_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)
        self._dataloaders['ood'] = get_ood_dataloader(self._data_root_path, 
                                                         self._ood_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)
    

    def train_model(self):
        # Set training parameters
        num_epochs = self._args.num_epochs if hasattr(self._args, 'num_epochs') else 20
        learning_rate = self._args.learning_rate if hasattr(self._args, 'learning_rate') else 0.001
        weight_decay = self._args.weight_decay if hasattr(self._args, 'weight_decay') else 0.001

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        train_loader = self._dataloaders['train']
        total_batches = len(train_loader)

        # Variables for managing checkpoints and results
        pathname = f'pvit_{self._args.id_data_name}_{self._args.alpha_weight}_{self._args.prior_model}'
        results_csv_path = os.path.join(self._args.model_save_path, f'{pathname}_{num_epochs}_training_results.csv')

        # Ensure model save path exists
        os.makedirs(self._args.model_save_path, exist_ok=True)

        # Initialize state dictionary
        state = {}

        # Write headers to the results CSV file
        with open(results_csv_path, 'w') as f:
            f.write("epoch,time(s),train_loss,test_error(%),prior_test_error(%)\n")

        print('Beginning Training\n')

        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}", flush=True)
            begin_epoch = time.time()

            self._model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, data in enumerate(train_loader):
                # Handle different data loader output formats
                if isinstance(data, (tuple, list)):
                    if len(data) == 2:
                        images, labels = data
                    else:
                        images = data[0]
                        labels = data[1]
                else:
                    images = data
                    labels = None

                images = images.to(self._device)
                if labels is not None:
                    labels = labels.to(self._device)

                # Get priors from the prior model
                with torch.no_grad():
                    try:
                        prior_feas, priors = self.prior_model(images)
                    except ValueError:
                        priors = self.prior_model(images)

                    if hasattr(priors, 'logits'):
                        priors = priors.logits
                    priors = priors if not isinstance(priors, tuple) else priors[0]

                # Forward pass through the model with images and priors
                prior_feats, x, outputs = self._model(images, priors)

                # Compute loss
                cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
                cls_outputs = cls_outputs.squeeze(1)
                loss = criterion(cls_outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(cls_outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (i + 1) % 100 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}", flush=True)

            # Update the scheduler
            scheduler.step()

            # Calculate epoch statistics
            state['epoch_loss'] = running_loss / total
            epoch_accuracy = 100.0 * correct / total
            elapsed_time = time.time() - begin_epoch

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {state['epoch_loss']:.4f}, "
                  f"Accuracy: {epoch_accuracy:.2f}%, Time: {elapsed_time:.2f}s", flush=True)

            # Test the model and update state with test accuracies
            test_state = self.test_model()
            state.update(test_state)

            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                model_save_path = os.path.join(self._args.model_save_path, f'{pathname}_{epoch}.pt')
                torch.save(self._model.state_dict(), model_save_path)

            # Delete the model snapshot from 10 epochs ago to save space
            if epoch >= 10:
                old_model_path = os.path.join(self._args.model_save_path, f'{pathname}_{epoch - 5}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            # Append results to the CSV file
            with open(results_csv_path, 'a') as f:
                f.write('%03d,%05d,%0.6f,%0.2f,%0.2f\n' % (
                    (epoch + 1),
                    int(time.time() - begin_epoch),
                    state['epoch_loss'],
                    100 - 100. * state['test_accuracy'],
                    100 - 100. * state['prior_test_accuracy'],
                ))

    def test_model(self):
        self._model.eval()
        correct_prior = 0
        correct = 0
        total = 0
        device = self._device

        # Use the 'id' dataloader for testing
        test_loader = self._dataloaders['id']

        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                # Handle different data loader output formats
                if isinstance(data, (tuple, list)):
                    if len(data) == 2:
                        images, labels = data
                    else:
                        images = data[0]
                        labels = data[1]
                else:
                    images = data
                    labels = None

                images = images.to(device)
                if labels is not None:
                    labels = labels.to(device)

                # Obtain priors from the prior model
                try:
                    prior_feas, priors = self.prior_model(images)
                except ValueError:
                    priors = self.prior_model(images)

                if hasattr(priors, 'logits'):
                    priors = priors.logits
                priors = priors if not isinstance(priors, tuple) else priors[0]

                # Forward pass through the main model
                prior_fes, x, outputs = self._model(images, priors)

                # Prior model predictions
                prior_outputs = torch.softmax(priors, dim=1)
                prior_predicted = torch.argmax(prior_outputs, dim=-1)

                # Main model predictions
                cls_outputs = outputs if not isinstance(outputs, tuple) else outputs[0]
                cls_outputs = cls_outputs.squeeze(1)
                _, predicted = torch.max(cls_outputs, 1)

                total += labels.size(0)
                correct_prior += (prior_predicted == labels).sum().item()
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        prior_test_accuracy = correct_prior / total

        print(f"Accuracy of the model on the test images: {100 * test_accuracy:.2f}%", flush=True)
        print(f"Accuracy of the prior model on the test images: {100 * prior_test_accuracy:.2f}%", flush=True)

        # Return test accuracies
        return {
            'test_accuracy': test_accuracy,
            'prior_test_accuracy': prior_test_accuracy
        }

    def apply_react(self):
        self._model = apply_react(self.prior_model, self._model, self._dataloaders['train'], self._device, 
                                  self._react_percentile)
    
    def get_model_outputs(self):
        model_outputs = {}
        for fold in self._folds:
            model_outputs[fold] = {}
            
            _dataloader = self._dataloaders[fold]
            
            if self._args.run_ablation:
                _tensor_dict = extract_features_pvit_ablation(self.prior_model, self._model, _dataloader, self._device)
            else:
                _tensor_dict = extract_features_pvit(self._model, self._model, _dataloader, self._device)

            model_outputs[fold]["feas"] = _tensor_dict["feas"]
            model_outputs[fold]["logits"] = _tensor_dict["logits"]
            model_outputs[fold]["labels"] = _tensor_dict["labels"]
            model_outputs[fold]["priors"] = _tensor_dict["priors"]
        
        return model_outputs['train'], model_outputs['id'], model_outputs['ood']

import numpy as np
from tqdm import tqdm
import os
def apply_react(prior_model, model, dataloader_train, device, react_percentile=0.95):
    
    model.eval()
    model = model.to(device)
    
    feas = [[]] * len(dataloader_train)
    for i, labeled_data in tqdm(enumerate(dataloader_train), desc=f"{apply_react.__name__}"):
        _x = labeled_data[0].to(device)

        with torch.no_grad():
            _prior_feas, priors = prior_model(_x)
            _feas, _ = model(_x, priors)

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

import torchvision.transforms as transforms
DATA_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

DATA_TRANSFORM_32 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

class ViT_Imagenet(nn.Module):
    def __init__(self):
        super(ViT_Imagenet, self).__init__()

        # Load the pre-trained ViT model
        self.encoder = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

        # Initialize the layer normalization using the model's configuration
        self.ln = nn.LayerNorm(self.encoder.config.hidden_size, eps=self.encoder.config.layer_norm_eps)
        
        # Access the classifier head
        self.fc = self.encoder.classifier

        # Replace these with identity for feature extraction
        self.encoder.classifier = nn.Identity()
        
    def forward(self, x):
        # Get the encoder outputs with hidden states
        outputs = self.encoder(x, output_hidden_states=True)
        
        # Extract the last hidden state as features
        last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Select the [CLS] token for classification
        cls_token_output = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]

        # Apply layer normalization
        x = self.ln(cls_token_output)

        # Compute logits using the classification head
        logits = self.fc(x)

        return x, logits