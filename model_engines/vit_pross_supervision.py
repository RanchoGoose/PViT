import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageClassification
from model_engines.interface import ModelEngine
from model_engines.assets import extract_features_vit_pross, compute_energy_reward, compute_loss, save_checkpoint, load_checkpoint
from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader
from utils import load_model
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

DATA_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

class ViTProcessSupervisionModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._id_data_name = args.id_data_name
        self._num_classes = 1000 if args.id_data_name == "imagenet1k" else 100
        self._epochs = args.epochs if hasattr(args, 'epochs') else 10
        self._num_layers = args.num_layers if hasattr(args, 'num_layers') else 2
        self._learning_rate = args.learning_rate if hasattr(args, 'learning_rate') else 1e-4
        self._weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else 0.001
        
        # Initialize the model
        self._model = ViTWithEnergyProcessSupervision(num_classes=self._num_classes, num_layers=self._num_layers)
        self._model.to(self._device)

        # Define optimizer and criterion
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
        self._criterion = nn.CrossEntropyLoss()
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._epochs)
        self._data_transform = DATA_TRANSFORM
        self._model_save_path = args.model_save_path
        
        if not args.train:
            self._model = load_model(args, self._model)
               
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
        """
        Trains the model using the training dataloader.
        """
        self._model.train()
        pathname = f'step_{self._id_data_name}'
        state = {}
        logging.info("Starting training...")
        
        for epoch in range(self._epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            for batch_idx, (inputs, labels, ood_labels) in enumerate(tqdm(self._dataloaders['train'], desc=f"Epoch {epoch+1}/{self._epochs}")):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                ood_labels = ood_labels.to(self._device)

                self._optimizer.zero_grad()

                # Forward pass
                final_logits, logits_list, _= self._model(inputs)

                # Compute energy scores for each logits
                energy_scores_list = []
                for logits in logits_list:
                    energy_scores = -torch.logsumexp(logits, dim=1)
                    energy_scores_list.append(energy_scores)

                # Compute energy rewards
                energy_rewards = compute_energy_reward(energy_scores_list, ood_labels)

                # Compute loss
                loss = compute_loss(final_logits, labels, energy_rewards, self._criterion)

                # Backward pass and optimization
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    logging.info(f"Batch {batch_idx+1}, Loss: {running_loss / (batch_idx+1):.4f}")

            # Scheduler step
            self._scheduler.step()

            epoch_loss = running_loss / len(self._dataloaders['train'])
            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Epoch [{epoch+1}/{self._epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f}s")

            # Test the model and update state with test accuracies
            test_state = self.test_model()
            state.update(test_state)

            # Save the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                model_save_path = os.path.join(self._model_save_path, f'{pathname}_{epoch}.pt')
                torch.save(self._model.state_dict(), model_save_path)

            # Delete the model snapshot from 10 epochs ago to save space
            if epoch >= 10:
                old_model_path = os.path.join(self._model_save_path, f'{pathname}_{epoch - 5}.pt')
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            # Append results to the CSV file
            results_csv_path = os.path.join(self._model_save_path, 'results.csv')
            with open(results_csv_path, 'a') as f:
                f.write('%03d,%05d,%0.6f,%0.2f,%0.2f\n' % (
                    (epoch + 1),
                    int(time.time() - epoch_start_time),
                    epoch_loss,
                    100. * test_state['test_accuracy'],
                ))
                
        logging.info("Training complete.")
        
    def test_model(self):
        self._model.eval()
        correct = 0
        total = 0
        device = self._device

        # Use the 'id' dataloader for testing
        test_loader = self._dataloaders['id']

        with torch.no_grad():
            for batch_idx, (inputs, labels, ood_labels) in enumerate(tqdm(self._dataloaders['train'], desc=f"Epoch {epoch+1}/{self._epochs}")):
                inputs = inputs.to(self._device)
                labels = labels.to(self._device)
                ood_labels = ood_labels.to(self._device)

                # Forward pass
                final_logits, logits_list, _ = self._model(inputs)

                # Obtain the predicted classes directly from logits
                _, predicted = torch.max(final_logits, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total

        logging.info(f"Accuracy of the model on the ID test images: {100 * test_accuracy:.2f}%")

        # Return test accuracies
        return {
            'test_accuracy': test_accuracy
        }
        
    def get_model_outputs(self):
        # ... (remains unchanged)
        model_outputs = {}
        for fold in self._folds:
            model_outputs[fold] = {}
            
            _dataloader = self._dataloaders[fold]
            _tensor_dict = extract_features_vit_pross(self._model, _dataloader, self._device)
            
            model_outputs[fold]["feas"] = _tensor_dict["feas"]
            model_outputs[fold]["logits"] = _tensor_dict["logits"]
            model_outputs[fold]["labels"] = _tensor_dict["labels"]
        
        return model_outputs['train'], model_outputs['id'], model_outputs['ood']
    
class ViTWithEnergyProcessSupervision(nn.Module):
    # ... (remains unchanged)
    def __init__(self, num_classes, num_layers):
        super(ViTWithEnergyProcessSupervision, self).__init__()
        # Load the pre-trained ViT model
        self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        # Replace the classifier head to match the number of classes
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, num_classes)
        
        self.num_layers = num_layers

    def forward(self, x):
        # Pass the input through the ViT model to get hidden states
        outputs = self.vit_model.vit(
            x,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states  # Tuple of hidden states at each layer
        
        logits_list = []
        # Collect logits from specified layers
        for i in range(self.num_layers):
            # Get the hidden state from layer i+1 (since hidden_states[0] is embeddings)
            cls_output = hidden_states[i+1][:, 0]  # Extract CLS token
            # Compute logits from CLS token
            logits = self.vit_model.classifier(cls_output)
            logits_list.append(logits)
        
        # Get the final output logits
        final_logits = self.vit_model.classifier(outputs.last_hidden_state[:, 0])
        logits_list.append(final_logits)
        
        return final_logits, logits_list, outputs.last_hidden_state[:, 0]