import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from model_engines.interface import ModelEngine
from model_engines.assets import extract_features
import timm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader

class ViTCifar100ModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ViTCifar100()
        
        self._model.to(self._device)
        self._model.eval()
        self._data_transform = DATA_TRANSFORM
        
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
        pass
    
    def get_model_outputs(self):
        model_outputs = {}
        for fold in self._folds:
            model_outputs[fold] = {}
            
            _dataloader = self._dataloaders[fold]
            _tensor_dict = extract_features(self._model, _dataloader, self._device)
            
            model_outputs[fold]["feas"] = _tensor_dict["feas"]
            model_outputs[fold]["logits"] = _tensor_dict["logits"]
            model_outputs[fold]["labels"] = _tensor_dict["labels"]
        
        return model_outputs['train'], model_outputs['id'], model_outputs['ood']


class ViTCifar100(nn.Module):
    def __init__(self):
        super(ViTCifar100, self).__init__()

        # Load the pre-trained ViT model
        self.encoder = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")

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


import torchvision.transforms as transforms
DATA_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])