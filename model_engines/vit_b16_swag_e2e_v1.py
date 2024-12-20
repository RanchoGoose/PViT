import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader

class ViTModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ViT(args.model_name)
        if args.model_name == 'vit-b-16':
            self._data_transform = ViT_B_16_Weights.DEFAULT.transforms()
        elif args.model_name == 'vit-b16-swag-e2e-v1':
            self._data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif args.model_name == 'vit-lp':
            self._data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()   
        self._model.to(self._device)
        self._model.eval()
    
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

class ViT(nn.Module):
    def __init__(self, model_name='vit-b16-swag-e2e-v1'):
        super(ViT, self).__init__()

        assert model_name in ['vit-b-16', 'vit-b16-swag-e2e-v1', 'vit-lp']

        # define network IMAGENET1K_SWAG_E2E_V1
        if model_name == 'vit-b-16':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif model_name == 'vit-b16-swag-e2e-v1':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        elif model_name == 'vit-lp':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self.ln = self.encoder.encoder.ln
        self.fc = self.encoder.heads.head
        self.encoder.encoder.ln = nn.Identity()
        self.encoder.heads = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        feas = x
        x = self.ln(x)

        # if self.post_ln:
        #     feas = x

        logits = self.fc(x)

        return feas, logits

