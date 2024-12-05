from torchvision.models.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights, Swin_T_Weights
from torch.hub import load_state_dict_from_url

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader

class SwinTModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        
        self._model = Swin_T()
        weights = eval(f'Swin_T_Weights.IMAGENET1K_V1')
        self._model.load_state_dict(load_state_dict_from_url(weights.url))
        self._data_transform = Swin_T_Weights.IMAGENET1K_V1.transforms()

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

class Swin_T(SwinTransformer):
    def __init__(self,
                 patch_size=[4, 4],
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=[7, 7],
                 stochastic_depth_prob=0.2,
                 num_classes=1000):
        super(Swin_T,
              self).__init__(patch_size=patch_size,
                             embed_dim=embed_dim,
                             depths=depths,
                             num_heads=num_heads,
                             window_size=window_size,
                             stochastic_depth_prob=stochastic_depth_prob,
                             num_classes=num_classes)
        self.feature_size = embed_dim * 2**(len(depths) - 1)

    def forward(self, x, return_feature=False):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        return  x, self.head(x)

        # if return_feature:
        #     return self.head(x), x
        # else:
        #     return self.head(x)

    def forward_threshold(self, x, threshold):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        feature = x.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.head(feature)

        return logits_cls

    def get_fc(self):
        fc = self.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.head
