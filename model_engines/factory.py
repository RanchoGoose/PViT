
from model_engines.interface import ModelEngine

def create_model_engine(model_name) -> ModelEngine:
    
    if model_name == "resnet50-react":
        from model_engines.resnet50_react import ResNet50ReActModelEngine as ModelEngine
    elif model_name == "resnet50-supcon":
        from model_engines.resnet50_supcon import ResNet50SupConModelEngine as ModelEngine
    elif model_name in ['vit-b-16', 'vit-b16-swag-e2e-v1', 'vit-lp']:
        from model_engines.vit_b16_swag_e2e_v1 import ViTModelEngine as ModelEngine
    elif model_name == "regnet-y-16gf-swag-e2e-v1":
        from model_engines.regnet import RegNetModelEngine as ModelEngine
    elif model_name == "mobilenet-v2":
        from model_engines.mobilenet_v2 import MobileNetModelEngine as ModelEngine
    elif model_name == "pvit":
        from model_engines.pvit import PViTModelEngine as ModelEngine
    elif model_name == "swin-t":
        from model_engines.swin_t import SwinTModelEngine as ModelEngine 
    elif model_name == "resnet18_cifar100":
        from model_engines.resnet18_cifar100 import ResNet18ModelEngine as ModelEngine
    elif model_name == "vit_cifar100":
        from model_engines.vit_cifar100 import ViTCifar100ModelEngine as ModelEngine       
    elif model_name == "vit_imagenet":
        from model_engines.vit_imagenet import ViTImagnetModelEngine as ModelEngine   
    elif model_name == "vit_pross":
        from model_engines.vit_pross_supervision import ViTProcessSupervisionModelEngine as ModelEngine        
    else:
        raise NotImplementedError()

    return ModelEngine()