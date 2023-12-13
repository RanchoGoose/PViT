import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_INFO = {
    'cifar10': {
        'num_classes': 10,
        'id': {
            'train': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/train_cifar10.txt'
            },
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/val_cifar10.txt'
            },
            'test': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_cifar10.txt'
            }
        },
        'csid': {
            'datasets': ['cifar10c'],
            'cinic10': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/val_cinic10.txt'
            },
            'cifar10c': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_cifar10c.txt'
            }
        },
        'ood': {
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/data/benchmark_imglist/cifar10/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar100', 'tin'],
                'cifar100': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_cifar100.txt'
                },
                'tin': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_svhn.txt'
                },
                'texture': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_texture.txt'
                },
                'places365': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar10/test_places365.txt'
                },
            }
        }
    },
    'cifar100': {
        'num_classes': 100,
        'id': {
            'train': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/train_cifar100.txt'
            },
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/val_cifar100.txt'
            },
            'test': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_cifar100.txt'
            }
        },
        'csid': {
            'datasets': [],
        },
        'ood': {
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/val_tin.txt'
            },
            'near': {
                'datasets': ['cifar10', 'tin'],
                'cifar10': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_cifar10.txt'
                },
                'tin': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_tin.txt'
                }
            },
            'far': {
                'datasets': ['mnist', 'svhn', 'texture', 'places365'],
                'mnist': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_mnist.txt'
                },
                'svhn': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_svhn.txt'
                },
                'texture': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_texture.txt'
                },
                'places365': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/cifar100/test_places365.txt'
                }
            },
        }
    },
    'imagenet200': {
        'num_classes': 200,
        'id': {
            'train': {
                'data_dir':
                '/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/train_imagenet200.txt'
            },
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/val_imagenet200.txt'
            },
            'test': {
                'data_dir':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_imagenet200.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_imagenet200_v2.txt'
            },
            'imagenet_c': {
                'data_dir':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_imagenet200_c.txt'
            },
            'imagenet_r': {
                'data_dir':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_imagenet200_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet200/test_openimage_o.txt'
                },
            },
        }
    },
    'imagenet': {
        'num_classes': 1000,
        'id': {
            'train': {
                'data_dir': '/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/train_imagenet.txt'
            },
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/val_imagenet.txt'
            },
            'test': {
                'data_dir': '/mnt/parscratch/users/coq20tz/data/imagenet/ILSVRC/Data/',
                'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_imagenet.txt'
            }
        },
        'csid': {
            'datasets': ['imagenet_v2', 'imagenet_c', 'imagenet_r'],
            'imagenet_v2': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_imagenet_v2.txt'
            },
            'imagenet_c': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_imagenet_c.txt'
            },
            'imagenet_r': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_imagenet_r.txt'
            },
        },
        'ood': {
            'val': {
                'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                'imglist_path':
                '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/val_openimage_o.txt'
            },
            'near': {
                'datasets': ['ssb_hard', 'ninco'],
                'ssb_hard': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_ssb_hard.txt'
                },
                'ninco': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_ninco.txt'
                }
            },
            'far': {
                'datasets': ['inaturalist', 'textures', 'openimage_o'],
                'inaturalist': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_inaturalist.txt'
                },
                'textures': {
                    'data_dir': '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_textures.txt'
                },
                'openimage_o': {
                    'data_dir':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/',
                    'imglist_path':
                    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist/imagenet/test_openimage_o.txt'
                },
            },
        }
    },
}

download_id_dict = {
    'cifar10': '1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1',
    'cifar100': '1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_',
    'tin': '1PZ-ixyx52U989IKsMA2OT-24fToTrelC',
    'mnist': '1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb',
    'svhn': '1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI',
    'texture': '1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam',
    'places365': '1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay',
    'imagenet_1k': '1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj',
    'species_sub': '1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq',
    'ssb_hard': '1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE',
    'ninco': '1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6',
    'inaturalist': '1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj',
    'places': '1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3',
    'sun': '1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp',
    'openimage_o': '1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE',
    'imagenet_v2': '1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho',
    'imagenet_r': '1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU',
    'imagenet_c': '1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt',
    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/benchmark_imglist': '1XKzBdWCqg3vPoj-D32YixJyJJ0hL63gP'
}

dir_dict = {
    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_classic/': [
        'cifar100', 'tin', 'tin597', 'svhn', 'cinic10', 'imagenet10', 'mnist',
        'fashionmnist', 'cifar10', 'cifar100c', 'places365', 'cifar10c',
        'fractals_and_fvis', 'usps', 'texture', 'notmnist'
    ],
    '/mnt/parscratch/users/coq20tz/OpenOOD/scripts/download/data/images_largescale/': [
        'imagenet_1k',
        'ssb_hard',
        'ninco',
        'inaturalist',
        'places',
        'sun',
        'openimage_o',
        'imagenet_v2',
        'imagenet_c',
        'imagenet_r',
    ],
    'images_medical/': ['actmed', 'bimcv', 'ct', 'hannover', 'xraybone'],
}

benchmarks_dict = {
    'cifar10':
    ['cifar10', 'cifar100', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'cifar100':
    ['cifar100', 'cifar10', 'tin', 'mnist', 'svhn', 'texture', 'places365'],
    'imagenet200': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
    'imagenet': [
        'imagenet_1k', 'ssb_hard', 'ninco', 'inaturalist', 'texture',
        'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r'
    ],
}

class TestTransform:
    def __init__(self, mean, std):
        self.transform_32 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, image):
        return {
                'image_32': self.transform_32(image),
                'image_224': self.transform_224(image)
                }
        
# Custom transform that returns both 32x32 and 224x224 resolutions
class DualResolutionTransform:
    def __init__(self, mean, std):
        self.transform_32 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    def __call__(self, image):
        return {
                'image_32': self.transform_32(image),
                'image_224': self.transform_224(image)
                }

# Custom dataset wrapper
class DualResolutionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image['image_32'], image['image_224'], label

class CustomDataset(Dataset):
    def __init__(self, data_dir, imglist_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_labels = []
        with open(imglist_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    path, label = parts
                    self.img_labels.append((path, int(label)))
                else:
                    print(f"Line in file {imglist_path} is malformed: {line}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        full_img_path = os.path.join(self.data_dir, img_path)
        image = Image.open(full_img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_transform(is_vit_model, mean, std):
    if is_vit_model:
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return DualResolutionTransform(mean, std)

def get_datasets(info, transform, is_vit_model):
    dataset_train = CustomDataset(
    data_dir=info['train']['data_dir'],
    imglist_path=info['train']['imglist_path'],
    transform=transform)
    dataset_test = CustomDataset(
    data_dir=info['test']['data_dir'],
    imglist_path=info['test']['imglist_path'],
    transform=transform)
    if not is_vit_model:
        dataset_train = DualResolutionDataset(dataset_train)
        dataset_test = DualResolutionDataset(dataset_test)
    return dataset_train, dataset_test

def get_data_loaders(args, dataset_train, dataset_test, batch_size):
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    return train_loader, test_loader

def build_dataset(args, batch_size):
    # mean and standard deviation 
    mean = [x / 255 for x in [125.3, 123.0, 113.9]] if 'cifar' in args.dataset else (0.485, 0.456, 0.406)
    std = [x / 255 for x in [63.0, 62.1, 66.7]] if 'cifar' in args.dataset else (0.229, 0.224, 0.225)

    # IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    # IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    if args.dataset == 'cifar10':
        if any(keyword in args.prior_model_name for keyword in ["vit", "BEiT"]):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            dataset_train = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            transform_dual = DualResolutionTransform(mean, std)
            cifar_train_dataset_dual = datasets.CIFAR10(root='./data', train=True, transform=transform_dual, download=False)
            train_dataset = DualResolutionDataset(cifar_train_dataset_dual)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            cifar_test_dataset_dual = datasets.CIFAR10(root='./data', train=False, transform=transform_dual, download=False)
            test_dataset = DualResolutionDataset(cifar_test_dataset_dual)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        if any(keyword in args.prior_model_name for keyword in ["vit", "BEiT"]):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to fit the input size of ViT
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            dataset_train = datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            transform_dual = DualResolutionTransform(mean, std)
            cifar_train_dataset_dual = datasets.CIFAR100(root='./data', train=True, transform=transform_dual, download=True)
            train_dataset = DualResolutionDataset(cifar_train_dataset_dual)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

            cifar_test_dataset_dual = datasets.CIFAR100(root='./data', train=False, transform=transform_dual, download=True)
            test_dataset = DualResolutionDataset(cifar_test_dataset_dual)
            test_loader = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, pin_memory=True)
        num_classes = 100
    else:
        data_info = DATA_INFO[args.dataset]['id']
        transform = get_transform(is_vit_model, mean, std)
        dataset_train, dataset_test = get_datasets(data_info, transform, is_vit_model)
        train_loader, test_loader = get_data_loaders(args, dataset_train, dataset_test, batch_size)
        num_classes = DATA_INFO[args.dataset]['num_classes']
    return train_loader, test_loader, num_classes

def get_test_near_ood_loader(args, ood_name, batch_size):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]] if 'cifar' in args.dataset else (0.485, 0.456, 0.406)
    std = [x / 255 for x in [63.0, 62.1, 66.7]] if 'cifar' in args.dataset else (0.229, 0.224, 0.225)
    
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    
    # for dataset in DATA_INFO[args.dataset]['ood']['near']['datasets']
    data_info = DATA_INFO[args.dataset]['ood']['near'][ood_name]
    transform = get_transform(is_vit_model, mean, std)
    ood_data = CustomDataset(
    data_dir=data_info['data_dir'],
    imglist_path=data_info['imglist_path'],
    transform=transform)
    if not is_vit_model:
        ood_data = DualResolutionDataset(ood_data)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)  
    
    return ood_loader

def get_test_far_ood_loader(args, ood_name, batch_size):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]] if 'cifar' in args.dataset else (0.485, 0.456, 0.406)
    std = [x / 255 for x in [63.0, 62.1, 66.7]] if 'cifar' in args.dataset else (0.229, 0.224, 0.225)
    
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    
    # for dataset in DATA_INFO[args.dataset]['ood']['near']['datasets']
    data_info = DATA_INFO[args.dataset]['ood']['far'][ood_name]
    transform = get_transform(is_vit_model, mean, std)
    ood_data = CustomDataset(
    data_dir=data_info['data_dir'],
    imglist_path=data_info['imglist_path'],
    transform=transform)
    if not is_vit_model:
        ood_data = DualResolutionDataset(ood_data)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)  
     
    return ood_loader

def load_imagenet_labels(file_path):
    with open(file_path, 'r') as file:
        # Read the entire file content
        content = file.read()

    # Since the file content resembles a dictionary, we can use eval to parse it
    # Be cautious using eval, ensure the file content is trusted
    imagenet_labels = eval(content)
    
    return imagenet_labels

def idx_to_label(args):
    if args.dataset == 'CIFAR10':
        return {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
    elif args.dataset == 'CIFAR100':
        return {
            0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver',
            5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle',
            10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly',
            15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree',
            60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
        }
    elif args.dataset == 'imagenet':
        # Ensure you have downloaded the ImageNet dataset
        # imagenet_data = ImageNet(root=DATA_INFO[args.dataset]['id']['train']['data_dir'], split='train')
        imagenet_labels = load_imagenet_labels('./data/imagenet1000_clsidx_to_labels.txt')
        return imagenet_labels
    else:
        raise ValueError(f"No idx_to_label mapping for dataset {args.dataset}")