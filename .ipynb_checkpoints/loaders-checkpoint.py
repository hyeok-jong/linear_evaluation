
import torch
import torchvision
import torchvision.transforms as transforms

class ToRGB(torch.nn.Module):
    def forward(self, image_tensor):
        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)
        return image_tensor




MEAN = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.50224, 0.4867, 0.4408],
    'stl10': [0.485, 0.456, 0.406],
    'imagenet': [0.485, 0.456, 0.406],
    'food101': [0.485, 0.456, 0.406],
    'aircraft': [0.485, 0.456, 0.406],
    'dtd': [0.485, 0.456, 0.406],
    'pet': [0.485, 0.456, 0.406],
    'caltech101': [0.485, 0.456, 0.406],
    'flowers': [0.485, 0.456, 0.406],
    'sun': [0.485, 0.456, 0.406],
}
STD = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'cifar100':[0.2675, 0.2565, 0.2761],
    'stl10': [0.229, 0.224, 0.225],
    'imagenet': [0.229, 0.224, 0.225],
    'food101': [0.229, 0.224, 0.225],
    'aircraft': [0.229, 0.224, 0.225],
    'dtd': [0.229, 0.224, 0.225],
    'pet': [0.229, 0.224, 0.225],
    'caltech101': [0.229, 0.224, 0.225],
    'flowers': [0.229, 0.224, 0.225],
    'sun': [0.229, 0.224, 0.225],
}

train_ratio = 1.0
data_root_dir = './data'

def set_loaders(dataset, batch_size = 256, method = 'string', size = 224, size_ = 256):# 224, 256
    ####################################################################################################
    if method == 'linear':
        train_transform = transforms.Compose([
            # automatically along the shorter side
            #transforms.Resize(size, interpolation = transforms.InterpolationMode.BICUBIC),
            #transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset], STD[dataset])
            ])
        test_transform = transforms.Compose([
            #transforms.Resize(size, interpolation = transforms.InterpolationMode.BICUBIC),
            #transforms.CenterCrop(size),            
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset], STD[dataset])
            ])
        
    elif method == 'finetune':
        train_transform = transforms.Compose([
            # automatically along the shorter side
            transforms.Resize(size_, interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset], STD[dataset])
            ])
        test_transform = transforms.Compose([
            transforms.Resize(size_, interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),            
            transforms.ToTensor(),
            transforms.Normalize(MEAN[dataset], STD[dataset])
            ])

    ####################################################################################################
    if dataset == 'cifar10':
        data_set = torchvision.datasets.CIFAR10(
            root = data_root_dir, 
            train = True,
            download = True, 
            transform = train_transform)
        
        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform

        test_set = torchvision.datasets.CIFAR10(
            root = data_root_dir, 
            train = False,
            download = True, 
            transform = test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
                                                shuffle = True, num_workers = 4)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size,
                                                   shuffle = False, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size,
                                                shuffle = False, num_workers = 4)
        
    elif dataset == 'cifar100':
    
        data_set = torchvision.datasets.CIFAR100(
            root = data_root_dir, 
            train = True,
            download = True, 
            transform = train_transform)
        
        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform

        test_set = torchvision.datasets.CIFAR100(root = data_root_dir, train = False,
                                            download = True, transform = test_transform)


    elif dataset == 'stl10':
        data_set = torchvision.datasets.STL10(
            root = data_root_dir,
            split = "train",
            download = True,
            transform = train_transform)
        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform
        test_set = torchvision.datasets.STL10(
            root = data_root_dir,
            split = 'test',
            download = True,
            transform = test_transform)
        
        
    elif dataset == 'food101':
        data_set = torchvision.datasets.Food101(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = train_transform)
        
        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform

        test_set = torchvision.datasets.Food101(
            root = data_root_dir,
            split = 'test',
            downlaod = True,
            transform = test_transform
        )


    elif dataset == 'stanfordcars':
        data_set = torchvision.datasets.StanfordCars(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = train_transform)
        
        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform

        test_set = torchvision.datasets.StanfordCars(
            root = data_root_dir,
            split = 'test',
            downlaod = True,
            transform = test_transform
        )

    elif dataset == 'aircraft':
        train_set = torchvision.datasets.FGVCAircraft(
            root = data_root_dir,
            split = 'train',
            download = True,
            annotation_level = 'variant',
            transform = train_transform)
        
        valid_set = torchvision.datasets.FGVCAircraft(
            root = data_root_dir,
            split = 'val',
            download = True,
            annotation_level = 'variant',
            transform = test_transform)
        
        test_set = torchvision.datasets.FGVCAircraft(
            root = data_root_dir,
            split = 'test',
            download = True,
            annotation_level = 'variant',
            transform = test_transform)
    
    elif dataset == 'dtd':
        train_set = torchvision.datasets.DTD(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = train_transform)

        valid_set = torchvision.datasets.DTD(
            root = data_root_dir,
            split = 'val',
            download = True,
            transform = test_transform)

        test_set = torchvision.datasets.DTD(
            root = data_root_dir,
            split = 'test',
            download = True,
            transform = test_transform)

    elif dataset == 'pet':
        data_set = torchvision.datasets.OxfordIIITPet(
            root = data_root_dir,
            split = 'trainval',
            target_types = 'category',
            download = True,
            transform = train_transform)

        train_len = int(len(data_set) * train_ratio)
        valid_len = len(data_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform

        test_set = torchvision.datasets.OxfordIIITPet(
            root = data_root_dir,
            split = 'test',
            target_types = 'category',
            download = True,
            transform = test_transform)


    elif dataset == 'caltech101':


        if method == 'linear':
            train_transform = transforms.Compose([
                # automatically along the shorter side
                transforms.Resize(size, interpolation = transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(MEAN[dataset], STD[dataset])
                ])
            test_transform = transforms.Compose([
                transforms.Resize(size, interpolation = transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),            
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(MEAN[dataset], STD[dataset])
                ])
            
        elif method == 'finetune':
            train_transform = transforms.Compose([
                # automatically along the shorter side
                transforms.Resize(size_, interpolation = transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(MEAN[dataset], STD[dataset])
                ])
            test_transform = transforms.Compose([
                transforms.Resize(size_, interpolation = transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),            
                transforms.ToTensor(),
                ToRGB(),
                transforms.Normalize(MEAN[dataset], STD[dataset])
                ])




        data_set = torchvision.datasets.Caltech101(
            root = data_root_dir,
            download = True,
            transform = train_transform)
        
        train_len = int(len(data_set) * 0.9)
        valid_len = len(data_set) - train_len
        #test_len = valid_len // 2

        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform
        #valid_set, test_set = torch.utils.data.random_split(data_set, [valid_len, test_len])
        test_set = valid_set

    elif dataset == 'flowers':
        train_set = torchvision.datasets.Flowers102(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = train_transform)

        valid_set = torchvision.datasets.Flowers102(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = test_transform)

        test_set = torchvision.datasets.Flowers102(
            root = data_root_dir,
            split = 'train',
            download = True,
            transform = test_transform)


    elif dataset == 'sun':
        data_set = torchvision.datasets.SUN397(
            root = data_root_dir,
            download = True,
            transform = train_transform)
        
        train_len = int(len(data_set) * 0.9)
        valid_len = len(data_set) - train_len
        #test_len = valid_len // 2

        train_set, valid_set = torch.utils.data.random_split(data_set, [train_len, valid_len])
        valid_set = torch.utils.data.Subset(valid_set.dataset, valid_set.indices)
        valid_set.dataset.transform = test_transform
        #valid_set, test_set = torch.utils.data.random_split(data_set, [valid_len, test_len])
        test_set = valid_set

        

        
        
        
        
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size,
                                            shuffle = True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size,
                                                shuffle = False, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size,
                                            shuffle = False, num_workers = 4) 
    
    for loader in [train_loader, valid_loader, test_loader]:
        print(loader.dataset.__len__(), len(loader))


    return train_loader, valid_loader, test_loader
        
