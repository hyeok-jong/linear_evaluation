import torch
import torchvision

NUM_CLS = {
    'cifar10' : 10,
    'cifar100' : 100,
    'food101' : 101,
    'aircraft' : 102, 
    'dtd' : 47, 
    'pet' : 37,
    'caltech101' : 101,
    'flowers' : 102,  
    'sun' : 397,
    'stl10': 10
}





class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Classifier(torch.nn.Module):
    def __init__(self, num_classes = 0, in_dim = 2048):
        super().__init__()
        self.fc = torch.nn.Linear(
            in_features = in_dim,
            out_features = num_classes
        )
    def forward(self, features):
        return self.fc(features)

    
"""
def set_models(path = './supcon.pth', freeze = True, dataset = 'model', cuda = True):

    '''
    https://github.com/HobbitLong/SupContrast
    Due to BatchNormalization layer, two models
    '''

    # loads model and delete last fully connetected layer
    model = torchvision.models.resnet50(weights = None)
    model.fc = Identity()

    # loads pre-trained model
    loaded = torch.load(f'{path}', map_location = 'cpu')
    loaded_state_dict = loaded['model']

    # rename parameters dict
    new_state_dict = dict()
    for name, params in loaded_state_dict.items():
        name = name.replace('module.encoder.', '')
        new_state_dict[name] = params

    # delete projection head parameters
    state_dict = dict()
    for name, params in new_state_dict.items():
        if not name.startswith('module.head'):
            state_dict[name] = params

    # feed processed parameters to model
    model.load_state_dict(state_dict, strict = True)

    # Freeze all the parameters in model
    if freeze:
        print('freezing....')
        for name, param in model.named_parameters():
            param.requires_grad = False
        print('done')
    
    # set classifier
    classifier = Classifier(NUM_CLS[dataset])

    # print grad
    '''
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    for name, param in classifier.named_parameters():
        print(name, param.requires_grad)
    '''
    
    if cuda:
        model = model.cuda()
        classifier = classifier.cuda()
        
    return model, classifier






def set_models(path = './trained_models/checkpoint_0100.pth.tar', freeze = True, dataset = 'model', cuda = True):

    '''
    https://github.com/sthalles/SimCLR
    Due to BatchNormalization layer, two models
    '''

    # loads model and delete last fully connetected layer
    model = torchvision.models.resnet18(weights = None)
    model.fc = Identity()

    # loads pre-trained model
    loaded = torch.load(f'{path}', map_location = 'cpu')
    loaded_state_dict = loaded['state_dict']

    # rename parameters dict
    # delete projection head parameters
    new_state_dict = dict()
    for name, params in loaded_state_dict.items():
        if not name.startswith('backbone.fc'):
            name = name.replace('backbone.', '')
            new_state_dict[name] = params

    # feed processed parameters to model
    model.load_state_dict(new_state_dict, strict = True)

    # Freeze all the parameters in model
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    
    # set classifier
    classifier = Classifier(NUM_CLS[dataset], in_dim = 512)

    # print grad
    '''
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    for name, param in classifier.named_parameters():
        print(name, param.requires_grad)
    '''
    
    if cuda:
        model = model.cuda()
        classifier = classifier.cuda()
        
    return model, classifier


"""



def set_models(path = './trained_models/checkpoint_40.tar', freeze = True, dataset = 'model', cuda = True):

    '''
    https://github.com/Spijkervet/SimCLR
    Due to BatchNormalization layer, two models
    '''

    # loads model and delete last fully connetected layer
    model = torchvision.models.resnet18(weights = None)
    model.fc = Identity()

    # loads pre-trained model
    loaded = torch.load(f'{path}', map_location = 'cpu')
    loaded_state_dict = loaded

    # rename parameters dict
    # delete projection head parameters
    new_state_dict = dict()
    for name, params in loaded_state_dict.items():
        if not name.startswith('projector.'):
            name = name.replace('encoder.', '')
            new_state_dict[name] = params

    # feed processed parameters to model
    model.load_state_dict(new_state_dict, strict = True)

    # Freeze all the parameters in model
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad = False
    
    # set classifier
    classifier = Classifier(NUM_CLS[dataset], in_dim = 512)

    # print grad
    '''
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    for name, param in classifier.named_parameters():
        print(name, param.requires_grad)
    '''
    
    if cuda:
        model = model.cuda()
        classifier = classifier.cuda()
        
    return model, classifier
