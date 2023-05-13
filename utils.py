import wandb
import torch

def init_wandb(args):
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_project,
        name=args.short,
        config=args,
    )
    wandb.run.save()
    return wandb.config

def multi_accuracy(output : torch.tensor, target : torch.tensor):
    '''
    batch = mini batch or full batch
    output : (batch, class) : dosent matter after softmax or not
    target : (batch, )
    '''
    output = torch.softmax(output, dim = 1)
    pred = torch.argmax(output, dim = 1)

    total_corrects = sum(pred == target)

    return total_corrects / len(pred)



def set_random(random_seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(random_seed)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)