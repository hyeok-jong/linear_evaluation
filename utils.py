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