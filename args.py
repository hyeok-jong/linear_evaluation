

import argparse
import datetime

def set_parser():
    parser = argparse.ArgumentParser('standard')
    parser.add_argument('--model_path', type = str, default = './supcon.pth')
    parser.add_argument('--dataset', type = str, default = 'cifar100')
    parser.add_argument('--method', type = str, default = 'finetune', help = ['fineture, linear'])

    # learning hyperparameters
    parser.add_argument('--epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 128)

    parser.add_argument('--wandb_entity', type=str, default='hyeokjong', help='Wandb ID')
    parser.add_argument('--wandb_project', type=str, default=None, help='Project name')
    parser.add_argument('--short', type=str, default=None, help='short name')
    
    args = parser.parse_args()

    if args.wandb_project == None:
        args.wandb_project = f'[Linear Evaluation]'
    if args.short == None:
        args.short = f'[{args.dataset}][{args.method}][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
    print(args)
    return args