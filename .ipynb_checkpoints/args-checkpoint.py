

import argparse
import datetime

def set_parser():
    parser = argparse.ArgumentParser('standard')
    parser.add_argument('--exper', type = str)
    parser.add_argument('--model_path', type = str, default = './trained_models/checkpoint_100.tar', help = ['./trained_models/our_supcon.pth.tar', './trained_models/our_genscl.pth.tar'])
    parser.add_argument('--dataset', type = str, default = 'stl10')
    parser.add_argument('--method', type = str, default = 'linear', help = ['fineture, linear'])

    # learning hyperparameters
    parser.add_argument('--epochs', type = int, default = 500)
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--batch_size', type = int, default = 256)

    parser.add_argument('--wandb_entity', type=str, default='hyeokjong', help='Wandb ID')
    parser.add_argument('--wandb_project', type=str, default=None, help='Project name')
    parser.add_argument('--short', type=str, default=None, help='short name')
    
    args = parser.parse_args()

    if args.wandb_project == None:
        args.wandb_project = f'[Linear Evaluation]'
    if args.short == None:
        args.short = f'[{args.dataset}][{args.method}][{args.lr}][{args.exper}][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'
    print(args)
    return args