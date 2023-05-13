import os
import torch
import wandb
import copy

from tqdm import tqdm
from args import set_parser
from models import set_models
from loaders import set_loaders
from utils import multi_accuracy, init_wandb, set_random
from trainer import Trainer
from itertools import chain


if __name__ == '__main__':
    set_random(0)

    args = set_parser()
    save_path = f'./saves/{args.method}/{args.dataset}_{args.lr}_{args.exper}'
    os.makedirs(save_path, exist_ok = True)

    # SETTING
    if args.method == 'linear':
        freeze = True
        encoder_check = True
    elif args.method == 'finetune':
        freeze = False
        encoder_check = False

    encoder, classifier = set_models(
        path = args.model_path,
        encoder_freeze = freeze,
        dataset = args.dataset,
        cuda = True
    )

    compare_encoder = copy.deepcopy(encoder.state_dict()).values()
    compare_classifier = copy.deepcopy(classifier.state_dict()).values()


    loss_function = torch.nn.CrossEntropyLoss()#.cuda()
    accuracy_function = multi_accuracy

    
    '''
    # Ours
    optimizer = torch.optim.SGD(
        chain(encoder.parameters(), classifier.parameters()), # Since for both linear evaluation and finetuning
        # classifier.parameters(),
        lr = args.lr, 
        momentum = 0.9, 
        weight_decay = 5e-4)
    '''
    # Spijkervet
    optimizer = torch.optim.Adam(
        chain(encoder.parameters(), classifier.parameters()), 
        lr = args.lr,
        weight_decay = 1.0e-6)
    
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer, 
        T_max = args.epochs)
        
    train_loader, valid_loader, test_loader = set_loaders(args.dataset, args.batch_size, args.method)

    init_wandb(args)
    trainer = Trainer()
    trainer.initialize(
        train_loader = train_loader,
        valid_loader = test_loader,
        test_loader = test_loader,
        encoder = encoder,
        classifier = classifier,
        optimizer = optimizer,
        lr_scheduler = lr_scheduler,
        loss_function = loss_function,
        accuracy_function = accuracy_function,
        method = args.method
    )


    best_val_acc = 0
    result_dict = dict()
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_result = trainer.one_epoch_train()
        valid_result = trainer.one_epoch_valid()
        
        result_dict.update(train_result)
        result_dict.update(valid_result)
        result_dict['learning rate'] = trainer.get_lr()
        wandb.log(result_dict, step = epoch)
        for key, val in result_dict.items():
            print(key, val)

        if best_val_acc < result_dict['valid acc']:
            best_val_acc = result_dict['valid acc']
            trainer.save(path = f'{save_path}/best.pth', others = {'epoch' : epoch})
        
        
        if epoch % 10 == 1:
            for i, j in zip(compare_encoder, trainer.encoder.state_dict().values()):
                assert encoder_check == (i == j).all().item(), "ERROR"
            print((i == j).all().item())
            for i, j in zip(compare_classifier, trainer.classifier.state_dict().values()):
                assert False == (i == j).all().item(), "classifier not updating...."
            print((i == j).all().item())



    trainer.load(f'{save_path}/best.pth')
    test_result = trainer.test()
    print(test_result)
    wandb.log(test_result)
    wandb.finish()

    
