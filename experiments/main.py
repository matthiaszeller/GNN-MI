import numpy as np
import os
import random
import torch
import argparse
import wandb

from util.train import GNN
from datasets import split_data

print("parsiiiiing")
print(torch.cuda.is_available())
#print(os.listdir("./data/"))

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='NoPhysicsGnn') #TsviPlusCnc or NoPhysicsGnn
parser.add_argument("--num_node_features", type=int, default=3) # 3 for Coord, 60 for WSS it seems
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--early_stop", type=int, default=10)
parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'SGD'])
parser.add_argument("--optim_lr", type=float, default=0.0005)
parser.add_argument("--optim_momentum", type=float, default=0.0)
parser.add_argument("--physics", type=int, default=0)
parser.add_argument("--weighted_loss", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--save_name", type=str, default='no_save')
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--cross_val", type=bool, default=True)
parser.add_argument("--k_cross", type=int, default='10')


# parser.add_argument("--cross_validation", action='store_true', default=False)
parser.add_argument("--path_data", type=str, default="../data/CoordToCnc")
parser.add_argument("--path_model", type=str, default="./experiments/util/")
args = parser.parse_args()

print("starting .......")

test_set, split_list = split_data(args.path_data, 
                        args.num_node_features,
                        cv=args.cross_val,
                        k_cross=args.k_cross)

print("Test set length: ", len(test_set))
print("Test patients: ", test_set.data)

for i, (train_set, val_set) in enumerate(split_list):
    print('cross val: ', i)
    print("Train set length: ", len(train_set))
    print("Val set length: ", len(val_set))
    print("Val patients: ", val_set.data)
    wandb.init(project='mi-prediction',
               group='test_noAugm',
               name='no_aug_save'+str(i),
               config=args)

    args.path_data = os.path.join(args.path_data) 

    optim_param = {
        'name': args.optim,
        'lr': args.optim_lr,
        'momentum': args.optim_momentum,
    }

    model_param = {
        'physics': args.physics,
        'name': args.model,
    }

    gnn = GNN(
        args.path_model + args.model,
        model_param,
        train_set,
        val_set,
        test_set,
        args.batch_size,
        optim_param,
        args.weighted_loss,
    )

    # gnn = GNN("./experiments/util/NoPhysicsGnn", {'physics': 0, 'name': 'NoPhysicsGnn'}, train_set,valid_set,test_set,5,{'name': 'Adam', 'lr': 0.0005, 'momentum': 0.0},0.5)

    print('Runnning!!')
    gnn.train(args.epochs, args.early_stop) #train the model. ERROR
    #print("Rotatioooooons")
    #gnn.train_aug_rot(args.epochs, args.early_stop, eps=args.eps)
    #gnn.evaluate(val_set=False)

    #if args.save_name != 'no_save' and not args.cross_val:
        #print("Saving under: ", args.save_name)
        #print("Test patients: ", test_set.data)
        #torch.save(gnn, args.save_name)
        # with open('model_CoordToCnc_rot(-45,45, 9).pt', 'rb') as f:
        #     gnn_re = torch.load(f)

    wandb.save()
    wandb.finish()
    print("Done!!")
