from datasets import train_test_for_eval

import numpy as np
import os
import random
import torch
import argparse
import wandb

import yaml
from itertools import product

from util.train import GNN

import numpy as np
import os
import random
import torch
import argparse
import wandb

from util.train import GNN
from datasets import split_data
from sklearn.model_selection import train_test_split

MODEL_TYPE = 'CoordToCnc_KNN5'
PATH_DATA = "../data/CoordToCnc_KNN5"
HYPER_PARAMS = "sandy-star-2721_params.yaml" #"lemon-shape-4234_params.yaml" # MagToCnc on left "royal-river-3568_10Rot_params.yaml"# "mild-sound-1171_10Rot_params.yaml" # "royal-river-3568_params.yaml"


test_set, split_list = split_data(path= PATH_DATA, 
                        num_node_feat=3,
                        cv=True,
                        k_cross=10)
#train_set, val_set = split_list[0]

# python3 evaluate.py --allow_stop 0 --batch_size 10000 --early_stop 10 --epochs 1000 --model NoPhysicsGnn --num_node_features 3 --optim Adam --optim_lr 0.00001 --optim_momentum 0 --path_data '../data/CoordTocnc' --path_model './experiments/util/' --physics 0 --seed 0 --weighted_loss 0.6

yaml_file = open(HYPER_PARAMS)
arg_dic = yaml.load(yaml_file, Loader=yaml.FullLoader)

keys = arg_dic.keys()
values = arg_dic.values()
for instance in product(*values): # should be length one
    config = dict(zip(keys, instance))
    args=config
    for count,( train_set, val_set) in enumerate(split_list):

        run = wandb.init(reinit=True,
                project='mi-prediction',
                group = 'evaluate',
                job_type=MODEL_TYPE,
                config=config,
                name='second_run'+str(count))
    
        print("run nb ", count)

        print("Train set length: ", len(train_set))
        print("Val set length: ", len(val_set))
        print("Val patients: ", val_set.data)

        #print("arguments for the run: ", args)

        print("Test set length: ", len(test_set))
        print("Test patients: ", test_set.data)

        optim_param = {
            'name': args['optim'],
            'lr': args['optim_lr'],
            'momentum': args['optim_momentum'],
        }

        model_param = {
            'physics': args['physics'],
            'name': args['model'],
        }

        gnn = GNN(
            args['path_model'] + args['model'],
            model_param,
            train_set,
            val_set,
            test_set,
            args['batch_size'],
            optim_param,
            args['weighted_loss'],
        )

        gnn.train(args['epochs'], args['early_stop'], args['allow_stop'], run)
        gnn.evaluate(val_set=False, run=run)

        torch.save(gnn, "models/"+MODEL_TYPE)
