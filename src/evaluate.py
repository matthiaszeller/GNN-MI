import argparse
import logging
from itertools import product

import torch
import wandb
import yaml

import setup
from datasets import split_data
from train import GNN
from utils import grid_search


def save_model(model, model_name):
    _, path_out = setup.get_data_paths()
    path = path_out.joinpath('models')
    if not path.exists():
        path.mkdir()

    path_file = path.joinpath(model_name)
    torch.save(model.model.state_dict(), path_file)
    logging.info(f'saved model in {path_file}')


"""TODO:
    3) make model saving optional
    5) compress code (model init) with main_cross_val one's
    6) rename to optimized hyper param"""

parser = argparse.ArgumentParser()
parser.add_argument('yaml_config', type=str, help='yaml file of hyperparameters')
args = parser.parse_args()
yaml_file = args.yaml_config

grid = list(grid_search(yaml_file))
# should be length one since testing should
# not be done on all instance of the grid
if len(grid) != 1:
    raise ValueError('grid search should be length one')

config = grid[0]
test_set, split_list = split_data(path=setup.get_dataset_path(config['dataset']['name']),
                                  num_node_feat=3,
                                  cv=True,
                                  k_cross=10, # TODO check with previous verison I made mistake
                                  seed=config['seed'])

# TODO remake this, bad design, only one split
# train on all split_list instances mimicing cross validation
for count, (train_set, val_set) in enumerate(split_list):
    # group name differentiates from the other two
    # similar naming as in main_cross_val.py
    run = wandb.init(reinit=True,
                     **setup.WANDB_SETTINGS,
                     group='evaluate',
                     job_type=f"model-{config['model']['name']}",
                     config=config,)
    logging.info(f'wandb run id {run.id}')

    # print things to see it from wandb for book-keeping
    logging.info(f'evaluation, counter = {count}')
    logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
    logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')
    logging.info(f'test set, length {len(test_set)}, {test_set.patients}')

    # set model hyperparameters
    optim_param = {
        'optimizer': config['optim'],
        'lr': config['optim_lr'],
        'momentum': config['optim_momentum'],
    }
    model_param = {
        'physics': config['physics'],
        'type': config['model']['type'],
    }

    # initialize model
    gnn = GNN(
            model_param,
            train_set,
            val_set,
            test_set,
            config['batch_size'],
            optim_param,
            config['weighted_loss'],
            config)

    # train then evaluate
    gnn.train(config['epochs'],
              config['early_stop'],
              config['allow_stop'],
              run)
    gnn.evaluate(val_set=False, run=run)
    run.save()
    run.finish()

    # save model
    save_model(gnn, f"{config['model']['name']}.pt")
