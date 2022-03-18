"""

"""
import argparse
import logging

import wandb
import yaml

import setup
from datasets import split_data
from train import GNN

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='yaml config file')
parser.add_argument('job_type', nargs='?', type=str, help='name of wandb job type')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f)

if 'job_type' in args:
    job_type = args.job_type
else:
    job_type = None

# --- Data splitting
test_set, split_list = split_data(path=setup.get_dataset_path(config['dataset.name']),
                                  num_node_features=config['dataset.num_node_features'],
                                  seed=config['cv.seed'],
                                  cv=True,
                                  k_cross=config['cv.k_fold'],
                                  in_memory=config['dataset.in_memory'])

for i, (train_set, val_set) in enumerate(split_list):
    run = wandb.init(**setup.WANDB_SETTINGS,
                     reinit=True,
                     group=f"model-{config['model.name']}",
                     job_type=job_type,
                     name=f'fold-{i+1}',
                     config=config)

    logging.info(f'KFoldCV {i+1}/{len(split_list)}')
    logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
    logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')
    logging.info(f'unused test set, length {len(test_set)}, {test_set.patients}')

    # --- Model initialization
    logging.debug('model initialization')
    gnn = GNN(
        config=config,
        train_set=train_set,
        valid_set=val_set,
        test_set=None,
    )
    
    # --- Train
    val_metrics = gnn.train(
        epochs=config['epochs'],
        early_stop=config['early_stop'],
        allow_stop=config['allow_stop'],
        run=run
    )

    # run.save()
    run.finish()

logging.info('end of run_kfold.py')
