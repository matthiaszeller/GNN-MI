
import argparse
import logging

import wandb
import yaml

import setup
from datasets import split_data
from train import GNN

# Dry run to retrieve config without having to parse
# only works if this script was run with wandb sweep agent
# otherwise parse from yaml file
run = wandb.init(**setup.WANDB_SETTINGS,
                 job_type='trash',
                 group='trash')
logging.info(f'launched empty run with id {run.id}')
config = run.config
random_name = run.name
if len(config.keys()) == 0:
    logging.info(f'parsing config from file')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='yaml config file')
    args = parser.parse_args()
    with open(args.file) as f:
        config = yaml.load(f)


# --- Data splitting
_, split_list = split_data(path=setup.get_dataset_path(config['dataset.name']),
                           num_node_features=config['dataset.num_node_features'],
                           cv=True,
                           k_cross=config['k_fold'],
                           seed=config['seed'],
                           in_memory=config['dataset.in_memory'])

# --- KFoldCV
n_folds = len(split_list)
for i, (train_set, val_set) in enumerate(split_list):
    run = wandb.init(**setup.WANDB_SETTINGS,
                     reinit=True,
                     group=f"model-{config['model.name']}",
                     job_type=random_name,
                     name=f'fold-{i+1}')

    logging.info(f'KFold CV {i + 1}/{n_folds}')
    logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
    logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')

    # Model initialization
    logging.debug('model initialization')
    gnn = GNN(
        config=config,
        train_set=train_set,
        valid_set=val_set,
        test_set=None,
    )

    # Train
    val_metrics = gnn.train(
        epochs=config['epochs'],
        early_stop=config['early_stop'],
        allow_stop=config['allow_stop'],
        run=run
    )

    run.save()
    run.finish()

logging.info('KFoldCV finished')

