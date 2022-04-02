"""

"""
import argparse
import json
import logging

import wandb
import yaml

import setup
import utils
from datasets import split_data
from train import GNN

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='yaml config file', nargs='?')
parser.add_argument('-j', '--job_type', nargs='?', type=str, help='name of wandb job type')
parser.add_argument('-i', '--run_id', type=str, help='run to base the config on', nargs='?')
args = parser.parse_args()

# Either config or run_id is provided
if (args.config is None) == (args.run_id is None):
    raise ValueError('either config or run_id should be provided, not both')

# Option 1: get config from yaml file
if args.config is not None:
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
# Option 2: copy config from run
else:
    logging.info(f'getting config from run {args.run_id}')
    api = wandb.Api()
    run = api.run(f'{setup.WANDB_SETTINGS["entity"]}/{setup.WANDB_SETTINGS["project"]}/{args.run_id}')
    config = json.loads(run.json_config)
    config = utils.unnest_json_config(config)

    # kfold might be missing because the run could be from a coarse sweep
    if not isinstance(config['cv.k_fold'], int):
        logging.info(f'replacing k_fold with default value!')
        config['cv.k_fold'] = setup.CONFIG_DEFAULT_K_FOLD

    desc = utils.display_json_config(config)
    logging.info(f'got config from run:\n{desc}')

if 'job_type' in args:
    job_type = args.job_type
else:
    job_type = None
    if args.run_id is not None:
        job_type = f'kfold-{args.run_id}'

logging.info(f'wandb job type: \n{job_type}\n\n')

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
