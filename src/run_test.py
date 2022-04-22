"""

"""
import argparse
import json
import logging
import string
from copy import deepcopy
from random import choices

import wandb
import yaml

import setup
import utils
from datasets import split_data
from train import GNN


# --- Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='yaml config file', nargs='?')
parser.add_argument('-i', '--run_id', help='wandb run id to base config on', nargs='*')
args = parser.parse_args()

if (args.config is None) == (args.run_id is None):
    raise ValueError('should specify either config or run id, not both')


# Specified config file
if args.config is not None:
    # --- Parse config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Single config
    configs = [config]

else:
    configs = []
    for run_id in args.run_id:
        config = utils.get_run_config(run_id)

        if 'cv.k_fold' not in config or \
           not isinstance(config['cv.k_fold'], int):
            logging.info('setting default value for cv.k_fold!')
            config['cv.k_fold'] = setup.CONFIG_DEFAULT_K_FOLD

        desc = utils.display_json_config(config)
        logging.info(f'got config from run:\n{desc}')

        configs.append(config)

job_types = []
for config in configs:
    # --- Generate job type
    random_id = ''.join(choices(string.ascii_lowercase + string.digits, k=6))
    job_type = f'test-{random_id}'
    job_types.append(job_type)

    logging.info(f'job type is \n{job_type}\n\n')
    logging.info(f'running config: \n{utils.display_json_config(config)}')

    # --- Data splitting
    test_set, split_list = split_data(
        path=setup.get_dataset_path(config['dataset.name']),
        num_node_features=config['dataset.num_node_features'],
        seed=config['cv.seed'],
        cv=True,
        k_cross=config['cv.k_fold'],
        in_memory=config['dataset.in_memory'],
        node_feat_transform=config.get('dataset.node_feat.transform')
    )

    for i, (train_set, val_set) in enumerate(split_list):
        run = wandb.init(**setup.WANDB_SETTINGS,
                         reinit=True,
                         group=f"model-{config['model.name']}",
                         job_type=job_type,
                         name=f'fold-{i + 1}',
                         config=config,
                         tags=[job_type])

        logging.info(f'Test model fold {i+1}/{config["cv.k_fold"]}')
        logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
        logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')
        logging.info(f'test set, length {len(test_set)}, {test_set.patients}')

        # --- Model initialization
        logging.debug('model initialization')
        gnn = GNN(
            config=config,
            train_set=train_set,
            valid_set=val_set,
            test_set=test_set
        )

        # --- Train
        val_metrics = gnn.train(
            epochs=config['epochs'],
            early_stop=config['early_stop'],
            allow_stop=config['allow_stop'],
            run=run
        )

        # --- Eval
        test_metrics = gnn.evaluate(val_set=False, run=run)
        logging.info(f'test metrics: {test_metrics}')

        gnn.save_predictions(run)

        run.finish()


logging.info(f'summary of job types: {", ".join(job_types)}')
logging.info('end of run_test.py')
