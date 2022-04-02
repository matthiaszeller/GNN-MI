"""

"""
import argparse
import json
import logging
import string
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
parser.add_argument('-i', '--run_id', help='wandb run id to base config on')
args = parser.parse_args()

if (args.config is None) == (args.run_id is None):
    raise ValueError('should specify either config or run id, not both')


# Specified config file
if args.config is not None:
    # --- Parse config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

else:
    api = wandb.Api()
    run = api.run(f'{setup.WANDB_SETTINGS["entity"]}/{setup.WANDB_SETTINGS["project"]}/{args.run_id}')
    config = json.loads(run.json_config)
    config = utils.unnest_json_config(config)

    if 'cv.test_reps' not in config or \
       not isinstance(config['cv.test_reps'], int):
        logging.info('setting default value for cv.test_reps!')
        config['cv.test_reps'] = setup.CONFIG_DEFAULT_TEST_REPS

    desc = utils.display_json_config(config)
    logging.info(f'got config from run:\n{desc}')


# --- Generate job type
random_id = ''.join(choices(string.ascii_lowercase + string.digits, k=6))
job_type = f'test-{random_id}'

logging.info(f'job type is \n{job_type}\n\n')


# --- Data splitting
test_set, ((train_set, val_set),) = split_data(
    path=setup.get_dataset_path(config['dataset.name']),
    num_node_features=config['dataset.num_node_features'],
    seed=config['cv.seed'],
    cv=False,
    valid_ratio=config['cv.valid_ratio'],
    in_memory=config['dataset.in_memory']
)


for i in range(config['cv.test_reps']):
    run = wandb.init(**setup.WANDB_SETTINGS,
                     reinit=True,
                     group=f"model-{config['model.name']}",
                     job_type=job_type,
                     name=f'run-{i + 1}',
                     config=config)

    logging.info(f'Test model {i+1}/{config["cv.test_reps"]}')
    logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
    logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')
    logging.info(f'unused test set, length {len(test_set)}, {test_set.patients}')

    # --- Model initialization
    logging.debug('model initialization')
    gnn = GNN(
        config=config,
        train_set=train_set,
        valid_set=val_set,
        test_set=test_set,
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

    run.finish()

logging.info('end of run_test.py')
