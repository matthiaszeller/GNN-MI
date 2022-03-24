"""

"""
import argparse
import logging
import string
from random import choices

import wandb
import yaml

import setup
from datasets import split_data
from train import GNN


# --- Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='yaml config file')
args = parser.parse_args()

# --- Generate job type
random_id = ''.join(choices(string.ascii_lowercase + string.digits, k=6))
job_type = f'test-{random_id}'

logging.info(f'job type is {job_type}')

# --- Parse config
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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
