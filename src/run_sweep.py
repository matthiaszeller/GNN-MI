"""
Run coarse hyperparameter tuning with wandb sweep framework.
Only use train/val split, i.e. no KFoldCV since coarse grid => many hyperparams.
Use run_sweep_kfold.py for finer hyperparam tuning.
"""
import logging
import sys

import wandb

import setup
from datasets import split_data
from train import GNN

# --- Model name parsing
# Retrieve model name from script params
# Example of what we're trying to recover: """--model.name=Equiv_GIN_KNN5"""
filtered_args = [
    arg for arg in sys.argv if 'model.name' in arg
]
if len(filtered_args) != 1:
    raise ValueError(f'no model name in parameters, this script can only be run with wandb agent for sweeps')

model_name = filtered_args[0].split('=')[1]

# --- Wandb
# The config is already included by wandb agent
run = wandb.init(**setup.WANDB_SETTINGS,
                 group=f"model-{model_name}",
                 job_type='sweep')
config = dict(run.config)

# --- Data splitting
_, ((train_set, val_set),) = split_data(path=setup.get_dataset_path(config['dataset.name']),
                                        num_node_features=config['dataset.num_node_features'],
                                        seed=config['cv.seed'],
                                        valid_ratio=config['cv.valid_ratio'],
                                        cv=False,
                                        in_memory=config['dataset.in_memory'])

logging.info(f'training set, length {len(train_set)}, {train_set.patients}')
logging.info(f'validation set, length {len(val_set)}, {val_set.patients}')

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

logging.info('end of run_sweep.py')