"""

Two cross validation modes are possible, determined by the configuration:
    - if `cv.kfold` is an existing and non-null field, we use Kfold cross validation
    - if `cv.mc_splits` is an existing and non-null field, we use Monte Carlo CV
If both are null or both are non-null, an exception is raised.
"""

import argparse
import json
import logging

import wandb
import yaml

import setup
import utils
from datasets import split_data, shuffle_split_data
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
    config = utils.read_config_file(args.config)

# Option 2: copy config from run
else:
    logging.info(f'getting config from run {args.run_id}')
    config = utils.get_run_config(args.run_id)

# Determine whether KFold or MC CV
kfold = config.get('cv.kfold', None)
mc = config.get('cv.mc_splits', None)

if isinstance(kfold, int) and isinstance(mc, int):
    raise ValueError('cannot specify both KfoldCV and Monte-Carlo CV')
elif (kfold is None) and (mc is None):
    raise ValueError('must specify one of the field `cv.kfold` xor `cv.mc_splits`')

# Display config
logging.info(f'config: \n{utils.display_json_config(config)}')
if 'job_type' in args:
    job_type = args.job_type
else:
    job_type = None

if (job_type is None) and (args.run_id is not None):
    job_type = f'kfold-{args.run_id}'

logging.info(f'wandb job type: \n{job_type}\n\n')

# --- Data splitting
N_splits = None
if kfold is not None:
    logging.info('Cross validation mode: KFOLD')
    # kfold might be missing because the run could be from a coarse sweep
    if not isinstance(config['cv.k_fold'], int):
        logging.info(f'replacing k_fold with default value!')
        config['cv.k_fold'] = setup.CONFIG_DEFAULT_K_FOLD

    test_set, split_list = split_data(path=setup.get_dataset_path(config['dataset.name']),
                                      num_node_features=config['dataset.num_node_features'],
                                      seed=config['cv.seed'],
                                      cv=True,
                                      k_cross=config['cv.k_fold'],
                                      # Args for PatientDataset
                                      in_memory=config['dataset.in_memory'],
                                      exclude_files=config.get('dataset.exclude_files'))
    N_splits = len(split_list)

else:
    logging.info('Cross validation mode: MONTE CARLO')

    test_set, split_list = shuffle_split_data(
        path=setup.get_dataset_path(config['dataset.name']),
        num_node_features=config['dataset.num_node_features'],
        seed=config['cv.seed'],
        n_split=config['cv.mc_splits'],
        # Args for PatientDataset
        in_memory=config['dataset.in_memory'],
        exclude_files=config.get('dataset.exclude_files')
    )
    N_splits = config['cv.mc_splits']


for i, (train_set, val_set) in enumerate(split_list):
    run = wandb.init(**setup.WANDB_SETTINGS,
                     reinit=True,
                     group=f"model-{config['model.name']}",
                     job_type=job_type,
                     name=f'fold-{i+1}',
                     config=config)

    logging.info(f'KFoldCV {i+1}/{N_splits}')
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

    preds = gnn.save_predictions(run)

    # run.save()
    run.finish()

logging.info('end of run_kfold.py')
