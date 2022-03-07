import argparse
import logging
import time
from itertools import product

import torch
import wandb
import yaml

import setup
from datasets import split_data
from setup import WANDB_SETTINGS
from train import GNN

# following is for book-keeping of experiments i.e. naming for wandb
from utils import grid_search

# MODEL_TYPE = 'Equiv_GIN_Gaussian2'# TODO get rid of all those things
# Options include: 'Equiv_FullyCon_3CVoutof10' # 'noAugEdge'
# #'MagToCnc_KNN5' # '10RandRot_CoordToCnc_CV10' or 'KNN5_CV10'
# PATH_DATA = "../data/CoordToCnc"
# Options include: 10RandRot_CoordToCnc  4RotCoordToCnc  CoordToCnc_KNN5


def run_cross_val(split_list, args, config_name):
    """
    Runs cross validation on dataset that is already split in the appropriate
    way into a list of (train, validation) splits, a test_set, arguments of the
    single run, and meta argument of the grid search.

    Parameters:
    ------------
    split_list : list of tuples of DataSet objects. Length of list is nb of
    runs in the cross validation. The first entry of the tuple is the
    training set, the second is the validation set.

    test_set :  DataSet object, only used to print the elements in the test
    set, as a sanity check. TODO: make optional

    args : dictionary of arguments to feed into the model. Check out the
    hyperparameters.yaml file for an example.

    wandb_job_type: name of job type for wandb
    """
    # metrics to log, val scores of each runs, to be averaged later:
    dic_val_metrics = {
        'avg_val_acurracy': [],
        'avg_val_precision': [],
        'avg_val_recall': [],
        'avg_val_sensitivity': [],
        'avg_val_specificity': [],
        'avg_val_f1score': [],
        'avg_val_graph_loss': []
    }
    # following loop corresponds to cross validation runs
    n_folds = len(split_list)
    for i, (train_set, val_set) in enumerate(split_list):
        # initialize wandb log to keep track of training:
        # each run is identified by name, group and job_type.
        # 'indiv-CV-run' group name prefix used for single runs
        # of cross val.
        # Each cross val runs have a numeral post-fix identifier
        run = wandb.init(reinit=True,
                         **WANDB_SETTINGS,
                         group=f"model-{args['model']['name']}",
                         job_type=config_name,
                         name=f'fold-{i + 1}',
                         config=args)

        # print intermediate things for sanity check
        logging.info(f'KFold CV {i + 1}/{n_folds}')
        logging.info(f'training set, length {len(train_set)}, {train_set.data}')
        logging.info(f'validation set, length {len(val_set)}, {val_set.data}')

        # set model hyperparameters
        optim_param = {
            'optimizer': args['optim'],
            'lr': args['optim_lr'],
            'momentum': args['optim_momentum'],
        }
        model_param = {
            'physics': args['physics'],
            'type': args['model']['type'],
        }

        # initialize the model
        logging.debug('model initialization')
        gnn = GNN(
            model_param=model_param,
            train_set=train_set,
            valid_set=val_set,
            test_set=None,
            batch_size=args['batch_size'],
            optim_param=optim_param,
            weight1=args['weighted_loss'],
            args=args
        )

        # Start training
        # logging the metrics in wandb is done in the train method
        (acc, prec, rec, sensitivity,
         specificity, f1score, graph_loss) = gnn.train(args['epochs'],
                                                       args['early_stop'],
                                                       args['allow_stop'],
                                                       run)
        run.save()
        run.finish()

        # accumulate metrics over each individual cross validation run:
        dic_val_metrics['avg_val_acurracy'].append(acc)
        dic_val_metrics['avg_val_precision'].append(prec)
        dic_val_metrics['avg_val_recall'].append(rec)
        dic_val_metrics['avg_val_sensitivity'].append(sensitivity)
        dic_val_metrics['avg_val_specificity'].append(specificity)
        dic_val_metrics['avg_val_f1score'].append(f1score)
        dic_val_metrics['avg_val_graph_loss'].append(graph_loss)

    # wnandb run to log metrics averaged over all cross val runs.
    # 'avg-CV-val-score' group name prefix for book keeping and not
    # confused them with individual run logs
    meta_run = wandb.init(reinit=True,
                          **WANDB_SETTINGS,
                          group=f"CV-avg",
                          job_type=f"model-{args['model']['name']}",
                          name=config_name,
                          config=args)
    # book keeping, these prints will appear in the wandb log
    logging.info(f'meta run, config {config_name}, {args}')

    # average over indiv runs and log metrics in wandb
    avg_dic = {}
    for key in dic_val_metrics.keys():
        avg_dic[key] = sum(dic_val_metrics[key]) / len(dic_val_metrics[key])
    meta_run.log(avg_dic)
    wandb.finish()
    logging.info('KFoldCV finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_config', type=str, help='yaml file of hyperparameters')
    args = parser.parse_args()
    yaml_file = args.yaml_config

    logging.info(f'cuda available: {torch.cuda.is_available()}')
    logging.info(f'starting grid search from file {yaml_file}')
    for count, config in enumerate(grid_search(yaml_file)):
        logging.info(f'grid search iter {count}, config = {config}')
        # split data as test set and then cross val list with
        # elements of format (train, valid)
        _, split_list = split_data(path=setup.get_dataset_path(config['dataset']),
                                   num_node_feat=3,
                                   cv=True,
                                   k_cross=10,
                                   seed=config['seed'])
        config_name = f'config-{count}'
        run_cross_val(split_list=split_list,
                      args=config,
                      config_name=config_name)

    logging.info(f'grid search finished')


if __name__ == '__main__':
    main()

