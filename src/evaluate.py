import torch
import wandb
import yaml
from itertools import product

from src.train import GNN
from src.datasets import split_data

"""TODO:
    3) make model saving optional
    5) compress code (model init) with main_cross_val one's
    6) rename to optimized hyper param"""

# TODO: infer the following from hyperparams
MODEL_TYPE = 'original_CoordToCnc_KNN5'
# PATH_DATA = "../data/CoordToCncKNN5"
HYPER_PARAMS = "optim_params/param_original_CoordToCncKNN5.yaml"

# load hyperparameters into dictionary
yaml_file = open(HYPER_PARAMS)
arg_dic = yaml.load(yaml_file, Loader=yaml.FullLoader)
keys = arg_dic.keys()
values = arg_dic.values()

# should be length one since testing should
# not be done on all instance of the grid
for instance in product(*values):
    config = dict(zip(keys, instance))  # hyperparams
    test_set, split_list = split_data(path=config["path_data"],
                                      num_node_feat=3,
                                      cv=True,
                                      k_cross=10,
                                      seed=config['seed'])
    # train on all split_list instances mimicing cross validation
    for count, (train_set, val_set) in enumerate(split_list):
        # group name differentiates from the other two
        # similar naming as in main_cross_val.py
        run = wandb.init(reinit=True,
                         project='mi-prediction',
                         group='evaluate',
                         job_type=MODEL_TYPE,
                         config=config,
                         name=MODEL_TYPE+"-"+str(count))

        # print things to see it from wandb for book-keeping
        print("run nb ", count)
        print("Train set length: ", len(train_set))
        print("Val set length: ", len(val_set))
        print("Val patients: ", val_set.data)
        print("Test set length: ", len(test_set))
        print("Test patients: ", test_set.data)

        # set model parameters
        optim_param = {
            'name': config['optim'],
            'lr': config['optim_lr'],
            'momentum': config['optim_momentum'],
        }
        model_param = {
            'physics': config['physics'],
            'name': config['model'],
        }

        # initialize model
        gnn = GNN(
                config['path_model'] + config['model'],
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

        # save model
        torch.save(gnn, "models/"+MODEL_TYPE+"-"+str(count))
