

import time
from itertools import product

import torch
import wandb
import yaml

from src.datasets import split_data
from src.train import GNN

# following is for book-keeping of experiments i.e. naming for wandb
MODEL_TYPE = 'Equiv_GIN_Gaussian2'
# Options include: 'Equiv_FullyCon_3CVoutof10' # 'noAugEdge'
# #'MagToCnc_KNN5' # '10RandRot_CoordToCnc_CV10' or 'KNN5_CV10'
# PATH_DATA = "../data/CoordToCnc"
# Options include: 10RandRot_CoordToCnc  4RotCoordToCnc  CoordToCnc_KNN5

print("cuda available: ", torch.cuda.is_available())


def run_cross_val(split_list, args, test_set, run_dic=None):
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

    run_dic : optional, dictionary defining gridsearch the run is apart of.
    Used only to keep track of experiments TODO: make optional
    """

    empty_run = wandb.init(reinit=True,
                           project='mi-prediction')
    meta_name = empty_run.name  # unique run name for book-keeping
    empty_run.finish()

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
    for i, (train_set, val_set) in enumerate(split_list):
        # initialize wandb log to keep track of training:
        # each run is identified by name, group and job_type.
        # 'indiv-CV-run' group name prefix used for single runs
        # of cross val.
        # Each cross val runs have a numeral post-fix identifier
        run = wandb.init(reinit=True,
                            project='mi-prediction',
                            group='indiv-CV-run_'+MODEL_TYPE,
                            job_type=meta_name,
                            name=meta_name+'-'+str(i),
                            config=args)

        # print intermediate things for sanity check
        print('Cross val: ', i)
        print("Train set length: ", len(train_set))
        print("Val set length: ", len(val_set))
        print("Val patients: ", val_set.data)
        print("arguments: ", args)
        print("")
        print("run dictionary: ", run_dic)

        # set model hyperparameters
        optim_param = {
            'optimizer': args['optim'],
            'lr': args['optim_lr'],
            'momentum': args['optim_momentum'],
        }
        model_param = {
            'physics': args['physics'],
            'name': args['model'],
        }

        # initialize the model
        gnn = GNN(
            model_param,
            train_set,
            val_set,
            test_set,
            args['batch_size'],
            optim_param,
            args['weighted_loss'],
            args
        )

        # start training model!
        # logging the metrics in wandb is done in the train method
        print('Runnning!!')
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
                          project='mi-prediction',
                          group='avg-CV-val-scores_'+MODEL_TYPE,
                          name=meta_name,
                          config=args)
    # book keeping, these prints will appear in the wandb log
    print("arguments for the k runs: ", args)
    print("Test set length: ", len(test_set))
    print("Test patients: ", test_set.data)
    print("run grid: ", run_dic)

    # average over indiv runs and log metrics in wandb
    avg_dic = {}
    for key in dic_val_metrics.keys():
        avg_dic[key] = sum(dic_val_metrics[key]) / len(dic_val_metrics[key])
    meta_run.log(avg_dic)
    wandb.finish()


print("grid search starting .......")
# following file contains all hyperparameters to try in the grid search
yaml_file = open("../experiments/hyper_params.yaml")
# load hyperparameters as dictionary
arg_dic = yaml.load(yaml_file, Loader=yaml.FullLoader)
keys = arg_dic.keys()
values = arg_dic.values()
tot_length = len(list(product(*values)))  # total nb of runs to estimate time
end = time.time()
for count, instance in enumerate(product(*values)):  # loops through the grid
    print(count, " out of ", tot_length, " options in: ", time.time()-end)
    config = dict(zip(keys, instance))  # config file to use as parameters
    # split data as test set and then cross val list with
    # elements of format (train, valid)
    test_set, split_list = split_data(path=config["path_data"],
                                      num_node_feat=3,
                                      cv=True,
                                      k_cross=10,
                                      seed=config['seed'])
    run_cross_val(split_list=split_list,
                  args=config,
                  test_set=test_set,
                  run_dic=arg_dic)
    end = time.time()
