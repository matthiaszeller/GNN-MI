
# MI-proj

## About

Graph Neural Networks for Mycardial Infarction Prediction. Semester Project in [LTS4](https://www.epfl.ch/labs/lts4/) lab, EPFL.

Forked from https://github.com/jacobbamberger/MI-proj.


## Getting started

1. Create the conda environment from `environment.yml`
2. Mount the "source" data folder, i.e. the folder `lts4-cardio/` provided by the lab
3. Create a data folder that will store the datasets
4. Put the folder locations of the two previous steps in `src/data-path.txt`, see the "Path management" section below
5. Create the datasets with `src/create_data.py`, see the "Data" section below 

You're now ready to train models!
One must now specify the configuration in a `json` or `yaml` file in the `config` folder,
see the "Configuration" section below. 
One can run models in three ways as described in the following section.

### Train/test single model

#### Without test metrics

* Script `src/run_cv.py`: run cross validation for a specific model
* The template configuration are 
`config/config*.yaml`, the parameters `cv.seed` and either `cv.k_fold` xor `cv.mc_splits` must be specified
* Example (use `--help` argument to see the script arguments):
```shell
python src/run_kfold.py config/config.yaml <name_of_wandb_job_type>
```

#### With test metrics

* Script `src/run_test.py`: run k-fold cross validation for a specific model and evaluate each of the k
trained models on the test set
* Specify either a configuration file, or a wandb run id (the same config will be used)

### Hyperparameter tuning

There are two modes for hyperparameter tuning.
They work with the "wandb agent" for sweeps (see the [docs](https://docs.wandb.ai/guides/sweeps)).
One doesn't directly run a script, but rather start an agent with a yaml file specifying the grid of hyperparameters.
The two modes differ in the way they explore the grid. 

An agent is started with:
```shell
wandb sweep <configuration-file>.yaml
```

#### Coarse hyperparameter tuning

This mode uses the *bayesian* search mode. Hyperparameters are specified with an a priori *distribution*,
and the agent randomly samples from a posteriori distribution with respect to some reference metric. 

The purpose of this mode is to quickly cover a large number of hyperparameter combinations,
and monitor performance with a **single validation set**, i.e. no k-fold.

See a template configuration file in `config/sweep_coarse*.yaml`. 
The agent will eventually call `src/run_sweep.py` (but don't do it yourself, this won't work).

#### Finer hyperparameter tuning

This mode should be run once the "coarse" hyperparameter tuning allowed to select plausible hyperparameters.
The goal is now to monitor performance with **k-fold cross validation**, 
so that we can assess the model variability.

See a template configuration file in `config/sweep_kfold*.yaml`.
The agent will eventually call `src/run_sweep_kfold.py`.


## Project Configuration

### Model Configuration

Each model is based either on `models.EGNN` or `models.GIN` class. 
The architectures (number of layers, number of hidden dimensions, auxiliary learning, etc...)
are dynamically generated based on the configuration parameters.

Below is the detailed explanation of each parameter.  Looking at template configurations in `config` folder 
is probably more useful, especially for sweep configurations.

* Training:
  * `allow_stop`, `early_stop`: for early stopping
  * `batch_size`
  * `epochs`
* Cross validation:
  * `cv.fold_id`: only used for "fine" hyperparameter tuning
  * `cv.valid_ratio`: for "coarse" hyperparameter tuning
  * `cv.k_fold`: number of folds for Kfold-CV
  * `cv.seed`: seed for random data splitting
  * `cv.test_reps`: deprecated
* Data:
  * `dataset.in_memory`: let it True, False is not implemented
  * `dataset.name`: to find the path of the dataset
  * `dataset.node_feat.transform`: either None or `fourier`
  * `dataset.num_graph_features`: should be 3
  * `dataset.num_node_features`: either 0 (coordinates only), 1 (e.g. perimeter of Tsvi) or 30 (Wss)
  * `dataset.sampler`: 
  * `dataset.standardize`: either None, `normalize` or `standardize`
* Model:
  * `model.desc`: textual description
  * `model.name`
  * `model.type`: either `Equiv`, `GIN`
  * `num_equiv`: number of equivariant layers, ignored for GIN model
  * `num_gin`: number of GIN layers
  * `num_hidden_dim`: number of hidden dimensions
  * `model.aux_task`: true or false
  * `model.aux_loss_weight`: float multiplying the auxiliary loss
* Loss & auxiliary loss:
  * `loss.weight`: loss weight for imbalanced classes
* Optimizer:
  * `optimizer.lr`: learning rate
  * `optimizer.momentum`
  * `optimizer.name`: should be `Adam`

### Path Management

In the code, the data paths are retrieved with the functions `get_data_path()` and `get_dataset_path()` 
from `src.setup`. 
This greatly eases path management and debugging. 

Path management relies on the file `src/data-path.txt`: it contains two lines, the
"source" data folder (containing raw data provided by the lab) and the local folder containing generated datasets.

The source data folder is assumed to point to a root with the following hierarchy:
```
.
├── CFD
│   ├── ClinicalCFD
│   │   ├── MagnitudeClinical
│   │   │   ├── ...
│   │   │   └── OLV050_RCA_WSSMag.vtp
│   │   └── VectorClinical
│   │       ├── ...
│   │       └── OLV050_RCA_WSS.vtp
│   └── labels
│       ├── ...
│       └── WSSdescriptors_AvgValues.xlsx
├── ...
```

## Data


### Data description

All models are based on the [`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)
object. Here's an example of a sample from the `CoordToCnc` and `WssToCnc` datasets:

```
Data(x=[3478, 0], edge_index=[2, 20766], y=0, coord=[3478, 3], g_x=1)
Data(x=[3478, 60], edge_index=[2, 20766], y=0, coord=[3478, 3], g_x=1)
```

Attributes are:
* `x`: node features
* `edge_index`: adjacency list, see pytorch geometric description
* `y`: label
* `coord`: (x, y, z) coordinates of each node
* `g_x`: graph features


### Dataset generation

Creating a dataset starts with the script `src/create_data.py`. 
This requires to set up the paths a priori (see Path management section). 

Example of the help output of `create_data.py`:
```shell
(gnn) root@pyt:/workspace/mynas/GNN-MI/src# python create_data.py -h
usage: create_data.py [-h]
                      [-n {CoordToCnc,WssToCnc,TsviToCnc,CoordToCnc+Tsvi}]
                      [-k AUGMENT_DATA] [-s DATA_SOURCE] [-l LABELS_SOURCE]

optional arguments:
  -h, --help            show this help message and exit
  -n {CoordToCnc,WssToCnc,TsviToCnc,CoordToCnc+Tsvi}, --dataset_name {CoordToCnc,WssToCnc,TsviToCnc,CoordToCnc+Tsvi}
                        name of dataset to be created
  -k AUGMENT_DATA, --augment_data AUGMENT_DATA
                        number of neighbours used for KNN
  -s DATA_SOURCE, --data_source DATA_SOURCE
                        path to raw data
  -l LABELS_SOURCE, --labels_source LABELS_SOURCE
                        path to data label
```
By default, `DATA_SOURCE` and `LABELS_SOURCE` do not need to be specified.

All models are based on variations of `CoordToCnc`, `WssToCnc`, `TsviToCnc`, `CoordToCnc+Tsvi`.
Specifically, datasets for auxiliary tasks require some post-processing. 
All routines are either in `src/data_augmentation.py` or `toolbox/reformat_data.py`.


#### Example workflow for perimeter computation

Use the function `data_augmentation.compute_dataset_perimeter` to create a new folder with shortest path data,
this produces a set of `.json` files. 

Example:
```python
from data_augmentation import compute_dataset_perimeter
from setup import get_dataset_path

path_in = get_dataset_path('CoordToCnc')
path_out = get_dataset_path('perimeter')
compute_dataset_perimeter(path_in, path_out)
```

Then augment a dataset with perimeter features:
```python
from data_augmentation import create_dataset_with_perimeter
from setup import get_dataset_path

path_in = get_dataset_path('CoordToCnc')
path_perim = get_dataset_path('perimeter')
path_out = get_dataset_path('CoordToCnc_perimeter')
create_dataset_with_perimeter(path_in, path_perim, path_out)
```

## Logging

All scripts perform extensive logging to track all operations and ease debugging.
The logging is performed both on the stdout (i.e. on terminal) and in a file `logs.log`.

Logging is setup in the `src/setup.py` file. The only option that a user might want to control is the logging level
(`INFO` or `DEBUG` are recommended), especially because `DEBUG` might get very verbose. This is done by changing the `level` argument of this part:
```python
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs.log"),
        logging.StreamHandler()
    ]
)
```


## Debugging

Some scripts have tests in their `if __name__ == '__main__'` section. 
Make sure to run and understand those (might require to change some paths).
Specifically, the most useful for model debugging is to 
copy locally a sample (or a few samples from different datasets) and run `models.py` in a debugger. 
For instance, in Pycharm, open the `models.py`, right click on any part of the code and click "Debug models". 
If anything goes wrong, the debugger will bring you to the problematic line and you can play in the interpreter with all
local variables to check the shapes, etc...

As a sidenote, if one wishes to run some scripts that need a wandb run instance, 
you may want to enable offline mode by running a session with the environment variable `WANDB_MODE=offline`.