
# MI-proj

## About

Graph Neural Networks for Mycardial Infarction Prediction. Semester Project in [LTS4](https://www.epfl.ch/labs/lts4/) lab, EPFL.

Forked from https://github.com/jacobbamberger/MI-proj.


## Getting started

1. Create the conda environment from `environment.yml`
2. Mount the "source" data folder, i.e. the folder provided by the lab
3. Create a data folder that will store the datasets
4. Put the folder locations of the two previous steps in `src/data-path.txt`, see the "Path management" section below
5. Create the datasets with `src/create_data.py`, see the "Data" section below 

You're now ready to train models!
One must now specify the configuration in one of the `yaml` file in the `config` folder. 
One can run models in three ways as described below.

### Train single model

* Script `src/run_kfold.py`: run k-fold cross validation for a specific model
* The template configuration are 
`config/config*.yaml`, the parameters `cv.seed` and `cv.k_fold` must be specified
* Example (use `--help` argument to see the script arguments):
```shell
python src/run_kfold.py config/config.yaml <name_of_wandb_job_type>
```

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

## Path management

In the code, the data paths are retrieved with the function `get_data_path()` from `src.setup`.
The repo is designed so that two workflows are possible:
1. Execute code from local repository, the folders containing the data are mounted from a network path 
  (e.g. `//sti1files.epfl.ch/cardio-project` `//filesX.epfl.ch/data/<my-gaspar>`)
2. Execute code in remote directory

The reason is that debugging is much more convenient to perform locally, but it's more efficient to eventually 
run the code remotely (e.g. from the cluster).

Option (2) assumes that the repository is cloned in the remote directory and that the repo contains a `data` folder at the 
root of the project. Option (1) requires to add a file `data-path.txt` in `src` that contains the data locations 
(see docstring of `setup.get_data_paths()`).

For both option (1) and (2), the structure of the data folder is assumed to be:
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

The first sample is from the `CoordToCnc` dataset, the second is the same sample but from `WssToCnc`:

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


### Data creation

With current working directory (cwd) being the root of this repo, run 
```shell
python src/create_data.py
```

One can provide several arguments, e.g. the type of dataset to generate and the paths to data sources. 
Note that by default, path to data sources are automatically inferred from `get_path_data()` 
(see path management section above).


#### Example workflow for perimeter computation

Use the function `data_augmentation.compute_dataset_perimeter` to create a new folder with shortest path data.
Example:
```python
compute_dataset_perimeter('CoordToCnc', 'perimeter')
```

Then augment a dataset with perimeter features:
```python

```

#### Example workflow for KNN data augmentation

Create the dataset with both KNN and rotation augmentation once, then manually create two datasets out of it.

For instance, create the dataset `CoordToCnc`:

```shell
python src/create_data.py -n CoordToCnc -k 5
```

Then the output directory will contain 6 versions of each data point, for instance:
* `OLV049_LAD.pt`
* `OLV049_LAD_KNN5.pt`
* `OLV049_LAD_rot-18.pt`
* `OLV049_LAD_rot-09.pt`
* `OLV049_LAD_rot009.pt`
* `OLV049_LAD_rot018.pt`

Now, split this dataset into two: `CoordToCnc_KNN5` and `CoordToCnc_rot`. Follow this procedure:
1. Rename
    ```shell
    mv CoordToCnc CoordToCnc_rot
    ```
2. Copy
    ```shell
    cp -r CoordToCnc_rot CoordToCnc_KNN5
    ```
3. Remove KNN files from rotation dataset:
    ```shell
    cd CoordToCnc_rot
    rm -v *_KNN5*
    ```
4. Remove rot files from KNN dataset:
    ```shell
    cd ../CoordToCnc_KNN5
    rm -v *_rot*
    ```

## Debugging

Some scripts have tests in their `if __name__ == '__main__'` section. Make sure to run and understand those.
Specifically, the most useful for model debugging is to 
copy locally a sample (or a few samples from different datasets) and run `models.py` in a debugger. 
For instance, in Pycharm, open the `models.py`, right click on any part of the code and click "Debug models". 
If anything goes wrong, the debugger will bring you to the problematic line and you can play in the interpreter with all
 local variables to check the shapes, etc...

