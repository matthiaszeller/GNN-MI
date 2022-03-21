
# MI-proj

## About

Graph Neural Networks for Mycardial Infarction Prediction. Semester Project in [LTS4](https://www.epfl.ch/labs/lts4/) lab, EPFL.

Forked from https://github.com/jacobbamberger/MI-proj.


## Setup

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

### Path management

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

For rotations, one must directly encode them in the script.

#### Recommended workflow

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

