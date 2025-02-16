import json
import logging
from itertools import product
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import torch.nn
import wandb
import yaml
from torch.fft import rfft

import setup
from datasets import PatientDataset


def compute_fourier_coefs(node_feature: torch.Tensor) -> torch.Tensor:
    """
    Compute FFT of real signal. Output has same size as input.
    Since the signal is real, coefficients are only computed for positive frequencies.
    The complex FFT coefficients are decomposed into cosine coefs (ak) and sine coefs (bk), the output is:
        [a0, a1, ..., an+1, b1, ..., bn]
    """
    ck = rfft(node_feature, dim=-1)
    ak = torch.real(ck)
    bk = torch.imag(ck)[:, 1:-1]
    out = torch.concat((ak, bk), dim=-1)
    return out


def get_run_config(run_id: str) -> Dict[str, Any]:
    """
    Get a dictionnary of the wandb run configuration from the run id.
    Conveniently re-orders the fields as they appear in one template configuration file (the first file matching the
    following pattern: config/config_*.yaml)
    """
    run = get_wandb_run(run_id)
    config = parse_json_config(run)
    return config


def get_wandb_run(run_id: str) -> wandb.apis.public.Run:
    run_path = setup.get_wandb_run_path(run_id)
    api = wandb.Api()
    run = api.run(run_path)
    return run


def unnest_json_config(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """When loading wandb config with run.json_config, we get some 'value' and 'desc' fields, just get values"""
    output = {
        key: subdict['value'] for key, subdict in json_config.items()
    }
    return output


def reorder_config(json_config: Dict[str, Any], template_config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict()
    # Add the keys in order defined in template
    for key in template_config.keys():
        if key in json_config:
            out[key] = json_config[key]

    # Add keys that were not in template
    for key in json_config.keys():
        if key not in template_config:
            out[key] = json_config[key]

    return out


def parse_json_config(run: wandb.apis.public.Run) -> Dict[str, Any]:
    nested_config = json.loads(run.json_config)
    config = unnest_json_config(nested_config)

    path_config = setup.get_project_root_path().joinpath('config')
    config_files = list(path_config.glob('config_*.yaml'))
    if len(config_files) == 0:
        return config

    config_file = config_files[0]
    with open(config_file) as f:
        template_config = yaml.load(f, Loader=yaml.FullLoader)

    config = reorder_config(config, template_config)
    return config


def read_config_file(fpath: Union[str, Path]) -> Dict[str, Any]:
    fpath = Path(fpath)
    if fpath.suffix == '.yaml':
        with open(fpath, 'r') as f:
            config = yaml.load(f, yaml.FullLoader)
    elif fpath.suffix == '.json':
        with open(fpath, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError('unrecognized extension')

    return config


def display_json_config(json_config: Dict[str, Any]) -> str:
    max_keylength = max(map(str.__len__, json_config.keys()))
    desc = '\n'.join(f'- {k:<{max_keylength}} = {v}' for k, v in json_config.items())
    return desc


# ------------------------------

def get_model_num_params(model: torch.nn.Module, only_requires_grad: bool = True):
    n = 0
    for param in model.parameters():
        if only_requires_grad and param.requires_grad:
            n += param.numel()
        else:
            n += param.numel()

    return n


def describe_data_split(dataset):
    logging.info(f'{dataset.train} set, length {len(dataset)}, {dataset.patients}')


def grid_search(path_yaml_file):
    """
    Generator of configurations used for grid search of hyperparameters.
    Hyperparameters are defined in a yaml file, at root level.
    Each hyperparam can contain two types of data structures:
        - a dictionnary
        - a list, elements are separated by a dash `-`

    This generator combines elements of lists, not elements of dictionnary.
    For instance, if the content of the yaml is
    ```yaml
        model:
            type: 'Equiv'
            name: 'MyModel'
        epochs:
            - 100
            - 1000
    ```
    The generator will generate:
    """
    def gen_combinations(dic):
        """Cartesian product of nested dictionnaries"""
        # Inspired from (but this implementation is wrong...)
        # https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
        def format_item(item):
            if isinstance(item, dict):
                return gen_combinations(item)
            elif isinstance(item, list):
                return item
            return [item]

        keys, values = dic.keys(), dic.values()
        values_choices = map(format_item, values)
        for combination in product(*values_choices):
            yield dict(zip(keys, combination))

    with open(path_yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    yield from gen_combinations(data)


if __name__ == '__main__':
    run = get_wandb_run('uf34jb1p')
    config = parse_json_config(run)
