import logging
from itertools import product
from typing import List, Tuple, Dict, Any

import torch.nn
import yaml

from datasets import PatientDataset


def unnest_json_config(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """When loading wandb config with run.json_config, we get some 'value' and 'desc' fields, just get values"""
    output = {
        key: subdict['value'] for key, subdict in json_config.items()
    }
    return output


def display_json_config(json_config: Dict[str, Any]) -> str:
    max_keylength = max(map(str.__len__, json_config.keys()))
    desc = '\n'.join(f'- {k:<{max_keylength}} = {v}' for k, v in json_config.items())
    return desc


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
    file = '../experiments/hyper_params.yaml'
    for combination in grid_search(file):
        print(combination)
