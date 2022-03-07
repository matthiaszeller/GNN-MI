from itertools import product

import yaml


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
