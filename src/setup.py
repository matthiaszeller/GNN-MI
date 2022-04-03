import logging
from functools import wraps
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs.log"),
        logging.StreamHandler()
    ]
)


CONFIG_DEFAULT_K_FOLD = 5
CONFIG_DEFAULT_TEST_REPS = 10


def get_project_root_path():
    # Hard-coded project structure: data location relative to this setup.py file
    return Path(__file__).parent.parent


def get_data_paths(force_local=False):
    """Return a Path object pointing to the data folders: the reading and writing paths.
    By default, assumes the data path is located at the project root, in which case reading and writing paths are the same
    If you want to override this, create a file `data-path.txt` containing the paths:
    the first list is the reading path, the second line is the writing path."""
    file_data_path = Path(__file__).parent.joinpath('data-path.txt')
    # Parse paths from file
    if force_local is False and file_data_path.exists():
        with open(file_data_path, 'r') as f:
            paths = f.read().strip('\n').split('\n')

        if len(paths) != 2:
            raise ValueError('The data-path.txt file is incorrectly formatted, see docstring of setup.get_data_paths()')

        path_in, path_out = tuple(map(Path, paths))
        if not path_in.exists() or not path_out.exists():
            raise ValueError('One of the data paths prescribed in `data-path.txt` does not exist')

        logging.info('get_data_path(): using user-defined data path')
        return path_in, path_out

    logging.info('get_data_path(): using default data folder')
    path = get_project_root_path().joinpath('data')
    return path, path


def get_dataset_path(dataset_name: str):
    _, path = get_data_paths()
    return path.joinpath(dataset_name)


def get_local_data_path():
    path = get_project_root_path().joinpath('data')
    if not path.exists():
        logging.info(f'creating local data path at {str(path)}')
        path.mkdir()

    return path


def arg_logger(fun):
    # Based on https://stackoverflow.com/questions/23983150/how-can-i-log-a-functions-arguments-in-a-reusable-way-in-python
    # Possible improvements with https://gist.github.com/DarwinAwardWinner/1170921
    @wraps(fun)
    def inner(*args, **kwargs):
        logging.debug(f'calling {fun.__name__} with args {args}, kwargs {kwargs}')
        fun(*args, **kwargs)

    return inner


# Wandb parameters
# Follow procedure on this webpage https://wandb.ai/quickstart/pytorch
# and set the `project` and `entity` fields below
WANDB_PATH = get_project_root_path().joinpath('data/wandb_auto')
if not WANDB_PATH.exists():
    WANDB_PATH.mkdir()

WANDB_SETTINGS = {
    'project': 'egnn-mi',
    'entity': 'mazeller',
    # the directory handled by wandb itself, distinct from parsed data
    'dir': WANDB_PATH
}


if __name__ == '__main__':
    logging.info('notebook setup...')
    path_src = get_project_root_path().joinpath('src')

    import sys
    sys.path.insert(0, str(path_src.absolute()))

    logging.info('added src path to sys.path')
