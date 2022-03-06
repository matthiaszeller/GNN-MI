

from pathlib import Path
import logging


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs.log"),
        logging.StreamHandler()
    ]
)


def get_data_paths():
    """Return a Path object pointing to the data folders: the reading and writing paths.
    By default, assumes the data path is located at the project root, in which case reading and writing paths are the same
    If you want to override this, create a file `data-path.txt` containing the paths:
    the first list is the reading path, the second line is the writing path."""
    file_data_path = Path(__file__).parent.joinpath('data-path.txt')
    # Parse paths from file
    if file_data_path.exists():
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
    # Hard-coded project structure: data location relative to this setup.py file
    path = Path(__file__).parent.parent.joinpath('data')
    return path, path

