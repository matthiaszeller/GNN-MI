

from pathlib import Path


def get_data_path():
    file = Path(__file__).parent.joinpath('data-path.txt')
    with open(file, 'r') as f:
        return Path(f.read().strip('\n'))


