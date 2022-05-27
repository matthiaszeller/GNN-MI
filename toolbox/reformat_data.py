import logging
from pathlib import Path
from typing import Union, Callable

import torch
from torch_geometric.data import Data
import setup


def feature_to_label(data: Data) -> Data:
    """
    For a data sample having a single feature, move them to the graph labels (concat with existing graph labels)
    """
    out = data.clone()
    assert out.x.shape[-1] == 1

    if isinstance(out.y, int):
        out.y = torch.tensor([out.y])
    assert out.y.shape[-1] == 1

    # Concatenate the labels and reshape
    out.y = torch.ones(out.x.shape[0], 1) * out.y
    out.y = torch.concat((out.y, out.x), dim=1)
    # Remove the node features
    out.x = torch.empty(out.x.shape[0], 0)

    return out


def dataset_apply(fun_apply: Callable, path_in: Union[str, Path], path_out: Union[str, Path]):
    path_in, path_out = Path(path_in), Path(path_out)
    if not path_in.exists():
        raise FileNotFoundError
    if path_out.exists():
        raise ValueError
    path_out.mkdir()

    for file in path_in.glob('*.pt'):
        new = fun_apply(torch.load(file))
        outpath = path_out.joinpath(file.name)
        logging.info(f'writing into {str(outpath)} from {str(file)}')
        torch.save(new, outpath)


if __name__ == '__main__':
    path = setup.get_dataset_path('CoordToCnc_perimeters')
    dataset_apply(feature_to_label, path, path.parent.joinpath('CoordToPerim+Cnc'))
