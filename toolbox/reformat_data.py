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


def add_target(features: Data, feat2label: Data):
    """
    Incorporte the feature of `feat2label` as a target,
    the features are taken from `features`.
    """
    # Check they have the same label
    if isinstance(features.y, int):
        assert features.y == feat2label.y
    else:
        # Pytorch tensor
        assert (features.y == feat2label.y).all()
    # Check same number of nodes
    assert features.x.shape[0] == feat2label.x.shape[0]

    # Add labels
    new_labels = feature_to_label(feat2label)
    out = features.clone()
    out.y = new_labels.y
    return out


def concat_features(data1: Data, data2: Data) -> Data:
    out = data1.clone()
    xconcat = data2.x.clone()
    if xconcat.ndim == 1:
        xconcat = xconcat.reshape(-1, 1)

    out.x = torch.concat((out.x, xconcat), dim=1)
    return out


def dataset_apply(fun_apply: Callable, path_in: Union[str, Path],
                  path_out: Union[str, Path], path_in2: Union[str, Path] = None):
    """
    Apply some function to each element of a dataset and write in a new dataset.
    Two input datasets can be used:
        - if path_in only is provided, fun_apply must receive a single argument
        - if both path_in and path_in2 are provided, fun_apply receives two arguments
    """
    path_in, path_out = Path(path_in), Path(path_out)
    if not path_in.exists():
        raise FileNotFoundError
    if path_out.exists():
        raise ValueError
    path_out.mkdir()

    if path_in2 is not None:
        path_in2 = Path(path_in2)
        if not path_in2.exists():
            raise FileNotFoundError

    for file in path_in.glob('*.pt'):
        if path_in2 is not None:
            # Find the corresponding file in the second input dataset
            file2 = path_in2.joinpath(file.name)
            new = fun_apply(torch.load(file), torch.load(file2))
        else:
            new = fun_apply(torch.load(file))

        outpath = path_out.joinpath(file.name)
        logging.info(f'writing into {str(outpath)} from {str(file)}')
        torch.save(new, outpath)


if __name__ == '__main__':
    path = setup.get_dataset_path('WssToCnc')
    path2 = setup.get_dataset_path('CoordToCnc_perimeter')
    out = setup.get_dataset_path('WssToPerim+Cnc')
    #dataset_apply(feature_to_label, path, path.parent.joinpath('CoordToPerim+Cnc'))
    dataset_apply(add_target, path, out, path2)


