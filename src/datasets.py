

import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Dataset as TorchDataset

import setup
from create_data import parse_data_file_name, load_patient_dict


def split_data(path, num_node_features, seed, cv=False, k_cross=10, valid_ratio=0.3, **kwargs):
    """
    Splits the data fetched from `path` folder.
    Returns train, valid, test split, each are PatientDataset objects.
    A design choice is that the split is done at the level of patients,
    and not at the level of arteries.

    Parameters:
    --------------
    path : string indicating what folder the data lives in
    num_node_features : int indicating how many features each node has
    cv : bool to indicate if cross validation is performed
    k_cross : int, which indicates the number of folds in the k-fold
              if cv above is False, this does not impact the code
    seed : int, controls random state. Ensuring same train val and
           test split across experiments
    kwargs: passed to PatientDataset
    """
    # (this ensures/facilitates that we don'T train on
    # augmented data from valid/test set.)
    patient_dict = load_patient_dict()
    patients = np.array(list(patient_dict.keys()))  # names of patients
    patients = np.sort(patients)  # Safety, for seed to make sense

    # first isolate 10% of data for test set:
    pretrain_patients, test_patients = train_test_split(patients,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed)
    test_set = PatientDataset(path,
                              test_patients,
                              train='test',
                              num_node_features=num_node_features,
                              **kwargs)
    if cv:  # cross val
        kf = KFold(k_cross, shuffle=True, random_state=seed)
        split_list = []
        for train_index, val_index in kf.split(pretrain_patients):
            # then split train and validation
            train_patients = pretrain_patients[train_index]
            val_patients = pretrain_patients[val_index]
            # list of (train_set, val_set) tuple
            split_list += [(PatientDataset(path,
                                           train_patients,
                                           train='train',
                                           num_node_features=num_node_features,
                                           **kwargs),
                            PatientDataset(path,
                                           val_patients,
                                           train='val',
                                           num_node_features=num_node_features,
                                           **kwargs))]
    else: # no cross val
        train_patients, val_patients = train_test_split(pretrain_patients,
                                                        test_size=valid_ratio,
                                                        shuffle=True,
                                                        random_state=seed)
        # list of (train_set, val_set) tuples
        split_list = [(PatientDataset(path,
                                      train_patients,
                                      train='train',
                                      num_node_features=num_node_features,
                                      **kwargs),
                       PatientDataset(path,
                                      val_patients,
                                      train='val',
                                      num_node_features=num_node_features,
                                      **kwargs))]
        # here split list has length 1, just to imitate the cross val format

    logging.info(f'split_data (test_set, split_list) = ({test_set}, {split_list})')
    check_splits(test_set, split_list)
    return test_set, split_list


class PatientDataset(TorchDataset):
    def __init__(self, path, patients, train='train', num_node_features=3, in_memory=True,
                 exclude_files: List[str] = None, node_feat_transform: str = None):
        """
        Creates a PatientDataset child object which works with Dataloaders
        for batching.

        Parameters:
        -----------
        path : string indicating where the folder in which the data is
        patients : list of strings, which are the patient ids to
                   consider putting in the PatientDataset
        train : string, either 'train', 'val' or 'test' indicating if
                the augmented data points should eb included in the
                PatientDataset (they are included only for training)
        num_node_features : int, number of features per node.
        in_memory: bool, whether to load all the dataset into memory, otherwise load whenever needed
        exclude_files: list of file names to exclude from the dataset
        node_feat_transform: None or "fourier"
        """
        super(PatientDataset, self).__init__()
        self.path = path
        self.patients = []

        patients = set(patients)
        for name in os.listdir(self.path):
            is_patient, patient_id, is_augmented = parse_data_file_name(name)
            if is_patient is False:
                continue
            # check if patient is in appropriate set: training, val, test
            # the following conditionals apply only to testing/val data,
            # and makes sure the augmented data is not included in them
            if is_augmented and train != 'train':
                continue
            # Trying to enforce default-deny
            # TODO: this is still some bad design, but trying to make things work and ensure backward compatibility
            if patient_id in patients:
                self.patients.append(name)  # name is name of file

        if len(self.patients) == 0:
            # Happens if val/test set and dataset contains only augmented data
            if train == 'train':
                raise ValueError('unexpected state')

            logging.warning(f'dataset contains only augmented data, they will be included in the {train} set')
            for name in os.listdir(self.path):
                is_patient, patient_id, is_augmented = parse_data_file_name(name)
                if is_patient is False:
                    continue
                if patient_id in patients:
                    self.patients.append(name)

        # Exclude
        if exclude_files is not None:
            current_files = [Path(f).stem for f in self.patients]
            exclude_files = [Path(f).stem for f in exclude_files]
            to_remove = set(current_files).intersection(exclude_files)
            if len(to_remove) > 0:
                current_files = set(current_files).difference(to_remove)
                self.patients = [f'{f}.pt' for f in current_files]
                logging.info(f'removed following files from {train} set: {list(to_remove)}')
                
        self.train = train
        self.patients = sorted(self.patients)  # TODO: necessary?
        self.num_classes = 2  # Culprit, non-culprit
        self._num_node_features = num_node_features # 0 for CoordToCnc, 3 for CoordToCnc
        self.length = len(self.patients)
        self._std_data = None
        self._norm_data = []

        self.in_memory = in_memory
        if self.in_memory:
            logging.debug(f'loading {train} patients into memory')
            self._data = [
                self._load_from_disk(i) for i in range(self.length)
            ]
            # TODO if in_memory False, this sanity check isn't done anywhere else and error is really hard to debug
            sample_feat = self._data[0].x
            if (sample_feat.ndim > 1 and sample_feat.shape[-1] != self.num_node_features) \
               or (sample_feat.ndim == 1 and self.num_node_features != 1):
                raise ValueError(f'data has last dimension of size {sample_feat.shape[-1]} '
                                 f'but num feature is {self.num_node_features}')
        else:
            # Because of data standardization, just don't handle it in the end
            raise NotImplementedError
            # TODO better handle this case
            # TODO chekc above comment about sanity check and num_node_features
            logging.warning('PatientDataset.in_memory set to False, this is highly inefficient')

        self._node_feat_transform = node_feat_transform
        self.transform_node_feat()

    def transform_node_feat(self):
        if self._node_feat_transform is None:
            return
        elif self._node_feat_transform != 'fourier':
            raise ValueError('only Fourier transform is supported')

        from utils import compute_fourier_coefs
        logging.info(f'transforming node features of {self.train} set with Fourier')
        for e in self._data:
            e.x = compute_fourier_coefs(e.x)

    def get_weighted_sampler(self, criterion: str = 'artery'):
        """
        Get a weighted sampler. Criterion is arety or label.
        """
        if criterion == 'label':
            counts = Counter([
                data.y for data in self._data
            ])
            weights = 1 / np.array([counts[0], counts[1]], dtype=float)
            samples_weight = np.array([weights[data.y] for data in self._data])
        elif criterion == 'artery':
            # Concatenate
            g_xs = torch.concat([
                data.g_x for data in self._data
            ])
            counts = g_xs.sum(dim=0)
            weights = 1 / counts
            samples_weight = torch.tensor([
                weights[e.g_x.to(bool).ravel()] for e in self._data
            ])
        else:
            raise ValueError

        logging.info(f'weights for {self.train} set: {weights}')
        return WeightedRandomSampler(samples_weight, len(samples_weight))

    def normalize(self):
        """Sample-wise normalization, i.e. for each sample, perform min-max normalization across all features"""
        logging.info(f'normalization of {self.train} set...')
        for sample in self._data:
            if sample.x.numel() == 0:
                continue

            min_ = sample.x.min()
            max_ = sample.x.max()
            sample.x = (sample.x - min_) / (max_ - min_)
            self._norm_data.append((min_, max_))

    def standardize(self, mean: float = None, std: float = None, restore: bool = False):
        """
        Standardize the data, two modes:
        - mean and std is given, use those to standardize (typically for val/test sets)
        - no input is given, standardize with `self` values and return mean, std (typically done for train set),
          in this case, statistics are computed on concatenated node features of all nodes from all graphs
        """
        if (mean is None) != (std is None):
            raise ValueError

        # Check if data was already standardized:
        if self._std_data is not None:
            if restore is False:
                raise ValueError('data already standardized, use restore=True if want to standardize again')
            logging.info('restoring original data before standardizing again')
            old_mean, old_std = self._std_data
            for sample in self._data:
                sample.x = sample.x * old_std + old_mean

        return_stats = False
        if mean is None:
            return_stats = True
            # Concatenate all data points
            data = torch.concat([d.x for d in self._data])
            # Compute self statistics
            mean = data.mean(dim=0)
            std = data.std(dim=0)

        # Apply
        logging.info(f'standardization of {self.train} set...')
        for sample in self._data:
            sample.x = (sample.x - mean) / std

        # Book-keeping
        self._std_data = (mean, std)

        if return_stats:
            return mean, std

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def download(self):
        pass

    def _download(self):
        pass

    def process(self):
        pass

    def _process(self):
        pass

    @property
    def num_node_features(self):
        return self._num_node_features

    def len(self):
        return self.length

    def get(self, idx):
        if self.in_memory:
            return self._data[idx]
        return self._load_from_disk(idx)

    def _load_from_disk(self, idx: int, remove_edge_weight: bool = True):
        data = torch.load(os.path.join(self.path, self.patients[idx]))
        if remove_edge_weight and 'edge_weight' in data.keys:
            del data.edge_weight

        if data.x.ndim == 1:
            data.x = data.x.unsqueeze(-1)

        return data

    def get_patient(self, patient: str):
        idx = self.patients.index(patient)
        return self.get(idx)

    def __repr__(self):
        transfo = '' if self._node_feat_transform is None else f', transform={self._node_feat_transform}'
        return f'PatientDataset({len(self)}, type={self.train}, in_memory={self.in_memory}{transfo})'


def check_splits(test_split: PatientDataset, split_list: List[Tuple[PatientDataset, PatientDataset]]):
    """
    Perform sanity checks on data splits. Checks that:
        - for a given train/val split, there's no train-val overlap AND none of them overlaps with test set
    """
    logging.info('sanity checking data splits')
    # Check test data is not in train nor val
    test_patients = test_split.patients
    oks = True

    val_patients_across_splits = set()
    train_patients_across_splits = set()
    for i, (train, val) in enumerate(split_list):
        train = train.patients
        val = val.patients
        inter_trainval = set(train).intersection(val)
        inter_testtrain = set(test_patients).intersection(train)
        inter_testval = set(test_patients).intersection(val)
        inter_all = inter_trainval.intersection(test_patients)

        intersections = (
            ('train-val', inter_trainval),
            ('test-train', inter_testtrain),
            ('test-val', inter_testval),
            ('train-val-test', inter_all)
        )
        ok = all(len(inter) == 0 for _, inter in intersections)
        oks = ok and oks
        if not ok:
            logging.error(f'data split error in split {i+1}/{len(split_list)}')
            for desc, inter in intersections:
                logging.error(f'intersection {desc}, num={len(inter)}, {inter}')
        else:
            logging.debug(f'data split {i+1}/{len(split_list)} - no error detected')

        val_patients_across_splits = val_patients_across_splits.union(val)
        train_patients_across_splits = train_patients_across_splits.union(train)

    # Check across splits
    # IMPORTANT NOTE: I'm letting this code because it's not obvious at first that the
    #                 test below *should* actually fail: because of augmented data
    # TODO: check that union of train sets is union of val sets AFTER filtering augmented data
    # if len(val_patients_across_splits) != len(train_patients_across_splits):
    #     oks = False
    #     logging.error(f'union of train sets (length {len(train_patients_across_splits)})'
    #                   f' is not the union of validation sets (length {len(val_patients_across_splits)})')

    if not oks:
        raise ValueError


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader

    path_dataset = setup.get_dataset_path('TsviToCnc')

    test_split, ((train, val), ) = split_data(path_dataset, num_node_features=1, seed=0, valid_ratio=0.2,
                                              exclude_files=['OLV046_LCX'],
                                              )#node_feat_transform='fourier')

    #check_splits(test_split, split_list)

    sampler = train.get_weighted_sampler('artery')
    train_loader = DataLoader(train, batch_size=8, sampler=sampler)
    # Check empirically statistics with weighted sampler
    ys, g_xs = [], []
    for _ in range(40):
        for b in train_loader:
            ys.append(b.y)
            g_xs.append(b.g_x)
    ys = torch.concat(ys)
    g_xs = torch.concat(g_xs)

    mean_ys = ys.to(float).mean() # should be ~ 50% if get_weighted_sampler('label')
    mean_gx = g_xs.mean(dim=0)    # should be ~ [1/3,1/3,1/3] if get_weighted_sampler('artery')
    a = 1
