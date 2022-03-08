import logging
import os
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
import pickle5 as pickle  # TODO: is this necessary? What does this change?
import torch
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Dataset as TorchDataset

import setup
from create_data import parse_data_file_name


def split_data(path, num_node_feat=3, cv=False, k_cross=10, seed=0, **kwargs):
    """
    Splits the data fetched from `path` folder.
    Returns train, valid, test split, each are PatientDataset objects.
    A design choice is that the split is done at the level of patients,
    and not at the level of arteries.

    Parameters:
    --------------
    path : string indicating what folder the data lives in
    num_node_feat : int indicating how many features each node has
    cv : bool to indicate if cross validation is performed
    k_cross : int, which indicates the number of folds in the k-fold
              if cv above is False, this does not impact the code
    seed : int, controls random state. Ensuring same train val and
           test split across experiments
    kwargs: passed to PatientDataset
    """
    # TODO: be convinced that this choice is the right one
    # (this ensures/facilitates that we don'T train on
    # augmented data from valid/test set.)
    _, path_output = setup.get_data_paths()

    with open(path_output.joinpath('patient_dict.pickle'), 'rb') as f:
        data_dict = pickle.load(f)
    patients = np.array(list(data_dict.keys()))  # names of patients
    patients = np.sort(patients)  # Safety, for seed to make sense

    # first isolate 10% of data for test set:
    pretrain_patients, test_patients = train_test_split(patients,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed)
    test_set = PatientDataset(path,
                              test_patients,
                              train='test',
                              num_node_feat=num_node_feat)
    if cv:  # cross val
        kf = KFold(k_cross, shuffle=True, random_state=seed)
        split_list = []
        for train_index, val_index in kf.split(pretrain_patients):
            # then split train and validation
            train_patients = patients[train_index]
            val_patients = patients[val_index]
            # list of (train_set, val_set) tuple
            split_list += [(PatientDataset(path,
                                           train_patients,
                                           train='train',
                                           num_node_feat=num_node_feat),
                            PatientDataset(path,
                                           val_patients,
                                           train='val',
                                           num_node_feat=num_node_feat))]
    else: # no cross val
        train_patients, val_patients = train_test_split(pretrain_patients,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed)
        # list of (train_set, val_set) tuples
        split_list = [(PatientDataset(path,
                                      train_patients,
                                      train='train',
                                      num_node_feat=num_node_feat),
                       PatientDataset(path,
                                      val_patients,
                                      train='val',
                                      num_node_feat=num_node_feat))]
        # here split list has length 1, just to imitate the cross val format

    logging.info(f'split_data (test_set, split_list) = ({test_set}, {split_list})')
    return test_set, split_list

# TODO: delete below
# def train_test_for_eval(path, num_node_feat=3, seed=0):
#     with open('../data/patient_dict.pickle', 'rb') as f:
#         data_dict = pickle.load(f)
#     patients = np.array(list(data_dict.keys())) #names of patients TODO: Is this list ordered? Shoudl be For reproducibility.
#     patients = np.sort(patients) # for the seed thing to make sense. If the order change, the idnexing changes...

#     train_patients, test_patients = train_test_split(patients, test_size = 0.1, shuffle=True, random_state=seed)
#     test_set = PatientDataset(path, test_patients, train='test', num_node_feat=num_node_feat)
#     train_set = PatientDataset(path, train_patients, train='train', num_node_feat=num_node_feat)
#     return train_set, test_set


class PatientDataset(TorchDataset):
    def __init__(self, path, patients, train='train', num_node_feat=3, in_memory=True):
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
        num_node_feat : int, number of features oper node.
        in_memory: bool, whether to load all the dataset into memory, otherwise load whenever needed
        """
        super(PatientDataset, self).__init__()
        self.path = path
        self.patients = []
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

        self.train = train
        self.patients = np.sort(np.array(self.patients))  # TODO: necessary?
        self.num_classes = 2  # Culprit, non-culprit
        self.num_node_feat = num_node_feat  # 3 for CoordToCnc
        self.length = len(self.patients)

        self.in_memory = in_memory
        if self.in_memory:
            logging.info(f'loading {train} patients into memory')
            self._data = [
                self._load_from_disk(i) for i in range(self.length)
            ]
        else:
            # TODO better handle this case
            logging.warning('PatientDataset.in_memory set to False, this is highly inefficient')

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
        return self.num_node_feat

    def len(self):
        return self.length

    def get(self, idx):
        if self.in_memory:
            return self._data[idx]
        return self._load_from_disk(idx)

    def _load_from_disk(self, idx: int):
        logging.debug(f'loading patient {self.patients[idx]}')
        data = torch.load(os.path.join(self.path, self.patients[idx]))
        return data

    def __repr__(self):
        return f'PatientDataset({len(self)}, type={self.train}, in_memory={self.in_memory})'


if __name__ == '__main__':
    path_in, path_out = setup.get_data_paths()
    path_dataset = path_out.joinpath('CoordToCnc_KNN5')
    test_split, split_list = split_data(path_dataset)


