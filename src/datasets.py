

import os
from typing import Union, List, Tuple

import numpy as np
import pickle5 as pickle  # TODO: is this necessary? What does this change?
import torch
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Dataset


def split_data(path, num_node_feat=3, cv=False, k_cross=10, seed=0):
    '''
    Splits the data fetched from `path` folder.
    Returns train, valid, test split, each are DataSet objects.
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
    '''
    # TODO: be convinced that this choice is the right one
    # (this ensures/facilitates that we don'T train on
    # augmented data from valid/test set.)

    with open('../data/patient_dict.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    patients = np.array(list(data_dict.keys()))  # names of patients
    patients = np.sort(patients)  # Safety, for seed to make sense

    # first isolate 10% of data for test set:
    pretrain_patients, test_patients = train_test_split(patients,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed)
    test_set = DataSet(path,
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
            split_list += [(DataSet(path,
                                    train_patients,
                                    train='train',
                                    num_node_feat=num_node_feat),
                            DataSet(path,
                                    val_patients,
                                    train='val',
                                    num_node_feat=num_node_feat))]
        return test_set, split_list
    else: # no cross val
        train_patients, val_patients = train_test_split(pretrain_patients,
                                                        test_size=0.1,
                                                        shuffle=True,
                                                        random_state=seed)
        # list of (train_set, val_set) tuples
        split_list = [(DataSet(path,
                               train_patients,
                               train='train',
                               num_node_feat=num_node_feat),
                       DataSet(path,
                               val_patients,
                               train='val',
                               num_node_feat=num_node_feat))]
        # here split list has length 1, just to imitate the cross val format
        return test_set, split_list

# TODO: delete below
# def train_test_for_eval(path, num_node_feat=3, seed=0):
#     with open('../data/patient_dict.pickle', 'rb') as f:
#         data_dict = pickle.load(f)
#     patients = np.array(list(data_dict.keys())) #names of patients TODO: Is this list ordered? Shoudl be For reproducibility.
#     patients = np.sort(patients) # for the seed thing to make sense. If the order change, the idnexing changes...

#     train_patients, test_patients = train_test_split(patients, test_size = 0.1, shuffle=True, random_state=seed)
#     test_set = DataSet(path, test_patients, train='test', num_node_feat=num_node_feat)
#     train_set = DataSet(path, train_patients, train='train', num_node_feat=num_node_feat)
#     return train_set, test_set


class DataSet(Dataset):
    def __init__(self, path, patients, train='train', num_node_feat=3):
        '''
        Creates a Dataset child object which works with Dataloaders
        for batching.

        Parameters:
        -----------
        path : string indicating where the folder in which the data is
        patients : list of strings, which are the data points to
                   consider puting in the DataSet
        train : string, either 'train', 'val' or 'test' indicating if
                the augmented data points should eb included in the
                DataSet (they are included only for training)
        num_node_feat : int, number of features oper node.
        '''
        super(DataSet, self).__init__()
        self.path = path
        self.data = []
        for name in os.listdir(self.path):
            # check if patient is in appropriate set: training, val, test
            if name[:6] not in patients:
                continue
            # the following conditionals apply only to testing/val data,
            # and makes sure the augmented data is not included in them
            if name[-4] == '1':  # and train != 'train':
                continue
            if name[-9:-6] == 'rot' and name[-4] != '0' and train != 'train':
                continue
            if name[-7:-4] == 'KNN' and train != 'train':
                continue
            if name[-9:-4] == 'NOISE' and train != 'train':
                continue
            self.data.append(name)  # name is name of file
        self.train = train
        self.data = np.sort(np.array(self.data))  # TODO: necessary?
        self.num_classes = 2  # Culprit, non-culprit
        self.num_node_feat = num_node_feat  # 3 for CoordToCnc
        self.length = len(self.data)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def num_node_features(self):
        return self.num_node_feat

    def len(self):
        # print("length is called")
        return self.length

    def get(self, idx):
        data = torch.load(os.path.join(self.path, self.data[idx]))
        return data
