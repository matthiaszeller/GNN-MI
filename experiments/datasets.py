import numpy as np
import os
import random
import torch
import pickle5 as pickle #to work with earlier python versions (~3.6) TODO: is this necessary? What does this change?
#import pickle
from typing import Union, List, Tuple
from torch_geometric.data import Dataset
from sklearn.model_selection import train_test_split, KFold

def split_data(path, num_node_feat=1, cv=False, k_cross=10, seed=0):
    ''' Returns train, valid, test split, each are DataSet objects.
    A design choice is that the split is done at the level of patients,
    and not at the level of arteries''' 
    #TODO: be convinced that this choice is the right one (this ensures/facilitates that we don'T train on augmented data from valid/test set.)

    with open('../data/patient_dict.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    patients = np.array(list(data_dict.keys())) #names of patients TODO: Is this list ordered? Shoudl be For reproducibility.
    patients = np.sort(patients) # for the seed thing to make sense. If the order change, the idnexing changes...
    
    # first isolate 10% of data for test set:
    pretrain_patients, test_patients = train_test_split(patients, test_size = 0.1, shuffle=True, random_state=seed)
    test_set = DataSet(path, test_patients, train='test', num_node_feat=num_node_feat)
    if cv: #cross val
        kf = KFold(k_cross, shuffle=True, random_state=seed)
        split_list = []
        for train_index, val_index in kf.split(pretrain_patients):
            # then split train and validation:
            train_patients = patients[train_index]
            val_patients = patients[val_index]
            split_list += [(DataSet(path, train_patients, train='train', num_node_feat=num_node_feat), #train dataset
                            DataSet(path, val_patients, train='val', num_node_feat=num_node_feat))] #val dataset
        return test_set, split_list #TODO be convinced not to return triples (train, val test)
    else: # no cross val
        train_patients, val_patients = train_test_split(pretrain_patients, test_size = 0.1,  shuffle=True, random_state=seed)
        split_list = [(DataSet(path, train_patients, train='train', num_node_feat=num_node_feat), #train dataset
                       DataSet(path, val_patients, train='val', num_node_feat=num_node_feat))] # val set
        return test_set, split_list # split list has length 1, just to imitate the cross val format
    
def train_test_for_eval(path, num_node_feat=3, seed=0):
    with open('../data/patient_dict.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    patients = np.array(list(data_dict.keys())) #names of patients TODO: Is this list ordered? Shoudl be For reproducibility.
    patients = np.sort(patients) # for the seed thing to make sense. If the order change, the idnexing changes...

    train_patients, test_patients = train_test_split(patients, test_size = 0.1, shuffle=True, random_state=seed)
    test_set = DataSet(path, test_patients, train='test', num_node_feat=num_node_feat)
    train_set = DataSet(path, train_patients, train='train', num_node_feat=num_node_feat)
    return train_set, test_set




    # # THEN do cross validation
    # # The following block is to do the test split within the cross validation. Apprently this is not the right way.
    
    # if cv: #cross val
    #     kf = KFold(k_cross)
    #     split_list = []
    #     for train_index, pretest_index in kf.split(patients):
    #         train_patients = patients[train_index] # train_patiens as list
    #         pretest_patients = patients[pretest_index] # both test and validation for given split
    #         test_patients, val_patients = train_test_split(pretest_patients, test_size=0.5)
    #         split_list += [(DataSet(path, train_patients, train='train', num_node_feat=num_node_feat), #train dataset
    #                         DataSet(path, val_patients, train='val', num_node_feat=num_node_feat), #val dataset
    #                         DataSet(path, test_patients, train='test', num_node_feat=num_node_feat)) ] #test dataset
    #     return split_list
    # else:
    #     train_patients, pretest_patients = train_test_split(patients, test_size = 0.2)
    #     test_patients, val_patients = train_test_split(pretest_patients, test_size=0.5)
    #     return [(DataSet(path, train_patients, train='train', num_node_feat=num_node_feat),
    #             DataSet(path, val_patients, train='val', num_node_feat=num_node_feat),
    #             DataSet(path, test_patients, train='test', num_node_feat=num_node_feat) )]

class DataSet(Dataset):
    def __init__(self, path, patients, train='train', num_node_feat=3):
        '''Creates a Dataset child object which works with Dataloaders to implement batching. '''
        super(DataSet, self).__init__() # To do: See documentation        
        self.path = path

        self.data = []
        for name in os.listdir(self.path):
            if name[:6] not in patients: #check if patient is in appropriate set: training, val, test
                continue
            # the following conditionals apply only to testing/val data, and makes sure the augmented data is deleted
            if name[-4] == '1':# and train != 'train': # check if this is the augmented part of the data. If it is, and we are test or validating, then skip
                continue
            if name[-9:-6] == 'rot' and name[-4]!='0' and train != 'train': # the rotation_augmented data has format: 'CHUV01_LAD_rot000.pt' where 000 is the angle.
                continue
            if name[-7:-4] == 'KNN' and train != 'train':
                continue
            self.data.append(name) # name is name of file
        self.train=train
        self.data = np.sort(np.array(self.data)) #list of data folders .pt # Question: What happens to the order here?
        self.num_classes = 2 # Culprit, non-culprit
        self.num_node_feat = num_node_feat # 3 for CoordToCnc
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
        data = torch.load(os.path.join(self.path, self.data[idx])) # To do: Question: this creates a MyData, what instance of MyData is it? is it the one above, or the classical one?
        return data



# class DataSet(Dataset):
#     def __init__(self, path, train='train', num_node_feat=3): # cv=False, k_cross=10):
#         '''
#         creates data set, splits train test val...
#         path : path to data folder, these contain files .pt with data
#         train : string either 'train', or 'val', or 'test'
#         num_node_feat : number of feature per node in the graph (30 for the 30 time stamps)
#         cv = -1 if not cross val, uses 'seed' otherwise
#         path is path to data, the data was made by the create_data.py'''
#         super(DataSet, self).__init__() # To do: See documentation 
#         self.path = path

#         with open('../data/patient_dict.pickle', 'rb') as f:
#             data_dict = pickle.load(f)
#         patients = list(data_dict.keys()) #names of patients
#         print("There are ", len(patients), " patients")
#         # random.shuffle(patients) # THIS SHUFFLE MAKES TRAIN, TEST AND VAL OVERLAP!! Unless that is what the seed takes care of...
#         # if cv != -1: # if cross val
#         #     test_idx = int(np.ceil(0.1 * len(patients))) # 10% are test patients
#         #     test_patients = patients[:test_idx]
#         #     train_patients = patients[test_idx:] # the rest are train
#         #     nb_per_split = len(train_patients) // 10  # train_set divided in 10, of size nb_per_split
#         #     start = cv * nb_per_split # what does this do??? cv determines some number of bins
#         #     end = min(len(train_patients), (cv + 1) * nb_per_split)  # end of what? 
#         #     if train == 'test':
#         #         patients = test_patients
#         #     elif train == 'val':
#         #         patients = train_patients[start:end] # this has nb_per split elements in it
#         #     else:
#         #         patients = patients[:start] + patients[end:] #train has all remaining elements
#         # else: # no cross val
#         if train == 'test':
#             end = int(np.ceil(0.1 * len(patients))) # 10% are test patients
#             patients = patients[:end]
#         elif train == 'val':
#             start = int(np.ceil(0.1 * len(patients)))
#             end = start + int(np.ceil(0.2 * len(patients))) #%20 % are val patients
#             patients = patients[start:end]
#         else:
#             start = int(np.ceil(0.1 * len(patients))) + int(np.ceil(0.2 * len(patients))) # 70% are train_patients
#             patients = patients[start:]

#         print("Selected ", len(patients), " patients in ", train, " set")


#         self.data = []
#         for name in os.listdir(self.path): # TO DO: if data augmentation is very very big, then the val and test sets might be tiny, since we forget about the rest..?
#             if name[:6] not in patients: #check if patient is in appropriate set: training, val, test
#                 continue
#             if name[-4] == '1' and train != 'train': # check if this is the augmented part of the data. If it is, and we are test or validating, then skip
#                 continue
#             if name[-9:-6] == 'rot' and train != 'train': # the rotation_augmented data has format: 'CHUV01_LAD_rot000.pt' where 000 is the angle.
#                 continue
#             self.data.append(name) # name is name of file
#         self.data = np.sort(np.array(self.data)) #list of data folders .pt # Question: What happens to the order here?
#         self.length = len(self.data)
#         self.num_classes = 2 # Culprit, non-culprit
#         self.num_node_feat = num_node_feat # 3 for CoordToCnc

#     @property
#     def raw_file_names(self) -> Union[str, List[str], Tuple]:
#         return []

#     @property
#     def processed_file_names(self) -> Union[str, List[str], Tuple]:
#         return []

#     def _download(self):
#         pass

#     def _process(self):
#         pass

#     @property
#     def num_node_features(self):
#         return self.num_node_feat

#     def len(self):
#         # print("length is called")
#         return self.length

#     def get(self, idx):
#         data = torch.load(os.path.join(self.path, self.data[idx])) # To do: Question: this creates a MyData, what instance of MyData is it? is it the one above, or the classical one?
#         return data
