import pandas as pd
import torch

import setup
from create_data import read_tsvi
from datasets import split_data, PatientDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from toolbox.wandb_export import unnest_dataframe
from sklearn.preprocessing import StandardScaler


config = {
    'dataset.name': 'CoordToCnc+Tsvi',
    'dataset.num_node_features': 0,
    'cv.seed': 0,
    'cv.k_fold': 5,
    'dataset.in_memory': True
}

test_set, split_list = split_data(path=setup.get_dataset_path(config['dataset.name']),
                                  num_node_features=config['dataset.num_node_features'],
                                  seed=config['cv.seed'],
                                  cv=True,
                                  k_cross=config['cv.k_fold'],
                                  # Args for PatientDataset
                                  in_memory=config['dataset.in_memory'])


path_data_in, _ = setup.get_data_paths()
path_labels = path_data_in.joinpath('CFD/labels/WSSdescriptors_AvgValues.xlsx')
tsvi = read_tsvi(path_labels)


def prepare_sklearn_dataset(dset: PatientDataset):
    # Extract y field
    buffer = torch.concat([e.y for e in dset])
    g = torch.concat([e.g_x for e in dset])
    # Features are artery type, tsvi
    X = torch.concat((buffer[:, [1]], g), dim=-1)
    y = buffer[:, 0].to(int)

    return X, y


metrics = []
for i, (train_set, val_set) in enumerate(split_list):
    train_x, train_y = prepare_sklearn_dataset(train_set)
    val_x, val_y = prepare_sklearn_dataset(val_set)
    test_x, test_y = prepare_sklearn_dataset(test_set)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    clf = LogisticRegression(penalty='none', random_state=0, class_weight='balanced')
    clf.fit(train_x, train_y)
    train_yhat = clf.predict(train_x)
    val_yhat = clf.predict(val_x)
    test_yhat = clf.predict(test_x)

    buffer = dict()
    buffer['train'] = classification_report(train_y, train_yhat, output_dict=True)
    buffer['val'] = classification_report(val_y, val_yhat, output_dict=True)
    buffer['test'] = classification_report(test_y, test_yhat, output_dict=True)

    train_scores = clf.predict_proba(train_x)[:, 1]
    val_scores = clf.predict_proba(val_x)[:, 1]
    test_scores = clf.predict_proba(test_x)[:, 1]

    buffer['train.1.auc'] = roc_auc_score(train_y, train_scores)
    buffer['val.1.auc'] = roc_auc_score(val_y, val_scores)
    buffer['test.1.auc'] = roc_auc_score(test_y, test_scores)

    metrics.append(buffer)


df = unnest_dataframe(pd.DataFrame(metrics))
summary = df.agg(['mean', 'std']).T
print(summary)

summary.to_csv(setup.get_project_root_path().joinpath('notebook/data/baseline.csv'))
