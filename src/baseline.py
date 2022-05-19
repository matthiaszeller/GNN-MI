import inspect
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import multiprocessing

import setup
from create_data import read_tsvi
from datasets import split_data, PatientDataset
from perimeter import get_perimeter_data
from toolbox.wandb_export import unnest_dataframe


# # Load perimeter data
# perimeter_data = dict()
# for file in setup.get_dataset_path('perimeters').glob('*.json'):
#     with open(file) as f:
#         perimeter_data[file.stem] = json.load(f)


def resample_normalized(x, y, nsample=100):
    x = (x - x.min()) / (x.max() - x.min())
    interpolator = interp1d(x, y, kind='cubic')
    grid = np.linspace(0, 1, nsample)
    return x, grid, interpolator(grid)


class Dataset:
    def __init__(self, include_tsvi: Dict = None, process_node_features: bool = False,
                 additional_graph_features: Dict[str, torch.Tensor] = None,
                 additional_graph_feature_names: List[str] = None):
        self.include_tsvi = include_tsvi
        self.process_node_features = process_node_features
        self.additional_graph_features = additional_graph_features
        self.additional_graph_feature_names = additional_graph_feature_names
        self.feature_names = []

    def process_data(self, dset: PatientDataset):
        # Artery types
        X = torch.concat([e.g_x for e in dset])
        self.feature_names = ['artery_0', 'artery_1', 'artery_2']

        if self.include_tsvi is not None:
            patients = [e.split('.')[0] for e in dset.patients]
            tsvi = torch.tensor([self.include_tsvi[p] for p in patients])
            X = torch.concat((X, tsvi.reshape(-1, 1)), dim=1)
            self.feature_names.append('tsvi')

        y = torch.tensor([e.y for e in dset])

        if self.process_node_features:
            node_features = [
                e.x for e in dset
            ]
            X = self._process_node_features(X, node_features)

        if self.additional_graph_features is not None:
            patients = [e.split('.')[0] for e in dset.patients]
            feat = [self.additional_graph_features[i].T for i in patients]

            if feat[0].ndim == 1:
                feat = torch.tensor(feat).reshape(-1, 1)
            else:
                feat = torch.concat(feat)

            X = torch.concat((X, feat), dim=1)
            self.feature_names.extend(self.additional_graph_feature_names)

        if len(self.feature_names) != X.shape[-1]:
            raise ValueError(f'only {len(self.feature_names)} feature names but data has {X.shape[-1]} features')

        X = X.numpy()
        y = y.numpy()
        return X, y

    def _process_node_features(self, X: torch.Tensor, node_features: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        if len(self.feature_names) == 0:
            raise ValueError('fit the model first to get the feature names')
        descriptor = {
            'tsvi': (self.include_tsvi is None),
            'node_feat': self.process_node_features,
            'graph_feat': (self.additional_graph_features is None),
            'feature_names': self.feature_names
        }
        return descriptor

    def __repr__(self):
        if len(self.feature_names) == 0:
            return super(Dataset, self).__repr__()
        return f'Dataset({self.feature_names})'


class DatasetPerimeterStats(Dataset):
    def __init__(self, **kwargs):
        super(DatasetPerimeterStats, self).__init__(process_node_features=True, **kwargs)

    def _process_node_features(self, X: torch.Tensor, node_features: List[torch.Tensor]) -> torch.Tensor:
        perim_stats = torch.tensor([
            [sample.min(), sample.mean(), sample.std(), sample.max()]
            for sample in node_features
        ])
        X = torch.concat((X, perim_stats), dim=1)
        self.feature_names.extend(['perim_min', 'perim_mean', 'perim_std', 'perim_max'])
        return X


class Model:
    def __init__(self, sklearn_classifier, dataset: Dataset):
        self.model = sklearn_classifier
        self.dataset = dataset
        self.verbose = False

    def pipeline(self, split_list: List[Tuple[PatientDataset, PatientDataset]], test_set: PatientDataset,
                 filter_summary=True):
        results = []
        for i, (train_set, val_set) in enumerate(split_list):
            data = [
                list(self.dataset.process_data(dset))
                for dset in (train_set, val_set, test_set)
            ]

            train_x = data[0][0]
            if self.verbose:
                print(f'training set X shape: {train_x.shape}')

            scaler = StandardScaler()
            scaler.fit(train_x)
            for i in range(3):
                data[i][0] = scaler.transform(data[i][0])

            self.model.fit(*data[0])

            # Predict
            yhats = [
                self.model.predict(x) for x, _ in data
            ]
            probas = [
                self.model.predict_proba(x)[:, 1] for x, _ in data
            ]

            # Metrics
            metrics = dict()
            for (x, y), yhat, proba, prefix in zip(data, yhats, probas, ('train', 'val', 'test')):
                metrics[prefix] = classification_report(y, yhat, output_dict=True, zero_division=0)
                metrics[f'{prefix}.1.auc'] = roc_auc_score(y, proba)

            results.append(metrics)

        df = unnest_dataframe(pd.DataFrame(results))
        summary = df.agg(['mean', 'std']).T

        if filter_summary:
            drop_columns = [
                e
                for e in summary.index
                if any(metric in e for metric in ('macro', 'support', 'weighted'))
            ]
            summary = summary.loc[summary.index.drop(drop_columns)]

        return summary

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def grid_search(self, split_list, grid: Dict, kfold: int = 5):
        # Concatenate all samples across the different splits
        # Because we use sklearn's CV features with itself performs kfold CV
        # We do this by concatenating all validation sets
        vals = []
        for _, val in split_list:
            vals.append(self.dataset.process_data(val))

        X_val, y_val = zip(*vals)
        X_train = np.concatenate(X_val)
        y_train = np.concatenate(y_val)

        X_train = (X_train - X_train.mean(0)) / X_train.std(0)

        clf = GridSearchCV(self.model, grid, cv=kfold, scoring='f1')#, n_jobs=multiprocessing.cpu_count())
        clf.fit(X_train, y_train)
        res = clf.cv_results_
        param_cols = [e for e in res.keys() if 'param_' in e]

        df = pd.DataFrame()
        for col in ('mean_test_score', 'std_test_score'):
            df[col] = res[col]

        for col in param_cols:
            df[col] = res[col]

        return df

    def describe_model(self) -> Dict[str, Any]:
        model_signature = inspect.signature(self.model.__class__)
        model_default_params = {
            k: v.default
            for k, v in model_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        # Collect non-default parameters of the instantiated model
        nondefault_params = {
            f'model_{k}': v for k, v in self.model.get_params().items()
            if model_default_params[k] != v
        }

        descriptor = dict(model_name=self.model.__class__.__name__)
        descriptor.update(nondefault_params)
        descriptor.update({
            f'data_{k}': v for k, v in self.dataset.describe().items()
        })

        return descriptor


    # def predict_proba(self, X):
    #     return self.model.predict_proba(X)[:, 1]

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    config = {
        'dataset.name': 'CoordToCnc_perimeters',
        'dataset.num_node_features': 1,
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

    # Load tsvi
    path_data_in, _ = setup.get_data_paths()
    path_labels = path_data_in.joinpath('CFD/labels/WSSdescriptors_AvgValues.xlsx')
    data_tsvi = read_tsvi(path_labels)

    path = setup.get_dataset_path('perimeters_graph_features')
    data_quadratic = torch.load(path.joinpath('graph_features.pt'))
    print(data_quadratic.keys())
    data_quadratic = data_quadratic['features']

    feat_names = ['xTLx'] + [f'xTM{i + 1}x' for i in range(next(iter(data_quadratic.values())).shape[0] - 1)]
    dset_quad_tsvi_full = Dataset(include_tsvi=data_tsvi, additional_graph_features=data_quadratic,
                                  additional_graph_feature_names=feat_names)
    logreg_quad_tsvi_full = Model(LogisticRegression(class_weight='balanced'), dset_quad_tsvi_full)

    # grid = {'C': [1e-1, 1e0, 1e1, 1e2]}
    grid = {'C': [1e3, ]}
    res = logreg_quad_tsvi_full.grid_search(split_list, grid)

    d = logreg_quad_tsvi_full.describe_model()
    a=0

#
# simple_dset = Dataset(include_tsvi=True)
# logreg = LogisticRegression()
# grid = {'C': (1e-2, 1e-1, 1e2, 1e4)}
# model = Model(logreg, simple_dset)
# res = model.grid_search(split_list, grid)
# model = Model(LogisticRegression(class_weight='balanced'), simple_dset)
# summary = model.pipeline(split_list, test_set)
# a=1
#
# # class Model2:
# #     def __init__(self, sklearn_model):
# #         self.model = sklearn_model
# #         self.include_tsvi = False
# #         self.process_node_feature = False
# #         self.process_perimeter_data = False
# #
# #     def process_dset(self, dset: PatientDataset) -> Tuple[torch.Tensor, torch.Tensor]:
# #         X = torch.concat([e.g_x for e in dset])
# #
# #         if self.include_tsvi:
# #             patients = [e.split('.')[0] for e in dset.patients]
# #             tsvi = torch.tensor([data_tsvi[p] for p in patients])
# #             X = torch.concat((X, tsvi.reshape(-1, 1)), dim=1)
# #
# #         y = torch.tensor([e.y for e in dset])
# #
# #         if self.process_node_feature:
# #             node_features = [
# #                 e.x for e in dset
# #             ]
# #             X = self._process_node_features(X, node_features)
# #
# #         if self.process_perimeter_data:
# #             X = self._process_perimeter_data(X, dset)
# #
# #         return X, y
# #
# #     def _process_node_features(self, X: torch.Tensor, node_features: List[torch.Tensor]) -> torch.Tensor:
# #         raise NotImplementedError
# #
# #     def _process_perimeter_data(self, X: torch.Tensor, artery_list: List[str]) -> torch.Tensor:
# #         raise NotImplementedError
#
#
# class Logreg(Model):
#     def __init__(self):
#         model = LogisticRegression(penalty='none', random_state=0, class_weight='balanced')
#         super(Logreg, self).__init__(model)
#
#
# class LogregTsvi(Logreg):
#     def __init__(self):
#         super(LogregTsvi, self).__init__()
#         self.include_tsvi = True
#
#
# class LogregPerimeterStatistics(Logreg):
#     """Get perimeter data from feature nodes"""
#     def __init__(self):
#         super(LogregPerimeterStatistics, self).__init__()
#         self.process_node_feature = True
#
#     def _process_node_features(self, X: torch.Tensor, node_features: List[torch.Tensor]) -> torch.Tensor:
#         perim_stats = torch.tensor([
#             [sample.min(), sample.mean(), sample.std(), sample.max()]
#             for sample in node_features
#         ])
#         X = torch.concat((X, perim_stats), dim=1)
#         return X
#
#
# class LogregTsviPerimeterStatistics(LogregPerimeterStatistics):
#     def __init__(self):
#         super(LogregTsviPerimeterStatistics, self).__init__()
#         self.include_tsvi = True
#
#
# class LogregPerimeterFreq(Logreg):
#     def __init__(self, n_freq: int, n_resampling: int = 200):
#         super(LogregPerimeterFreq, self).__init__()
#         self.process_perimeter_data = True
#         self.n_freq = n_freq
#         self.n_resampling = n_resampling
#
#     def _process_perimeter_data(self, X: torch.Tensor, dset: PatientDataset) -> torch.Tensor:
#         res = []
#         for artery_name, sample in zip(dset.patients, dset):
#             artery_name = Path(artery_name).stem
#             x, y = get_perimeter_data(perimeter_data[artery_name], sample)
#             _, grid, ytilde = resample_normalized(x, y, nsample=self.n_resampling)
#             freqs = np.fft.rfft(ytilde)[:self.n_freq]
#             res.append(freqs)
#
#         res = torch.tensor(res).to(torch.float)
#         X = torch.concat((X, res), dim=1)
#         return X
#
#
# def pipeline(model: Model, train_set, val_set, test_set, verbose: bool = False):
#     data = [
#         list(model.process_dset(dset))
#         for dset in (train_set, val_set, test_set)
#     ]
#
#     train_x = data[0][0]
#     if verbose:
#         print(f'training set X shape: {train_x.shape}')
#
#     scaler = StandardScaler()
#     scaler.fit(train_x)
#     for i in range(3):
#         data[i][0] = scaler.transform(data[i][0])
#
#     model.model.fit(*data[0])
#
#     # Predict
#     yhats = [
#         model.model.predict(x) for x, _ in data
#     ]
#     probas = [
#         model.model.predict_proba(x)[:, 1] for x, _ in data
#     ]
#
#     # Metrics
#     metrics = dict()
#     for (x, y), yhat, proba, prefix in zip(data, yhats, probas, ('train', 'val', 'test')):
#         metrics[prefix] = classification_report(y, yhat, output_dict=True)
#         metrics[f'{prefix}.1.auc'] = roc_auc_score(y, proba)
#
#     return metrics
#
#
# metrics = []
# models = [
#     ('logistic regression', Logreg()),
#     ('logistic regression with tsvi', LogregTsvi()),
#     ('logistic regression with perimeter stats', LogregPerimeterStatistics()),
#     ('logistic regression with tsvi & perimeter stats', LogregTsviPerimeterStatistics()),
#     ('', LogregPerimeterFreq(n_freq=10)),
#     ('', LogregPerimeterFreq(n_freq=15)),
#     ('', LogregPerimeterFreq(n_freq=20)),
# ]
#
#
# dfs = []
# for desc, model in models:
#     buffer = []
#     print(f'\n\nLaunching model {desc}\n')
#     for i, (train_set, val_set) in enumerate(split_list):
#         results = pipeline(model, train_set, val_set, test_set, verbose=i == 0)
#         buffer.append(results)
#     # Aggregate folds
#     df = unnest_dataframe(pd.DataFrame(buffer))
#     summary = df.agg(['mean', 'std']).T
#     print(summary)
#     dfs.append(summary)
#

# for i, (train_set, val_set) in enumerate(split_list):
#     train_x, train_y = prepare_sklearn_dataset(train_set)
#     val_x, val_y = prepare_sklearn_dataset(val_set)
#     test_x, test_y = prepare_sklearn_dataset(test_set)
#
#     scaler = StandardScaler()
#     scaler.fit(train_x)
#     train_x = scaler.transform(train_x)
#     val_x = scaler.transform(val_x)
#     test_x = scaler.transform(test_x)
#
#     clf = LogisticRegression(penalty='none', random_state=0, class_weight='balanced')
#     clf.fit(train_x, train_y)
#     train_yhat = clf.predict(train_x)
#     val_yhat = clf.predict(val_x)
#     test_yhat = clf.predict(test_x)
#
#     buffer = dict()
#     buffer['train'] = classification_report(train_y, train_yhat, output_dict=True)
#     buffer['val'] = classification_report(val_y, val_yhat, output_dict=True)
#     buffer['test'] = classification_report(test_y, test_yhat, output_dict=True)
#
#     train_scores = clf.predict_proba(train_x)[:, 1]
#     val_scores = clf.predict_proba(val_x)[:, 1]
#     test_scores = clf.predict_proba(test_x)[:, 1]
#
#     buffer['train.1.auc'] = roc_auc_score(train_y, train_scores)
#     buffer['val.1.auc'] = roc_auc_score(val_y, val_scores)
#     buffer['test.1.auc'] = roc_auc_score(test_y, test_scores)
#
#     metrics.append(buffer)
#
#
# df = unnest_dataframe(pd.DataFrame(metrics))
# summary = df.agg(['mean', 'std']).T
# print(summary)
#
# summary.to_csv(setup.get_project_root_path().joinpath('notebook/data/baseline.csv'))
