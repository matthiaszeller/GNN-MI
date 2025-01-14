import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, OrderedDict

import numpy as np
import torch
import wandb
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.loader import DataLoader

import setup
from datasets import PatientDataset
from models import EGNN, GNNBase, checkpoint_model, GIN_GNN, Mastered_EGCL, EGNNMastered
from utils import get_model_num_params
from torch_scatter import scatter_mean


# class MSELoss(torch.nn.Module):
#     def __init__(self, nodewise_loss: bool = False):
#         super(MSELoss, self).__init__()
#         self.nodewise = nodewise_loss
#
#     def forward(self, yhat: torch.Tensor, ytrue: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
#         # Row-wise squared differences
#         diff = ((yhat - ytrue) ** 2).sum(-1)
#         if not self.nodewise:
#             return diff.mean()
#
#         # Batch-wise mean
#         mses = scatter_mean(diff, batch)
#         return mses


def process_y_batch_auxiliary(y: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y, y_aux = y[:, 0], y[:, 1:]

    batch_unique = torch.unique(batch)
    batch_size = len(batch_unique)

    if batch_size != y.shape[-1]:
        # Main task
        y_main = []
        for b in batch_unique:
            mask = batch == b
            y_sample = y[mask]
            assert len(torch.unique(y_sample)) == 1
            y_main.append(y_sample[0])
        y_main = torch.tensor(y_main, device=y.device)
    else:
        y_main = y

    # Aux task
    return y_main, y_aux



class GNN:
    """
    Base class that contains the model, and information about the model.
    Currently this covers only the "NoPhysicsGnn" and "Equiv" models.

    Performs the following steps:
        - initialize torch.device
        - instantiate data loaders
        - instantiate model and move to device
        - instantiate optimizer
        - instantiate criterion

    For auxiliary learning, the following structure was chosen:
        - graph-level prediction: the y tensor (labels) for 1 sample has shape 1 x (1 + num_aux_features)
          the +1 accounts for the main task label
        - node-level prediction: the y tensor for 1 sample has shape n_nodes x (1 + num_aux_features)
          the +1 accounts for the main task label, duplicated n_nodes times (just easier implementation)
    """
    model: GNNBase

    def __init__(
            self,
            config: Dict[str, Any],
            train_set: PatientDataset,
            valid_set: PatientDataset,
            test_set: Union[PatientDataset, None] = None,
            model_save_path: Union[str, Path] = None
    ):
        """
        :param config:
        :param train_set:
        :param valid_set:
        :param test_set:
        :param model_save_path:
        """
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(dev)
        logging.info(f'Using device: {self.device}')
        if dev != 'cuda':
            logging.warning('running on CPU, do you really want to do this !?')

        self.model_type = config['model.type']
        self.model_save_path = model_save_path
        if self.model_save_path is None:
            _, p = setup.get_data_paths(path_out_only=True)
            self.model_save_path = p.joinpath('models')

        self.ratio = 1.0  # only useful for phys models

        self.auxiliary_task = config.get('model.aux_task', False)
        self.auxiliary_task_nodewise = False
        if self.auxiliary_task:
            self.aux_loss_weight = config['model.aux_loss_weight']
            # Sanity check
            if isinstance(train_set[0].y, int) or train_set[0].y.shape[-1] == 1:
                raise ValueError('auxiliary learning mode but y has size 1')
            # Is the aux task node wise or graph wise ?
            if train_set[0].y.shape[0] > 1:
                self.auxiliary_task_nodewise = True
        else:
            self.aux_loss_weight = 0.0

        if self.model_type == 'NoPhysicsGnn':
            raise NotImplementedError
            self.physics = False
            self.automatic_update = False
            self.model = NoPhysicsGnn(train_set)
        elif self.model_type == 'Equiv':
            self.physics = False
            self.automatic_update = False
            self.model = EGNN(num_classes=train_set.num_classes,
                              num_hidden_dim=config['num_hidden_dim'],
                              num_graph_features=config['dataset.num_graph_features'],
                              num_node_features=train_set.num_node_features,
                              num_equiv=config['num_equiv'],
                              num_gin=config['num_gin'],
                              auxiliary_task=self.auxiliary_task,
                              auxiliary_nodewise=self.auxiliary_task_nodewise)
        elif self.model_type == 'EquivMastered':
            self.physics = False
            self.automatic_update = False
            self.model = EGNNMastered(num_classes=train_set.num_classes,
                                      num_hidden_dim=config['num_hidden_dim'],
                                      num_graph_features=config['dataset.num_graph_features'],
                                      num_node_features=train_set.num_node_features,
                                      num_equiv=config['num_equiv'],
                                      num_gin=config['num_gin'])
        elif self.model_type == 'GIN':
            self.physics = False
            self.automatic_update = False
            self.model = GIN_GNN(num_classes=train_set.num_classes,
                                 num_hidden_dim=config['num_hidden_dim'],
                                 num_graph_features=config['dataset.num_graph_features'],
                                 num_node_features=train_set.num_node_features,
                                 num_gin=config['num_gin'],
                                 use_coords=config.get('model.use_coords', True))
        else:
            raise ValueError('unrecognized model type')

        logging.info(f'\nUsing model:\n{self.model}')
        logging.info(f'number of params: {get_model_num_params(self.model)}')
        self.model.to(self.device)

        # Debugging
        logging.info(f'Samples from train, val, test sets:\n{train_set[0]}\n{valid_set[0]}\n'
                      f'{test_set[0] if test_set is not None else ""}')
        # initialize data loader for batching
        sample = config.get('dataset.sampler', None)
        if sample is not None:
            logging.info(f'using sampler to balance {sample}')
            train_sampler = train_set.get_weighted_sampler(criterion=sample)
        else:
            train_sampler = None

        train_shuffle = True if train_sampler is None else None
        self.train_loader = DataLoader(train_set, config['batch_size'], shuffle=train_shuffle,
                                       sampler=train_sampler, drop_last=True)
        self.val_loader = DataLoader(valid_set, config['batch_size'], shuffle=False)
        self.test_loader = DataLoader(test_set, config['batch_size'], shuffle=False) if test_set is not None else None

        if config['optimizer.name'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=config['optimizer.lr'])
        elif config['optimizer.name'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=config['optimizer.lr'],
                                             momentum=config['optimizer.momentum'])

        weights = torch.tensor([1 - config['loss.weight'], config['loss.weight']])  # for class imbalance

        self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        if self.auxiliary_task:
            self.aux_criterion = torch.nn.MSELoss().to(self.device)#MSELoss(nodewise_loss=self.auxiliary_task_nodewise).to(self.device)

        self.epoch = None

        # --- Data standardization
        std = config.get('dataset.standardize')
        if std is not None:
            if std == 'standardize':
                self.standardize_node_features()
            elif std == 'normalize':
                self.normalize_node_features()
            else:
                raise ValueError
        else:
            if train_set.num_node_features > 0:
                logging.warning('node feature are not normalized')

    def normalize_node_features(self):
        self.train_loader.dataset.normalize()
        self.val_loader.dataset.normalize()
        if self.test_loader is not None:
            self.test_loader.dataset.normalize()
        logging.info('normalized node features')

    def standardize_node_features(self):
        if self.train_loader.dataset.num_node_features == 0:
            return

        logging.info('standardizing nodes features...')
        train_mean, train_std = self.train_loader.dataset.standardize()
        logging.info(f'training means and stds:\n{train_mean}\n{train_std}')
        # Apply standardization based on training values
        self.val_loader.dataset.standardize(train_mean, train_std)
        if self.test_loader is not None:
            self.test_loader.dataset.standardize(train_mean, train_std, restore=True)

    def save_model(self, state_dic: OrderedDict, optimizer_dic: Dict, epoch: int,
                   validation_metrics: Dict, run: wandb.sdk.wandb_run.Run):
        """
        Save model in `state_dic` locally and in wandb.
        One must provide state_dic because of early stopping: the current model is not necessarily
        the one we want to save.
        """
        if not self.model_save_path.exists():
            self.model_save_path.mkdir()

        file_name = f'model-{run.id}.pt'
        file_path = self.model_save_path.joinpath(file_name)
        # Save file locally
        checkpoint_model(
            path=file_path,
            model_dict=state_dic,
            optimizer_dict=optimizer_dic,
            epoch=epoch,
            metrics=validation_metrics
        )
        # Save file in wandb
        # run.save(file_path)

    def save_predictions(self, run):
        def data_loader_iterator(dataset: PatientDataset):
            preds, preds_aux = [], []
            loader = DataLoader(dataset, shuffle=False, batch_size=self.train_loader.batch_size)
            with torch.no_grad():
                for data in loader:
                    data = data.to(self.device)
                    pred = self.model(data.x, data.coord, data.g_x, data.edge_index, data.batch)
                    if self.auxiliary_task:
                        pred, pred_aux = pred
                    else:
                        pred_aux = torch.empty((0, 0))
                    preds.append(pred.detach().cpu().numpy())
                    preds_aux.append(pred_aux.detach().cpu().numpy())

            preds = np.concatenate(preds)
            preds_aux = np.concatenate(preds_aux)
            output = []
            for i in range(len(dataset)):
                sample = dataset.get(i)
                y = sample.y
                # TODO: this thing is a huge mess
                if self.auxiliary_task:
                    if self.auxiliary_task_nodewise:
                        y_aux = y[:, 1:]
                        y = torch.unique(y[:, 0])
                    else:
                        y = y.squeeze()
                        y, y_aux = y[0], y[1:]
                else:
                    if isinstance(y, torch.Tensor):
                        y = y.squeeze()
                    y_aux = torch.Tensor([np.nan])

                if isinstance(y, torch.Tensor):
                    y = y.item()
                if y_aux.ndim == 2:
                    y_aux = y_aux.tolist()
                else:
                    y_aux = y_aux.item()
                output.append({
                    'file': dataset.patients[i],
                    'type': dataset.train,
                    'g_x': sample.g_x.tolist(),
                    'pred': preds[i].tolist(),
                    'pred_aux': preds_aux[i].item() if self.auxiliary_task else np.nan,
                    'y': y,
                    'y_aux': y_aux
                })

            return output

        preds = []
        self.model.eval()
        preds.extend(data_loader_iterator(self.train_loader.dataset))
        preds.extend(data_loader_iterator(self.val_loader.dataset))
        if self.test_loader is not None:
            preds.extend(data_loader_iterator(self.test_loader.dataset))

        save_path = self.model_save_path.joinpath(f'pred-{run.id}.json')
        logging.info(f'saving predictions in file {str(save_path)}')
        with open(save_path, 'w') as f:
            json.dump(preds, f, indent=4)

        #wandb.save(str(save_path.absolute()))
        #run.save(str(save_path.absolute()), policy='end')

        return preds

    @staticmethod
    def calculate_binary_classif_metrics(y_pred, y_true, y_score) -> Dict[str, float]:
        """
        Calculates several metrics to evaluate model.

        Parameters
        -----------
        y_pred : np.array with predictions of model
        y_true : np.array with labels
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report['1']['auc'] = roc_auc_score(y_true, y_score[:, 1])
        report['0']['auc'] = roc_auc_score(y_true, y_score[:, 0])
        return report

    def get_losses(self, data: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assumes the data.y is 2D tensor: 1st dim for batch, 2nd dim: elem 0 is y_classif, elems 1: are for regression
        Returns:
            - loss
            - auxiliary loss for regress        (may be empty)
            - yhat for classif
            - yhat for regress                  (may be empty)
            - true y for classif
            - true y for regress                (may be empty)
        """
        if self.physics:
            raise ValueError("!! Wrong argument: self.physics set to true. Not supported in this project.")

        if self.auxiliary_task:
            yhat, y_regr_hat = self.model(data.x, data.coord, data.g_x, data.edge_index, data.batch)

            y, y_regr = process_y_batch_auxiliary(data.y, data.batch)
            # y, y_regr = data.y[:, 0], data.y[:, 1:]
            # # Check if node-level or graph-level auxiliary task
            # if data.y.shape[0] != self.train_loader.batch_size:
            #     # Then the main label is duplicated for each node
            #

            # Need to convert y back to int
            y = y.to(torch.long)
            loss = self.criterion(yhat, y)
            aux_loss = self.aux_criterion(y_regr_hat, y_regr)
            return loss, aux_loss, yhat, y_regr_hat, y, y_regr

        yhat = self.model(data.x, data.coord, data.g_x, data.edge_index, data.batch)
        loss = self.criterion(yhat, data.y)
        dummy_aux_loss = torch.tensor([0.0], requires_grad=True, device=loss.device)
        return loss, dummy_aux_loss, yhat, torch.empty((0, 0)), data.y, torch.empty((0, 0))

    def train(self, epochs: int, early_stop: int, allow_stop: int = 200,
              run: wandb.sdk.wandb_run.Run = wandb):
        self.model.train()
        epochs_no_improve = 0
        min_val_loss = 1e8
        metrics, val_metrics = None, None
        last_best_model, last_best_optimizer = None, None
        last_best_val_metrics, last_best_train_metrics = None, None
        for epoch_idx in range(1, epochs + 1):
            self.epoch = epoch_idx
            if epoch_idx % 10 == 0:
                logging.info(f'epoch {epoch_idx} / {epochs}')

            running_loss, running_loss_aux = [], []
            ys_pred, ys_true, ys_score = [], [], []
            ys_pred_regression, ys_true_regression = [], []
            # --- Training loop over batches
            for batch_idx, data in enumerate(self.train_loader):
                # Small problem: if we're unlucky, the last batch may contain a single sample, batchnorm will fail and
                #                raise an error. In this case, just ignore this last sample
                # print(f'epoch {epoch_idx:<5} batch {batch_idx}')
                # TODO find another workaround
                if data.y.shape[0] == 1:
                    logging.error(f'skipping batch {batch_idx} of epoch {epoch_idx} containing a single sample '
                                  f'(because of Batchnorm)')
                    continue

                # Batch initialization
                data = data.to(self.device)
                self.optimizer.zero_grad()
                # Predict
                loss, aux_loss, y_pred, y_pred_regression, y, y_regression = self.get_losses(data)
                # Backward propagation
                # if there's no auxiliary task, aux_loss is disconnected to the computational graph,
                # i.e., it does not interferes with the main task
                (loss + aux_loss * self.aux_loss_weight).backward()
                # Optimization step
                self.optimizer.step()

                # Process classification output
                ys_score.append(y_pred.detach().cpu().numpy())
                pred = y_pred.argmax(dim=1)
                ys_pred.append(pred.detach().cpu().numpy())
                ys_true.append(y.detach().cpu().numpy())
                running_loss.append(loss.detach().cpu().item())
                # Process regression output
                running_loss_aux.append(aux_loss.detach().cpu().item())
                ys_pred_regression.append(y_pred_regression.detach().cpu().numpy())
                ys_true_regression.append(y_regression)

            # --- Compute metrics
            train_loss = float(np.mean(running_loss))
            train_loss_aux = float(np.mean(running_loss_aux))
            ys_score = np.concatenate(ys_score)
            ys_true = np.concatenate(ys_true)
            ys_pred = np.concatenate(ys_pred)
            metrics = self.calculate_binary_classif_metrics(ys_pred, ys_true, ys_score)
            metrics['loss'] = train_loss
            metrics['epoch'] = epoch_idx
            metrics['aux'] = {'MSE': train_loss_aux}

            # --- WandB logging, don't commit i) for performance ii) to align train and eval logs
            run.log({'train': metrics}, commit=False)

            # --- Evaluation
            self.model.eval()
            val_metrics = self.evaluate(val_set=True)

            # --- Early stopping
            # Current epoch is the best -> reset
            # TODO: change this to average of last 3 val_loss is < min avg of 3 consecutive val losses
            if val_metrics['loss'] < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_metrics['loss']
                last_best_model = self.model.state_dict()
                last_best_optimizer = self.optimizer.state_dict()
                last_best_val_metrics = val_metrics.copy()
                last_best_train_metrics = metrics.copy()
            # Otherwise, increase counter
            else:
                epochs_no_improve += 1
            # Check if early stopping should apply
            if epochs_no_improve > early_stop and epoch_idx > allow_stop:
                logging.info(f'early stop at epoch {epoch_idx}')
                # # Save best model
                # last_best_epoch = self.epoch - early_stop
                # self.save_model(last_best_model, last_best_optimizer,
                #                 last_best_epoch, last_best_val_metrics, run)
                # # Update summary: should reflect the validation metrics at last best epoch
                # # Warning: this doesn't work as if you call run.summary *once a run has finished*
                # run.summary.update({
                #     '_step': last_best_epoch,
                #     'train': last_best_train_metrics,
                #     'val': last_best_val_metrics,
                #     'val_loss': last_best_val_metrics['loss'],
                #     'early_stop': True
                # })
                return val_metrics

        logging.info('training done')
        run.summary.update({'early_stop': False})
        self.save_model(self.model.state_dict(), self.optimizer.state_dict(),
                        self.epoch, val_metrics, run)
        return val_metrics

    def evaluate(self, val_set: bool, run: wandb.sdk.wandb_run.Run = wandb) -> Dict[str, float]:
        """
        val_set: bool indicating whether we use validation or test set
        """
        if val_set:
            dataloader = self.val_loader
            prefix = 'val'
        else:
            dataloader = self.test_loader
            prefix = 'test'

        running_loss, running_loss_aux = [], []
        ys_pred, ys_true, ys_score = [], [], []
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                loss, aux_loss, y_pred, y_pred_regression, y, y_regression = self.get_losses(data)
                ys_score.append(y_pred.cpu().detach().numpy())
                pred = y_pred.argmax(dim=1)
                ys_pred.append(pred.cpu().detach().numpy())
                ys_true.append(y.cpu().detach().numpy())
                running_loss.append(loss.detach().cpu().item())
                running_loss_aux.append(aux_loss.detach().cpu().item())

        val_loss = float(np.mean(running_loss))
        val_loss_aux = float(np.mean(running_loss_aux))
        ys_score = np.concatenate(ys_score)
        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)
        metrics = self.calculate_binary_classif_metrics(ys_pred, ys_true, ys_score)
        metrics['loss'] = val_loss
        metrics['epoch'] = self.epoch
        metrics['aux'] = {'MSE': val_loss_aux}

        run.log({
            # must use top-level metric for sweep logging, see wandb sweep documentation
            prefix + '_loss': val_loss,
            prefix + '_f1_score': self.compute_f1_mean(metrics),
            prefix: metrics
        })
        return metrics

    def compute_f1_mean(self, metrics):
        # Harmonic mean of f1 scores for + and - classes
        f1_1 = metrics['1']['f1-score']
        f1_0 = metrics['0']['f1-score']
        if f1_1 == 0.0 or f1_0 == 0.0:
            return 0.0

        return 2 * (f1_1 * f1_0) / (f1_1 + f1_0)


if __name__ == '__main__':
    from datasets import split_data

    #config_path = 'toolbox/config-4786gy4q.json'
    config_path = 'config/config-debug.json'
    with open(setup.get_project_root_path().joinpath(config_path)) as f:
        config = json.load(f)

    test_set, ((train_set, val_set),) = split_data(path=setup.get_dataset_path(config['dataset.name']),
                                                   num_node_features=config['dataset.num_node_features'],
                                                   seed=config['cv.seed'],
                                                   cv=False,
                                                    valid_ratio=0.2,
                                                   in_memory=config['dataset.in_memory'],
                                                   exclude_files=['CHUV03_LAD', 'OLV036_LAD'])

    gnn = GNN(
        config=config,
        train_set=train_set,
        valid_set=val_set,
        test_set=test_set,
    )

    # Check data std
    def get_stats(dset: PatientDataset):
        data = torch.concat([e.x for e in dset._data])
        return data.mean(dim=0), data.std(dim=0)

    run = wandb.init(**setup.WANDB_SETTINGS, group='trash', job_type='trash')
    metrics = gnn.train(1, 1000, 1000, run=run)
    mtest = gnn.evaluate(False, run)
    gnn.save_predictions(run)
    a=0 # for breakpoint

    #run = wandb.init(**setup.WANDB_SETTINGS, job_type='trash', group='trash')
    #gnn.train(config['epochs'], early_stop=config['early_stop'], allow_stop=config['allow_stop'], run=run)
    #gnn.save_predictions(run)



