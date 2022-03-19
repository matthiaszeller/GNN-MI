

import logging
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, OrderedDict

import numpy as np
import torch
import wandb
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader

import setup
from datasets import PatientDataset
from models import NoPhysicsGnn, EGNN, GNNBase, checkpoint_model


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
    """

    # TODO: Add a model "EquivWSS" which is exactly like "Equiv", but
    # which uses the WSS scalars as invariant features h. 

    # TODO: Reimplement the models that includes Physics
    # (i.e. loss term at node level) from Yunshu's project
    # (https://github.com/yooyoo9/MI-detection), and combine that
    # to the Equivariant framework. Note that this is different than,
    # EquivWSS, as the latter does not do any prediction at the node level. 
    # Good luck! :)

    model: GNNBase

    def __init__(
            self,
            config: Dict[str, Any],
            train_set: PatientDataset,
            valid_set: PatientDataset,
            test_set: Union[PatientDataset, None] = None,
            model_save_path: Union[str, Path] = None
    ):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(dev)
        logging.info(f'Using device: {self.device}')
        if dev != 'cuda':
            logging.warning('running on CPU, do you really want to do this !?')

        self.model_type = config['model.type']
        self.model_save_path = model_save_path
        if self.model_save_path is None:
            _, p = setup.get_data_paths()
            self.model_save_path = p.joinpath('models')

        self.ratio = 1.0  # only useful for phys models

        if self.model_type == 'NoPhysicsGnn':
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
                              num_gin=config['num_gin'])
        else:
            raise ValueError('unrecognized model type')

        self.model.to(self.device)

        # Debugging
        logging.debug(f'Samples from train, val, test sets:\n{train_set[0]}\n{valid_set[0]}\n'
                      f'{test_set[0] if test_set is not None else ""}')
        # initialize data loader for batching
        self.train_loader = DataLoader(train_set, config['batch_size'], shuffle=True)
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

        # TODO: check consistency with last layer of classifier. (Seems to apply softmax twice.)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        self.epoch = None

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
        #run.save(file_path)

    @staticmethod
    def calculate_binary_classif_metrics(y_pred, y_true) -> Dict[str, float]:
        """
        Calculates several metrics to evaluate model.

        Parameters
        -----------
        y_pred : np.array with predictions of model
        y_true : np.array with labels
        """
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics = report['1']
        metrics['accuracy'] = report['accuracy']
        return metrics

    def get_losses(self, data: DataLoader) -> Tuple[torch.Tensor, List, torch.Tensor]:
        if self.physics:
            raise ValueError("!! Wrong argument: self.physics set to true. Not supported in this project.")

        yhat = self.model(data.x, data.coord, data.g_x, data.edge_index, data.batch)
        loss = self.criterion(yhat, data.y)
        aux_loss = []
        return loss, aux_loss, yhat

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

            running_loss = []
            ys_pred, ys_true = [], []
            # --- Training loop over batches
            for data in self.train_loader:
                # Batch initialization
                data = data.to(self.device)
                self.optimizer.zero_grad()
                # Predict
                loss, aux_loss, y_pred = self.get_losses(data)
                # Backward propagation
                loss.backward()
                # Optimization step
                self.optimizer.step()

                pred = y_pred.argmax(dim=1)
                ys_pred.append(pred.detach().cpu().numpy())
                ys_true.append(data.y.detach().cpu().numpy())
                running_loss.append(loss.detach().cpu().item())

            # --- Compute metrics
            train_loss = float(np.mean(running_loss))
            ys_true = np.concatenate(ys_true)
            ys_pred = np.concatenate(ys_pred)
            metrics = self.calculate_binary_classif_metrics(ys_pred, ys_true)
            metrics['loss'] = train_loss
            metrics['epoch'] = epoch_idx

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

        running_loss = []
        ys_pred, ys_true = [], []
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                loss, aux_loss, y_pred = self.get_losses(data)
                pred = y_pred.argmax(dim=1)
                ys_pred.append(pred.cpu().detach().numpy())
                ys_true.append(data.y.cpu().detach().numpy())
                running_loss.append(loss.detach().cpu().item())

        val_loss = float(np.mean(running_loss))
        ys_true = np.concatenate(ys_true)
        ys_pred = np.concatenate(ys_pred)
        metrics = self.calculate_binary_classif_metrics(ys_pred, ys_true)
        metrics['loss'] = val_loss
        metrics['epoch'] = self.epoch

        run.log({
            # must use top-level metric for sweep logging, see wandb sweep documentation
            'val_loss': val_loss,
            'val_f1_score': metrics['val.f1-score'],
            prefix: metrics
        })
        return metrics
