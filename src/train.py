import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from torch_geometric.loader import DataLoader

from .models import NoPhysicsGnn, EquivNoPhys


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
    def __init__(
            self,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            optim_param,
            weight1,
            args
    ):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(dev)
        print("Using device:", self.device)
        self.model_type = model_param['type']
        self.ratio = 1.0  # only useful for phys models

        if self.model_type == 'NoPhysicsGnn':
            self.physics = False
            self.automatic_update = False
            self.model = NoPhysicsGnn(train_set)
        elif self.model_type == 'Equiv':
            self.physics = False
            self.automatic_update = False
            self.model = EquivNoPhys(train_set,
                                     num_gin=args['num_gin'],
                                     num_equiv=args['num_equiv'])
    #                                 , args['n_layers'])
        else:
            raise ValueError('unrecognized model type')

        self.model.to(self.device)

        # initialize data loader for batching
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True)
        self.val_loader = DataLoader(valid_set, batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False) if test_set is not None else None

        if optim_param['optimizer'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=optim_param['lr'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=optim_param['lr'],
                                             momentum=optim_param['momentum'])
        weights = torch.tensor([1 - weight1, weight1])  # for class imbalance

        self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device) # TODO: check consistency with last layer of classifier. (Seems to apply softmax twice.)

    @staticmethod
    def calculate_metrics(y_pred, y_true):
        """
        Calculates several metrics to evaluate model.

        Parameters
        -----------
        y_pred : np.array with predictions of model
        y_true : np.array with labels
        """
        accuracy = accuracy_score(y_true, y_pred)
        # tp/(tp + fp) i.e. fracttion of right positively labelled guesses
        precision = precision_score(y_true, y_pred, zero_division=0)
        # tp / (tp + fn) i.e. fraction of rightly guessed CULPRITS
        recall = recall_score(y_true, y_pred)
        # harmonic mean btw precision and recall
        f1score = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        # tn / (tn + fp) like recall but for Non Culprits
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        # tp /(fn + tp): this is equal to recall
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        return accuracy, precision, recall, sensitivity, specificity, f1score

    def get_losses(self, data):
        if self.physics:
            raise ValueError("!! Wrong argument: self.physics set to true. Not supported in this project.")

        cnc = data.y
        cnc_pred = self.model(data.x,
                              data.edge_index,
                              data.batch,
                              data.segment)
        loss = loss_cnc = self.criterion(cnc_pred, data.y)
        return loss, loss_cnc, cnc_pred, cnc

    def train(self, epochs, early_stop, allow_stop=200, run=wandb):
        self.model.train()
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(1, epochs + 1):
            if epoch_idx % 10 == 0:
                logging.info(f'epoch {epoch_idx} / {epochs}')
            running_loss_cnc = 0.0  # Remains from physics models.
            y_pred = np.array([])
            y_true = np.array([])
            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                loss.backward()
                self.optimizer.step()
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()

            train_loss_cnc = running_loss_cnc / len(self.train_loader.dataset)
            (acc, prec, rec,
             sens, spec, f1score) = self.calculate_metrics(y_pred, y_true)
            run.log({
                'ratio': self.ratio,
                'train_accuracy': acc,
                'train_precision': prec,
                'train_recall': rec,
                'train_sensitivity': sens,
                'train_specificity': spec,
                'train_f1score': f1score,
                'train_loss_graph': train_loss_cnc
            })

            self.model.eval()
            # val score:
            (acc, prec, rec, sensitivity,
            specificity, f1_score, val_loss) = self.evaluate(val_set=True)
            if val_loss < min_val_loss: # TODO: change this to average of last 3 val_loss is < min avg of 3 consecutive val losses
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve > early_stop and epoch_idx > allow_stop:
                logging.info(f'early stop at epoch {epoch_idx}')
                return (acc, prec, rec, sensitivity,
                        specificity, f1_score, val_loss)

        logging.info('training done')
        return acc, prec, rec, sensitivity, specificity, f1_score, val_loss

    def evaluate(self, val_set, run=wandb):
        """
        val_set: bool indicating whether we use validation or test set
        """
        if val_set:
            dataloader = self.val_loader
            prefix = 'val'
        else:
            dataloader = self.test_loader
            prefix = 'test'
        self.model.eval()
        running_loss_cnc, running_loss = 0.0, 0.0
        y_pred = np.array([])
        y_true = np.array([])
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                loss, loss_cnc, cnc_pred, cnc = self.get_losses(data)
                pred = cnc_pred.argmax(dim=1)
                y_pred = np.append(y_pred, pred.cpu().detach().numpy())
                y_true = np.append(y_true, cnc.cpu().detach().numpy())
                running_loss_cnc += loss_cnc.item()
                running_loss += loss.item()

        val_loss_cnc = running_loss_cnc / len(self.val_loader.dataset)
        val_loss = running_loss / len(self.val_loader.dataset)
        (acc, prec, rec, sensitivity,
         specificity, f1_score) = self.calculate_metrics(y_pred, y_true)
        run.log({
            prefix + '_accuracy': acc,
            prefix + '_precision': prec,
            prefix + '_recall': rec,
            prefix + '_sensitivity': sensitivity,
            prefix + '_specificity': specificity,
            prefix + '_f1score': f1_score,
            prefix + '_loss_graph': val_loss_cnc
        })
        return acc, prec, rec, sensitivity, specificity, f1_score, val_loss
