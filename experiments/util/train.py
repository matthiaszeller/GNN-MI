import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from torch_geometric.loader import DataLoader

from .models import NoPhysicsGnn, EquivNoPhys


class GNN:
    def __init__(
            self,
            model_path,
            model_param,
            train_set,
            valid_set,
            test_set,
            batch_size,
            optim_param,
            weight1,
            args
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        self.model_name = model_param['name']
        self.model_path = model_path
        self.ratio = 1.0 # not sure what this does. Assuming its for phys models. TODO: might delete

        if self.model_name == 'NoPhysicsGnn':
            self.physics = False
            self.automatic_update = False
            self.model = NoPhysicsGnn(train_set)
        elif self.model_name == 'Equiv':
            self.physics = False
            self.automatic_update = False
            self.model = EquivNoPhys(train_set, args['n_layers'])
        else:
            print("Unrecognized model name")
        print("pushing model to: ", self.device)
        self.model.to(self.device)

        self.train_loader = DataLoader(train_set, batch_size, shuffle=True) # set pin_memory=True to go faster? https://pytorch.org/docs/stable/data.html
        self.val_loader = DataLoader(valid_set, batch_size, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size, shuffle=False)

        if optim_param['name'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_param['lr'])
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=optim_param['lr'], momentum=optim_param['momentum'])
        weights = torch.tensor([1 - weight1, weight1]) # is this due to class imbalance??

        self.criterion = torch.nn.CrossEntropyLoss(weight=weights).to(self.device)
        # self.criterion_node = torch.nn.MSELoss().to(self.device)

    @staticmethod
    def calculate_metrics(y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0) # = tp/(tp + fp) i.e. fracttion of right positively labelled guesses:: wrongly classified as culprits
        recall = recall_score(y_true, y_pred) # tp / (tp + fn) i.e. fraction of rightly guessed CULPRITS: rightly classified culprits
        f1score = f1_score(y_true, y_pred) # harmonic mean btw precision and recall

        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) # tn / (tn + fp) like recall but for Non Culprits
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) # tp /(fn + tp): this is equal to recall
        return accuracy, precision, recall, sensitivity, specificity, f1score

    def get_losses(self, data):
        if self.physics:
            print("!! Wrong argument: self.physics set to true. Not supported in this project.")
        else:
            cnc = data.y
            cnc_pred = self.model(data.x, data.edge_index, data.batch, data.segment)
            loss = loss_cnc = self.criterion(cnc_pred, data.y)
        return loss, loss_cnc, cnc_pred, cnc

    def train(self, epochs, early_stop, allow_stop=200, run=wandb):
        self.model.train()
        epochs_no_improve = 0
        min_val_loss = 1e8
        for epoch_idx in range(epochs):
            if epoch_idx %10==0:
                print('epoch nb:', epoch_idx)
            running_loss_cnc = 0.0 # what is this??
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
            acc, prec, rec, sens, spec, f1score = self.calculate_metrics(y_pred, y_true)
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
            acc, prec, rec, sensitivity, specificity, f1_score, val_loss = self.evaluate(val_set=True)
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if epochs_no_improve > early_stop and epoch_idx>allow_stop:
                print("Early stoped at epoch: ", epoch_idx)
                return acc, prec, rec, sensitivity, specificity, f1_score, val_loss
        print("Done training!")
        return acc, prec, rec, sensitivity, specificity, f1_score, val_loss

    def evaluate(self, val_set, run=wandb): # val set is boolean, saying whether we use calidation or test set
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
        acc, prec, rec, sensitivity, specificity, f1_score = self.calculate_metrics(y_pred, y_true)
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

        
