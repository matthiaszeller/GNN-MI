# MI-proj
Code and report for my semester project on using rotational and translational equivariant graph neural networks to predict cardiac arrest from 3 dimensional reconstructed arteries. For more information about the experiments, please check out my [write up](https://github.com/jacobbamberger/MI-proj/blob/main/MI_pred_Report.pdf)!

## Recommendation on navigating this repo
The way I recommend navigating this repo is to:
1) Look at the [write up](https://github.com/jacobbamberger/MI-proj/blob/main/MI_pred_Report.pdf) 
2) Look at the code structure below
3) Look at the [experiments results](https://wandb.ai/yayabambam/mi-prediction) on wandb platform. For more information on how to navigate that, see the appendix of the write-up.
4) If you want to run your own results. You will need an MI-proj/data folder, containing a patient_dict.pickle file as described below in datasets.py description, and a data folder, for example CoordToCnc with your mesh data.
5) Generate your experiments by running python main_cross_val with appropriate hyperparameters in hyper_params.yaml.

## Code structure
 All code is in the experiments folder:\ 
 
 name                 |      Description\
 create_data.py       |      Data fetching and preprocessing. This should be run from the MI-proj directory.\
                      |      The executed function is at the bottom of the file, note that our dataset is not\
                      |      public, so you won't have access to the path and label_path directories. <br/>
 data_augmentation.py |      Contains all the data augmentation schemes attempted. Used in create_data.py.\
 datasets.py          |      Contains our custom DataSet object which is how we store the meshes. Also contains\
                      |      custom split_data function which does the train, validation, and test splits at\
                      |      the patient level. Note that you will need a file "MI-proj/data/patient_dict.pickle"\
                      |      containing the dictionary with patients as keys and artery name list as value.<br/>
hyper_params.yaml     |      File containing all hyperparameters of a given model. Used in evaluate.py and\ 
                      |      main_cross_val.py. If you plan on using it for evaluate.py, there should be one\ 
                      |      value per hyperparameter.<br/>
main_cross_val.py     |      Runs a grid search with cross validation on all combinations of hyperparameters in\ 
                      |      hyper_params.yaml. All experiments are recorded on the wandb platform. Make sure to\ 
                      |      change and remmember MODEL_TYPE to be able to retrieve the experiment on the wandb\
                      |      platform! This does not use the test set. This should be called from inside the\ 
                      |      MI-proj/experiments directory.<br/>
evaluate. py          |      Same as cross_validation, but evaluates the model on test set once it has finished\
                      |      training. This should be run with only one value per hyperparameter in hyper_params.yaml.\
                      |      It is crucial to use the same seed here as used when doing the grid search. Also records\
                      |      all results on the wandb platform. This should be called from inside the MI-proj/experiments\
                      |      directory.<br/>
gnnexplainer.ipynb    |      Coming soon! Jupyter notebook for the GNNExplainer experiment and visualization.<br/>


More code is found in the experiments/util:\

GNNExplainer.py       |     Slightly modified code from the [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/pdf/1903.03894.pdf) paper.\
                      |     Code was obtained from [this](https://github.com/RexYing/gnn-model-explainer) repo.<br/>
egnn.py               |     Slightly modified code from the [E(n) Equivariant Graph Neural Networks](https://arxiv.org/pdf/2102.09844.pdf) paper.\
                      |     Code was obtained from [this](https://github.com/vgsatorras/egnn) repo.<br/>
models.py             |     Contains all different models used in experiments.<br/>
train.py              |     Contains a custom GNN object definition. Main script used for training and evaluating our models.<br/>


 
