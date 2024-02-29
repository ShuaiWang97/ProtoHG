from config import config
args = config.parse()
from data_hete import data
from collections import defaultdict
import torch, os, numpy as np

import scipy.sparse as sp
from sklearn.metrics import f1_score
import wandb
import statistics

# set_seed, store_result, normalize_matrxi, f1_score_ 
def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    # seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_sparse(mx):
    """
    Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    print("rowsum is: ",rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def store_result(res, filename):

    file = open(filename,"w")
    file.write(str(args))
    file.write('\n')
    file.write("RESULTS FOR SPLITS:\n")
    for r in res:
        file.write(str(r)+'\n')
    file.write("\nSTATISTICS:\n")
    file.write("f1 macro mean: " + str(statistics.mean(res[:,1])))
    file.write("f1 macro sdev: " + str(statistics.stdev(res[:,1])))
    file.write(str(r)+'\n')
    file.write("f1 micro mean: " + str(statistics.mean(res[:,0])))
    file.write("f1 micro sdev: " + str(statistics.stdev(res[:,0])))
    file.close()
    print("f1 macro mean: " + str(statistics.mean(res[:,1])))
    print("f1 micro mean: " + str(statistics.mean(res[:,0])))

def normalize_matrxi(mx):
    """
    Row-normalize normal matrix
    """
    row_sums = mx.sum(axis=1)
    new_matrix = mx / row_sums[:, np.newaxis]

    return new_matrix

def f1_score_(labels, pred):
    """
    input the predictions and ground truth
    """
    scores = defaultdict()
    f1_micro = f1_score(y_true = labels.detach().cpu(), y_pred = np.argmax(pred.detach().cpu(), axis=1), average='micro')
    f1_macro = f1_score(y_true = labels.detach().cpu(), y_pred = np.argmax(pred.detach().cpu(), axis=1), average='macro')

    return f1_micro, f1_macro

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.save_path)
        self.val_loss_min = val_loss


def build_hyperedge(E_list):
    """
    input hyperedge list
    return incidence matrix (node_num * hyperedge_num)!! load this into the transformer.
    get the transformer from the .layer code and try to make it for heterogenous
    """
    # k is index and V are nodes
    # for k, v in E.items():
    #     E[k] = list(v)
    #     if k not in E[k]:
    #         E[k].append(k)
    # E = dict(sorted(E.items()))

    # Get E features of nodes
    # a = list(E.values())
    map(max, E_list)
    list(map(max, E_list))
    max_val = max(map(max, E_list))
    edge_feature = np.zeros((len(E_list), max_val + 1))

    # Build E matrix
    for row, list_ in enumerate(E_list):
        for i in np.arange(len(list_)):
            edge_feature[row, E_list[row][i]] = 1  # (4057, 26128)

    edge_feature = np.transpose(edge_feature)
    edge_feature = torch.from_numpy(edge_feature)
    return edge_feature
