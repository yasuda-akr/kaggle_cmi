# src/utils.py
import random
import os
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
import wandb
from lightgbm.callback import CallbackEnv
from pytorch_tabnet.callbacks import Callback

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_rounder(predictions, thresholds):
    return np.where(predictions < thresholds[0], 0,
                    np.where(predictions < thresholds[1], 1,
                             np.where(predictions < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, predictions):
    rounded_preds = threshold_rounder(predictions, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_preds)

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
