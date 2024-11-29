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

def lgb_wandb_callback():
    def _callback(env: CallbackEnv):
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            wandb.log({f"{data_name}_{eval_name}": result, 'iteration': env.iteration})
    _callback.order = 10  # コールバックの順序を指定
    return _callback

class TabNetWandbCallback(Callback):
    def on_epoch_end(self, epoch_idx):
        logs = self.trainer.history['train']
        for key, value in logs.items():
            wandb.log({f"TabNet_train_{key}": value[-1], 'epoch': epoch_idx})
        if self.trainer.history['val_0']:
            val_logs = self.trainer.history['val_0']
            for key, value in val_logs.items():
                wandb.log({f"TabNet_valid_{key}": value[-1], 'epoch': epoch_idx})
