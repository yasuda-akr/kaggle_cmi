# src/utils.py

import numpy as np
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_rounder(predictions, thresholds):
    return np.where(predictions < thresholds[0], 0,
                    np.where(predictions < thresholds[1], 1,
                             np.where(predictions < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, predictions):
    rounded_preds = threshold_rounder(predictions, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_preds)
