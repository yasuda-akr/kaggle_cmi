# scripts/evaluate.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# パスを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import quadratic_weighted_kappa

def main():
    # 予測結果と正解ラベルをロード
    submission = pd.read_csv('submission.csv')
    true_labels = pd.read_csv('data/raw/train.csv')

    # 'id'でマージ
    merged = pd.merge(submission, true_labels[['id', 'sii']], on='id', how='inner')

    # 評価指標の計算
    qwk = quadratic_weighted_kappa(merged['sii_y'], merged['sii_x'])
    print(f"Quadratic Weighted Kappa Score: {qwk:.4f}")

if __name__ == "__main__":
    main()
