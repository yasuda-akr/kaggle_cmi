# Child Mind Institute - Problematic Internet Use Competition

## Overview
This repository contains the solution code for the [Child Mind Institute - Problematic Internet Use Competition](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) on Kaggle.

## Competition Objective
Develop predictive models to identify early signs of problematic internet use (PIU) in children and adolescents based on their physical activity data.

## Evaluation Metric
- Quadratic Weighted Kappa (QWK)
- Score ranges from 0 (random agreement) to 1 (complete agreement)

## Directory Structure
├── config/
│   └── parameters.yaml     # Model and training parameters
├── data/                   # Dataset (gitignored)
├── models/                 # Trained models (gitignored)
├── notebooks/             # EDA and experiment notebooks
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── src/
│   ├── data_preprocessing.py  # Data preprocessing
│   ├── feature_engineering.py # Feature engineering
│   ├── models/               # Model definitions
│   │   ├── base.py
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   ├── catboost_model.py
│   │   ├── tabnet_model.py
│   │   └── ensemble.py
│   └── utils.py              # Utility functions
├── requirements.txt
└── README.md
Copy
## Technologies Used
- Python 3.8+
- LightGBM
- XGBoost
- CatBoost
- PyTorch (TabNet)
- Weights & Biases (Experiment tracking)
- pandas
- numpy
- scikit-learn
