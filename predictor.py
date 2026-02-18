# predictor.py
"""
Prediction utilities: preprocessing helpers and a thin prediction wrapper.

This module contains:
- `safe_div`: elementwise safe division for pandas Series/arrays.
- `preprocess_data`: create derived columns and fill relationships.
- `prepare_features`: clip outliers and replace infinities.
- `predictor`: returns prediction probabilities for the positive class.

These functions are intended to be used by the harness; they do not
perform any model estimation or use holdout data for training.
"""
from constants import FEATURE_COLS, Q_LOW, Q_HIGH
import pickle
import numpy as np

def apply_calibration(raw_p, scale, slope, shift):
    transformed = scale * np.exp(slope * raw_p) + shift
    return transformed.clip(0, 1)

def filter_feature_cols(df):
    return df[FEATURE_COLS].copy()

def transform_data(df):
    df_new = df.copy()

    for col in FEATURE_COLS:
        df_new[col] = df_new[col].clip(lower=Q_LOW[col], upper=Q_HIGH[col])

    with open('quantile_transformer.pkl', 'rb') as file:
        loaded_qt = pickle.load(file)

    return loaded_qt.transform(df_new)

def predictor(df, model):
    """Return model predicted probability for positive class.

    Expects `X` to be a DataFrame or array with the same features the model
    was trained on. This wrapper keeps the harness code simple.
    """
    X = filter_feature_cols(df)
    X = transform_data(X)
    preds = model.predict_proba(X)[:, 1]
    return apply_calibration(preds, -0.48262383, -1.62781953, 0.48462383)