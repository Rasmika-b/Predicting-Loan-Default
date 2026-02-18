# harness.py
import argparse
import pandas as pd
import numpy as np
import joblib

from estimator import estimator
from predictor import predictor
from preprocess import preprocess_data

def parse_args():
    parser = argparse.ArgumentParser(description="Process input and output CSV files.")
    parser.add_argument("--input_csv", required=True, type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", required=True, type=str, help="Path to the output CSV file.")
    return parser.parse_args()

def harness(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    df_processed = preprocess_data(df)
    model = joblib.load("logit.joblib")

    predictions = predictor(df_processed, model)

    pd.DataFrame(predictions).to_csv(output_csv, index=False, header=False)

    print(f"Saved {len(predictions)} predictions to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    harness(args.input_csv, args.output_csv)