import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pymc as pm
from pymc_bart import BART
import arviz as az

def main():
    # Paths
    data_path = "../data/processed_ufc_data_for_prediction.csv"
    model_path = "../models/bart_model_for_prediction.nc"

    # Load and prepare data
    df = pd.read_csv(data_path)
    df = df.drop(columns=[col for col in df.columns if "Fighter" in col], errors='ignore')
    df['RedWin'] = (df['Winner'] == 1).astype(int)
    X = df.drop(columns=["RedWin", "Winner"])
    y = df["RedWin"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_test_np = X_test.to_numpy()

    m = 50  # number of trees

    # Train BART model
    with pm.Model() as model:
        X_shared = pm.MutableData("X", X_train_np)
        μ = BART("μ", X_shared, Y=y_train_np, m=m)
        p = pm.Deterministic("p", pm.math.sigmoid(μ))
        y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train_np)
        trace = pm.sample(2000, tune=1000, cores=2, random_seed=42)

        # Switch shared data to test inputs
        pm.set_data({"X": X_test_np})

        # Sample posterior predictive on test data
        idata = pm.sample_posterior_predictive(
            trace,
            var_names=["μ"],
            return_inferencedata=True,
            predictions=True,
            extend_inferencedata=True
        )

        # Save model trace
        az.to_netcdf(idata, model_path)

    # Evaluate test predictions 
    μ_test_samples = idata.predictions["μ"]
    μ_test_mean = μ_test_samples.mean(dim=["chain", "draw"]).values
    p_test_mean = 1 / (1 + np.exp(-μ_test_mean))  # Apply sigmoid
    y_pred = (p_test_mean > 0.5).astype(int)

    # Print evaluation stats
    print("\n===== Evaluation on Test Set =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("==================================\n")

if __name__ == '__main__':
    main()
