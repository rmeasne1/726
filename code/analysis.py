import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
model_path = "../models/bart_model_for_prediction.nc"
data_path = "../data/processed_ufc_data_for_prediction.csv"

# Load model trace with predictions
idata = az.from_netcdf(model_path)

# Ensure predictions were saved
if "μ" not in idata.predictions:
    raise ValueError("μ (test-time predictions) not found in InferenceData.predictions. Check if they were saved properly.")

# Extract test-time μ samples
μ_test_samples = idata.predictions["μ"].values  # shape: (chains, draws, test_samples)

# Use std dev across samples as a proxy signal
μ_std = μ_test_samples.std(axis=(0, 1))  # shape: (test_samples,)

# Sort and visualize the test instances with most uncertain μ
top_k = 20
top_indices = np.argsort(μ_std)[-top_k:][::-1]

# Plot
plt.figure(figsize=(12, 6))
plt.title("Most Uncertain Test Predictions (StdDev of μ)")
plt.bar(range(top_k), μ_std[top_indices], align="center")
plt.xticks(range(top_k), top_indices, rotation=90)
plt.xlabel("Test Sample Index")
plt.ylabel("StdDev of μ")
plt.tight_layout()
plt.show()
