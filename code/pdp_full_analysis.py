import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data_path = "../data/processed_ufc_data_for_prediction.csv"
df = pd.read_csv(data_path)
df = df.drop(columns=[col for col in df.columns if "Fighter" in col], errors='ignore')
df['RedWin'] = (df['Winner'] == 1).astype(int)
X = df.drop(columns=["RedWin", "Winner"])
y = df["RedWin"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Load BART outputs
model_path = "../models/bart_model_for_prediction.nc"
idata = az.from_netcdf(model_path)
μ_mean = idata.posterior["μ"].mean(dim=["chain", "draw"]).values

# Check match
if μ_mean.shape[0] != X_train.shape[0]:
    raise ValueError(f"Mismatch: μ_mean has {μ_mean.shape[0]} samples but X_train has {X_train.shape[0]} rows")

# Train surrogate
y_surrogate = 1 / (1 + np.exp(-μ_mean))
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
surrogate.fit(X_train_scaled, y_surrogate)

# Features
single_features = [
    "SigStrDif", "AvgTDDif", "AvgSubAttDif",
    "WinStreakDif", "LongestWinStreakDif", "TotalRoundDif",
    "HeightDif", "ReachDif", "AgeDif",
    "KODif", "SubDif",
    "RedAvgSigStrPct", "RedAvgTDPct",
    "BlueAvgSigStrPct", "BlueAvgTDPct",
    "BetterRank"
]

pair_features = [
    ("SigStrDif", "AvgTDDif"),
    ("WinStreakDif", "AgeDif"),
    ("ReachDif", "TotalRoundDif"),
    ("HeightDif", "WinStreakDif"),
    ("RedAvgSigStrPct", "RedAvgTDPct"),
    ("BlueAvgSigStrPct", "BlueAvgTDPct"),
    ("KODif", "SubDif"),
    ("SigStrDif", "ReachDif"),
    ("AvgSubAttDif", "AvgTDDif"),
    ("BetterRank", "SigStrDif"),
]

# Output folders
os.makedirs("../figures/pdp", exist_ok=True)
os.makedirs("../figures/pdp_pairs", exist_ok=True)

# Single feature PDP/ICE plots
for feature in single_features:
    print(f"Plotting PDP/ICE for: {feature}")
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_idx = X_train.columns.get_loc(feature)

    PartialDependenceDisplay.from_estimator(
        surrogate,
        X_train_scaled,
        features=[feature_idx],
        feature_names=X_train.columns,
        kind="both",
        subsample=100,
        n_jobs=-1,
        grid_resolution=50,
        ax=ax,
        random_state=42
    )
    plt.title(f"PDP and ICE: {feature}")
    plt.tight_layout()
    plt.savefig(f"../figures/pdp/{feature}_pdp_ice.png")
    plt.close()

# Pairwise feature PDP
for feature_pair in pair_features:
    print(f"Plotting PDP for pair: {feature_pair}")
    fig, ax = plt.subplots(figsize=(8, 5))
    feature_idx = [X_train.columns.get_loc(f) for f in feature_pair]

    PartialDependenceDisplay.from_estimator(
        surrogate,
        X_train_scaled,
        features=[tuple(feature_idx)],
        feature_names=X_train.columns,
        kind="average",
        n_jobs=-1,
        grid_resolution=30,
        ax=ax,
        random_state=42
    )
    plt.title(f"PDP: {feature_pair[0]} vs {feature_pair[1]}")
    plt.tight_layout()
    plt.savefig(f"../figures/pdp_pairs/{feature_pair[0]}_{feature_pair[1]}_pdp_pair.png")
    plt.close()

plt.close('all')
print("\nAll PDP plots generated")
