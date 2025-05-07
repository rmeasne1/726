import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data_path = "../data/processed_ufc_data_for_prediction.csv"
df = pd.read_csv(data_path)
df = df.drop(columns=[col for col in df.columns if "Fighter" in col], errors='ignore')
df['RedWin'] = (df['Winner'] == 1).astype(int)
X = df.drop(columns=["RedWin", "Winner"])
y = df["RedWin"]

# Apply the same train-test split used in bart.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale features for PDP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Load BART posterior and extract mean μ
model_path = "../models/bart_model_for_prediction.nc"
idata = az.from_netcdf(model_path)
μ_mean = idata.posterior["μ"].mean(dim=["chain", "draw"]).values

# Verify shape matches training set
if μ_mean.shape[0] != X_train.shape[0]:
    raise ValueError(f"Mismatch: μ_mean has {μ_mean.shape[0]} samples but X_train has {X_train.shape[0]} rows")

# Use sigmoid(μ) as continuous surrogate target
y_surrogate = 1 / (1 + np.exp(-μ_mean))

# Train surrogate regressor for PDP
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
surrogate.fit(X_train_scaled, y_surrogate)

# Select features for PDP/ICE
features_to_plot = [
    "SigStrDif",
    "AvgTDDif",
    "WinStreakDif",
    "AgeDif",
    ("SigStrDif", "AvgTDDif")
]

# Convert names to indices for scaled input
feature_indices = [
    X_train.columns.get_loc(f) if isinstance(f, str)
    else tuple(X_train.columns.get_loc(feat) for feat in f)
    for f in features_to_plot
]

# Plot PDP and ICE
PartialDependenceDisplay.from_estimator(
    surrogate,
    X_train_scaled,
    features=feature_indices,
    feature_names=X_train.columns,
    kind=["both", "both", "both", "both", "average"],
    subsample=100,
    n_jobs=-1,
    grid_resolution=50,
    random_state=42
)

plt.tight_layout()
plt.show()
