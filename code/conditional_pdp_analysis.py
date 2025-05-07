import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# Config 
data_path = "../data/processed_ufc_data_for_prediction.csv"
model_path = "../models/bart_model_for_prediction.nc"
figures_dir = "../figures/conditional_pdp"
os.makedirs(figures_dir, exist_ok=True)

split_std_multiplier = 1.5  # Stronger split: +/- 1.5 std devs

# Load data
df = pd.read_csv(data_path)
df = df.drop(columns=[col for col in df.columns if "Fighter" in col], errors='ignore')
df['RedWin'] = (df['Winner'] == 1).astype(int)
X = df.drop(columns=["RedWin", "Winner"])
y = df["RedWin"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Load BART predictions
idata = az.from_netcdf(model_path)
μ_mean = idata.posterior["μ"].mean(dim=["chain", "draw"]).values

if μ_mean.shape[0] != X_train.shape[0]:
    raise ValueError("Mismatch between μ and training samples!")

# Train surrogate
y_surrogate = 1 / (1 + np.exp(-μ_mean))
surrogate = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
surrogate.fit(X_train_scaled, y_surrogate)

# Features
target_features = [
    "SigStrDif", "AvgTDDif", "WinStreakDif", "AgeDif", "HeightDif", "ReachDif",
    "AvgSubAttDif", "TotalRoundDif", "KODif", "SubDif",
    "RedAvgSigStrPct", "RedAvgTDPct", "BlueAvgSigStrPct", "BlueAvgTDPct", "BetterRank"
]

conditioning_features = [
    "ReachDif", "HeightDif", "AgeDif", "WinStreakDif", "TotalRoundDif",
    "KODif", "SubDif", "RedAvgSigStrPct", "BlueAvgSigStrPct", "RedAvgTDPct", "BlueAvgTDPct"
]

# Conditional PDP Analysis
interaction_strengths = {}

for target_feature in target_features:
    for cond_feature in conditioning_features:
        if target_feature == cond_feature:
            continue

        print(f"Analyzing conditional PDP: {target_feature} | conditioned on {cond_feature}")

        feature_idx = X_train.columns.get_loc(target_feature)
        cond_idx = X_train.columns.get_loc(cond_feature)

        cond_vals = X_train_scaled[:, cond_idx]
        mean = np.mean(cond_vals)
        std = np.std(cond_vals)

        low_mask = cond_vals <= (mean - split_std_multiplier * std)
        high_mask = cond_vals >= (mean + split_std_multiplier * std)

        n_low = np.sum(low_mask)
        n_high = np.sum(high_mask)

        if n_low < 20 or n_high < 20:
            print(f"  Skipping {target_feature} | {cond_feature} (too few samples: low={n_low}, high={n_high})")
            continue

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, mask, label in zip(
            axs, [low_mask, high_mask], [f"{cond_feature} Low", f"{cond_feature} High"]
        ):
            PartialDependenceDisplay.from_estimator(
                surrogate,
                X_train_scaled[mask],
                features=[feature_idx],
                feature_names=X_train.columns,
                kind="both",
                subsample=100,
                grid_resolution=50,
                ax=ax,
                random_state=42
            )
            ax.set_title(label)

        plt.suptitle(f"Conditional PDP: {target_feature} | {cond_feature} Split")
        plt.tight_layout()
        save_path = os.path.join(figures_dir, f"{target_feature}_cond_{cond_feature}.png")
        plt.savefig(save_path)
        plt.close()

        # Difference of PDP means
        disp_low = surrogate.predict(X_train_scaled[low_mask])
        disp_high = surrogate.predict(X_train_scaled[high_mask])
        interaction_strengths[(target_feature, cond_feature)] = abs(disp_high.mean() - disp_low.mean())

# Rank strongest interactions
print("\n--- Strongest Interactions (based on avg PDP difference) ---")
ranked = sorted(interaction_strengths.items(), key=lambda x: x[1], reverse=True)
for (target, cond), strength in ranked[:25]:
    print(f"{target} conditioned on {cond}: strength = {strength:.5f}")

print("\nConditional PDP analysis complete!")
