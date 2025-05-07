import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup
data_path = "../data/processed_ufc_data_for_prediction.csv"
save_dir = "../figures/box_plots/"
os.makedirs(save_dir, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
df['RedWin'] = (df['Winner'] == 1).astype(int)

# Features to analyze
features = [
    "SigStrDif", "AvgTDDif", "WinStreakDif", "AgeDif", "HeightDif", "ReachDif",
    "AvgSubAttDif", "TotalRoundDif", "KODif", "SubDif",
    "RedAvgSigStrPct", "RedAvgTDPct", "BlueAvgSigStrPct", "BlueAvgTDPct", "BetterRank"
]

# Clip range
clip_min, clip_max = -20, 20

# Process each feature
for feature in features:
    if feature not in df.columns:
        print(f"Skipping missing feature: {feature}")
        continue

    print(f"Plotting: {feature}")

    df[feature] = df[feature].clip(lower=clip_min, upper=clip_max)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='RedWin', y=feature, data=df, whis=[5, 95], showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})

    sns.stripplot(x='RedWin', y=feature, data=df, size=3, color=".3", alpha=0.3, jitter=True)

    plt.title(f"{feature}")
    plt.xticks([0, 1], ['Blue Wins (0)', 'Red Wins (1)'])
    plt.xlabel("Fight Outcome")
    plt.ylabel(feature)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save figure
    plot_filename = os.path.join(save_dir, f"{feature}_boxplot.png")
    plt.savefig(plot_filename)
    plt.close()

    # Save summary stats
    summary = df.groupby('RedWin')[feature].agg(['mean', 'std', 'median', 'count'])

    summary_filename = os.path.join(save_dir, f"{feature}_summary.txt")
    with open(summary_filename, "w") as f:
        f.write(f"Summary Statistics for {feature}:\n\n")
        f.write(summary.to_string())
        f.write("\n")

print("\nAll box plots and summaries saved!")
