import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
input_path = "../data/ufc-master.csv"
output_path = "../data/processed_ufc_data_for_prediction.csv"
df = pd.read_csv(input_path)

# Drop fighter names
df.drop(columns=['RedFighter', 'BlueFighter'], inplace=True, errors='ignore')

# If predicting future fights, drop fight outcome columns
predicting = True
if predicting:
    drop_cols = ['Finish', 'FinishDetails', 'FinishRound', 'FinishRoundTime', 'TotalFightTimeSecs']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    output_path = "../data/processed_ufc_data_for_prediction.csv"

# Drop betting odds and expected value columns
drop_odds_cols = [
    'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
    'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds'
]
df.drop(columns=[col for col in drop_odds_cols if col in df.columns], inplace=True)

# Process Date
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df.drop(columns=['Date'], inplace=True)

# Encode categorical variables
label_enc_cols = [
    'Location', 'Country', 'Winner', 'WeightClass', 'Gender',
    'RedStance', 'BlueStance', 'BetterRank'
]
for col in label_enc_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

# Encode booleans
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Flip difference columns so that higher = better for Red
diff_cols = [
    "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif", "WinDif", "LossDif",
    "TotalRoundDif", "TotalTitleBoutDif", "KODif", "SubDif", "HeightDif",
    "ReachDif", "AgeDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif"
]

for col in diff_cols:
    if col in df.columns:
        df[col] = -df[col]  # Flip the sign for Red frame of reference

# Save processed
df.to_csv(output_path, index=False)
print(f"Processed dataset saved to: {output_path}")
