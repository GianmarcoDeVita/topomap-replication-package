import os
import pandas as pd

# Load the file
folder = "/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_stats/mean/"
df = pd.read_csv(os.path.join(folder, "ALL.csv"))

# Get the last two column names
last_two_cols = df.columns[-2:]

# Group by 'MO' and aggregate
grouped = (
    df.groupby("MO")[last_two_cols]
    .mean()                          # compute averages
    .round(2)                        # round to 2 decimals
)

# Add count of aggregated rows
grouped["RowCount"] = df.groupby("MO").size()

# Add percentage columns (last two columns * 100)
for col in last_two_cols:
    grouped[f"{col}_perc"] = (grouped[col] * 100).round(2)

# Reset index so 'MO' becomes a column
result = grouped.reset_index()

# Save to new file
result.to_csv(os.path.join(folder, "ALL_MO.csv"), index=False)
