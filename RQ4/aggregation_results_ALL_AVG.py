import pandas as pd
import glob
import os

# Folder with CSV files
folder = "/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_stats/mean"   # folder with the *_mean_agg.csv files

# Subject to abbreviation mapping
SUBJECT_MAP = {
    'audio':"AU",
    'mnist':"MN",
    'cifar10':"CF",
    'udacity':"UD",
    'reuters':"RT",
    'lenet':"UE"
}

# Collect rows
rows = []
for filepath in glob.glob(os.path.join(folder, "*_mean_agg.csv")):
    subject_name = os.path.basename(filepath).split("_")[0]  # get X from "X_mean_agg.csv"
    subj_abbr = SUBJECT_MAP.get(subject_name, subject_name)  # fallback to name if not in dict

    df = pd.read_csv(filepath)

    # Take last row, drop first column
    last_row = df.iloc[[-1], 1:].reset_index(drop=True)

    # Insert Subject column at the beginning
    last_row.insert(0, "Subject", subj_abbr)

    rows.append(last_row)

# Stack into one DataFrame
all_df = pd.concat(rows, ignore_index=True)

# Add percentage columns based on the last two columns
all_df[f"{all_df.columns[-2]}_perc"] = all_df[all_df.columns[-2]] * 100
all_df[f"{all_df.columns[-2]}_perc"] = all_df[all_df.columns[-2]] * 100

# Save to CSV
output_path = os.path.join(folder, "ALL_AVG_SUBJ.csv")
all_df.to_csv(output_path, index=False)

print(f"Saved to {output_path}")
