import os
import glob
import pandas as pd

# === CONFIG ===
INPUT_DIR = "/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_stats/mean"   # folder with the *_mean_agg.csv files
OUTPUT_CSV = "/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_stats/mean/ALL.csv"

# Map subject name (X in "X_mean_agg.csv") -> abbreviation you want in the 'Subject' column
SUBJECT_MAP = {
    'audio':"AU",
    'mnist':"MN",
    'cifar10':"CF",
    'udacity':"UD",
    'reuters':"RT",
    'lenet':"UE"
}

# === GATHER FILES ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_mean_agg.csv")))
if not files:
    raise SystemExit("No *_mean_agg.csv files found.")

# Read the first file just to get the header (column names)
sample_df = pd.read_csv(files[0])
original_columns = list(sample_df.columns)

# Build the combined dataframe
chunks = []
for fpath in files:
    # Extract subject name X from "X_mean_agg.csv"
    fname = os.path.basename(fpath)
    if not fname.endswith("_mean_agg.csv"):
        continue
    subject_name = fname[:-len("_mean_agg.csv")]

    if subject_name not in SUBJECT_MAP:
        raise SystemExit(f"Subject '{subject_name}' not found in SUBJECT_MAP. "
                         f"Add it before running.")

    # Read data, drop the last row, keep headers for columns
    df = pd.read_csv(fpath)

    if len(df) == 0:
        continue  # nothing to add

    # Drop the last row (regardless of what it contains)
    df = df.iloc[:-1, :] if len(df) > 0 else df

    if df.empty:
        continue

    # Ensure the columns match the first file’s header (optional but safer)
    df = df.reindex(columns=original_columns)

    # Add the Subject column (as the FIRST column)
    df.insert(0, "Subject", SUBJECT_MAP[subject_name])

    chunks.append(df)

if not chunks:
    raise SystemExit("No valid rows found after dropping last rows.")

all_df = pd.concat(chunks, ignore_index=True)

# Reorder columns: Subject first, then the original file columns
all_df = all_df[["Subject"] + original_columns]

# Save
all_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(all_df)} rows to {OUTPUT_CSV}")
