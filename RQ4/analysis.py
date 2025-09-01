import csv
import os
# import numpy as np
import pandas as pd

from glob import glob


search_folder = '/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_aggregated/'
result_folder = '/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/selected_clusters_stats/'

agg_method = 'mean'

res_files = glob(os.path.join(search_folder, '*.csv'))
# res_files = ['/Users/nhumbatova/Documents/Projects/TopoMap/RQ3_2/selected_clusters/selected_clusters_aggregated/lenet.csv']
# print(res_files)

for res_file in res_files:
    print(res_file)

    subj = res_file.replace(search_folder, '').replace('.csv', '')
    # print(subj)

    df = pd.read_csv(res_file)
    print(df)


    group_by_cols = ['MO', 'ITR', 'NUM_CLUSTER']
    exclude_cols = ['MO', 'CONFIG', 'ITR', 'NUM_CLUSTER']

    agg_cols = df.columns.difference(exclude_cols)

    grouped_df = df.groupby(group_by_cols)[agg_cols].mean()
    grouped_df['RowCount'] = df.groupby(group_by_cols).size()

    final_columns = group_by_cols + [col for col in df.columns if col not in exclude_cols] + ['RowCount']
    result_df = grouped_df.reset_index()[final_columns]


    result_df['KillFlagHalf'] = result_df[agg_cols].apply(lambda row: 1 if (row >= 0.5).any() else 0, axis=1)
    result_df['KillFlagFull'] = result_df[agg_cols].apply(lambda row: 1 if (row == 1).any() else 0, axis=1)

    # for each row, is there a cell that is >= 0.5: 1 or 0
    # find average across 10 runs
    # for each run, for each clusters get rows >=0

    csv_file_name = os.path.join(result_folder, agg_method, subj + '_' + agg_method + '.csv')
    result_df.to_csv(csv_file_name, index=False)

    group_by_cols_2 = ['MO', 'RowCount']
    agg_cols_2 = ['KillFlagHalf', 'KillFlagFull']

    grouped_df_2 = result_df.groupby(group_by_cols_2)[agg_cols_2].mean()
    grouped_df_2['CountCheck'] = result_df.groupby(group_by_cols_2).size()
    result_df_2 = grouped_df_2.reset_index()[group_by_cols_2+agg_cols_2]

    result_df_2.insert(1, "MOCount", 1)
    # result_df_2["MOCount"] = 1

##########Option1
    # # Compute the mean for all numeric columns except the first
    # avg_values = result_df_2.iloc[:, 1:].mean().round(2)
    #
    # # Create a new row with 'AVG' in the first column and the averages in the rest
    # avg_row = pd.DataFrame([["AVG"] + avg_values.tolist()], columns=result_df_2.columns)
    #
    # # Append to the original df
    # result_df_2 = pd.concat([result_df_2, avg_row], ignore_index=True)

    ##########Option2
    # # Compute averages (ignoring first and last columns)
    # avg_values = result_df_2.iloc[:, 1:-1].mean().round(2)
    #
    # # Compute sum for the last column
    # sum_last = result_df_2.iloc[:, -1].sum()
    #
    # # Build the new row
    # new_row = [None] * len(result_df_2.columns)  # placeholder
    # new_row[0] = "AVG"  # first col
    # new_row[1:-1] = avg_values.tolist()  # middle cols
    # new_row[-1] = sum_last  # last col
    #
    # # Append row
    # result_df_2.loc[len(result_df_2)] = new_row

    ##########Option3
    # Compute averages for all columns except the first two
    avg_row = result_df_2.iloc[:, 2:].mean().round(2)

    # Build new row with "AVG" in first col, sum in second col, and averages for the rest
    new_row = pd.Series(
        ["AVG", result_df_2.iloc[:, 1].sum()] + avg_row.tolist(),
        index=result_df_2.columns
    )

    # Append to result_df_2
    result_df_2 = pd.concat([result_df_2, new_row.to_frame().T], ignore_index=True)


    csv_file_name_2 = os.path.join(result_folder, agg_method, subj + '_' + agg_method + '_agg.csv')
    result_df_2.to_csv(csv_file_name_2, index=False)


