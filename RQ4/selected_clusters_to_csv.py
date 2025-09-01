import csv
import os
import numpy as np
import pandas as pd

from glob import glob


def update_max_cluster_num(current_max, new_val):
    if current_max < new_val:
        return new_val
    else:
        return current_max




# def get_mutation_params(mutant):
#     mutant = mutant.split("_mutated0_MP_")
#     op_name = mutant[0]
#
#     mutant_config = None
#
#     if op_type == 'BS':
#         mutant_params = mutant[1].split("_")
#
#         if mutant_params[0] == 'False':
#             mutant_config = mutant_params[1]
#         else:
#             mutant_config = mutant_params[0]
#
#         mutant_config = float(mutant_config)
#     elif op_type == 'NON-BS':
#         mutant_config = mutant[1].split("_HIST")
#         mutant_config = mutant_config[0].rstrip('_')
#
#         if op_name == 'mnist_disable_batching':
#             mutant_config = '0.0'
#
#     return mutant_config, op_data


subjects = [
'audio','mnist','cifar10','udacity','reuters','lenet'
#     'audio',
    #'mnist',
    # 'cifar10',
    # 'udacity',
    # 'reuters',
    # 'lenet'
]

root_folder = '/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters'
killability_folder = '/Users/nhumbatova/Documents/Projects/TopoMap/RQ_2_SVD/selected_clusters/killability_info/'

for subj in subjects:
    killability_df = pd.read_csv(os.path.join(killability_folder, subj+'.csv'))
    killability_dict = dict(zip(killability_df['Mutant'], killability_df['Killed_WLX']))
    # print(killability_dict)

    subj_folder = os.path.join(root_folder, 'selected_clusters_' + subj)
    res_files = glob(os.path.join(subj_folder, '*.npy'))
    subj_results = []
    max_cluster_num = 0
    iter_cluster_num = {}

    for res_file in res_files:
        res_name = res_file.replace(subj_folder, '').split(subj+'_')

        mutant_name_full = subj + '_' + res_name[1].replace('.npy', '')
        # print(mutant_name_full)
        # print(killability_dict[mutant_name_full])

        if killability_dict[mutant_name_full]:
            itr = res_name[0].split('_')[-2]

            mutant = res_name[1].split("_mutated0_MP_")
            op_name = mutant[0]
            config = mutant[1].replace('_.npy', '')

            res_kill = np.load(res_file)
            res_kill = res_kill.tolist()
            max_cluster_num = update_max_cluster_num(max_cluster_num, len(res_kill))
            if itr not in iter_cluster_num:
                iter_cluster_num[itr] = len(res_kill)

            res_kill.insert(0, iter_cluster_num[itr])
            res_kill.insert(0, itr)
            res_kill.insert(0, config)
            res_kill.insert(0, op_name)
            subj_results.append(res_kill)

    header = ['MO', 'CONFIG', 'ITR', 'NUM_CLUSTER']
    for i in range(max_cluster_num):
        header.append('CLUSTER_'+str(i+1))

    subj_results.insert(0, header)

    with open(os.path.join(root_folder, 'selected_clusters_aggregated', subj + '.csv'), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n', )
        writer.writerows(subj_results)