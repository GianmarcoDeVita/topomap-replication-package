from __future__ import print_function
import sys
import os

import numpy as np
import pandas as pd
import pickle
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
np.set_printoptions(threshold=sys.maxsize)

"""#Download data"""
max_words = 10000

def get_test_data():

    (x_train_init, y_train_init), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2, seed=42)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    # of Test Samples: 2246
    # of Classes: 46
    index_to_word = {}
    for key, value in word_index.items():
        index_to_word[value] = key

    tokenizer = Tokenizer(num_words=max_words)
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    return x_test, y_test

(x_test, y_test) = get_test_data()

class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']
print("Our class names: {}".format(class_names, ))

## Prepare Input

def get_clusters(experiment, embedding, clustering, K_=None):

    # Load cluster labels from results of the first pipeline
    y_cluster_labels_test = None
    clusters_labels_name = "test_" + str(embedding) + "_" + str(clustering) + "_" + "experiment" + "_" + str(experiment)
    if K_:
        clusters_labels_name = "test_" + str(embedding) + "_" + str(clustering) + "_" + str(K_) + "_" + "experiment" + "_" + str(experiment)
    clusters_labels_loc = os.path.join("../Pipeline1/experiments_clusters", clusters_labels_name)
    with open(clusters_labels_loc, 'rb') as file:
        y_cluster_labels_test = pickle.load(file)

    return y_cluster_labels_test

# Experimental run to be considered
experiment_n = 0

# Load results obtained from first pipeline
df = pd.read_csv('../Pipeline1/results/pipeline1_results_pairwise_reuters.csv', sep=',')

# Configuration selection criterion
selected_configuration = df.loc[df["PairWiseMinAccuracy"].idxmax()]

# Configuration
embedding = selected_configuration['Embedding']
clustering = selected_configuration['Clustering']
K = int(selected_configuration['K'])

cluster_rho_a_exps = []
max_size = -1

# Experimental run to be considered
for experiment_n in range(0, 10):

    # Get cluster labels
    y_cluster_labels_test = get_clusters(experiment_n, embedding, clustering, K)

    # Get killable mutants
    df = pd.read_csv('./misclassifications_distribution/reuters_misclassification_distribution_and_mutant_killing.csv', delimiter=',')
    killable_mutants_ = df.index[df['Killed_WLX'] == True].tolist()

    # Load selected clusters by mutant
    selected_clusters_by_mutant = []

    for mutant in killable_mutants_:
      selected_clusters_file = "selected_clusters_" + str(embedding) + "_" + str(clustering) + "_" + str(K) + "_" + str(experiment_n) + "_" + str(mutant) + ".npy"
      selected_clusters_loc = os.path.join("./selected_clusters", selected_clusters_file)
      selected_clusters = np.load(selected_clusters_loc)

      # If selected clusters [1, 0, 1, 0, 1 ...]
      selected_clusters_by_mutant.append(selected_clusters)

    selected_clusters_by_mutant = np.asarray(selected_clusters_by_mutant)
    selected_clusters_by_mutant = np.transpose(selected_clusters_by_mutant)

    # Sort row of matrix
    row_sums = np.sum(selected_clusters_by_mutant, axis=1)
    sorted_rows = np.argsort(row_sums)[::-1]
    selected_clusters_by_mutant = selected_clusters_by_mutant[sorted_rows.astype(int)]

    # Get statistics on selected clusters
    clusters_rho_a = []
    for row in range(len(selected_clusters_by_mutant)):
        clusters_rho_a.append(np.sum(selected_clusters_by_mutant[row]/len(selected_clusters_by_mutant[row])))

    if len(clusters_rho_a) > max_size:
        max_size = len(clusters_rho_a)

    clusters_rho_a.sort(reverse=True)
    cluster_rho_a_exps.append(clusters_rho_a)

for column in range(0, len(cluster_rho_a_exps)):
    while len(cluster_rho_a_exps[column]) < max_size:
        cluster_rho_a_exps[column].append(0)

cluster_rho_a_exps = np.asarray(cluster_rho_a_exps)
cluster_rho_a_s = np.mean(cluster_rho_a_exps, axis=0)
cluster_rho_a_s = np.sort(cluster_rho_a_s)[::-1]

print("---------------------------")
print("---------------------------")
print("Most selected cluster (MSC)")

print("First: ", cluster_rho_a_s[0])
print("Second: ", cluster_rho_a_s[1])
print("Third: ", cluster_rho_a_s[2])

print("---------------------------")
print("---------------------------")
print("Clusters with rho_a greater than:")

for perc in [0.50, 0.25, 0.10]:
    print(perc, "% ", sum(item > perc for item in cluster_rho_a_s)/len(cluster_rho_a_s))