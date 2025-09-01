import pandas as pd
import numpy as np
import random as rnd
import pickle
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

seed = 0
rnd.seed(seed)
np.random.seed(seed)

"""#Download data"""

def get_all_data():
    validation_split = 0.2
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Determine shape of the data
    img_width, img_height, img_num_channels = 32, 32, 3
    input_shape = (img_width, img_height, img_num_channels)
    # Parse numbers as floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalize data
    x_train = x_train / 255
    x_test = x_test / 255

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split, random_state=0)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

# Download train, validation and test sets.
x_train, x_valid, x_test, y_train, y_valid, y_test = get_all_data()

"""Loading Settings"""

df = pd.read_csv("../results/pipeline1_pairwise_accuracies.csv", delimiter=',')

# Find the row with the maximum PairwiseAccuracy
row_with_max_pairwise_accuracy = df.loc[df['PairWiseAccuracy'].idxmax()]

# Extract the Cluster1 and Cluster2 with the maximum PairwiseAccuracy
max_embedding = row_with_max_pairwise_accuracy['Embedding']
max_cluster = row_with_max_pairwise_accuracy['Clustering']
max_k = row_with_max_pairwise_accuracy['K']
max_exp = row_with_max_pairwise_accuracy['Experiment']
max_cluster1 = row_with_max_pairwise_accuracy['Cluster1']
max_cluster2 = row_with_max_pairwise_accuracy['Cluster2']
max_pw_acc = row_with_max_pairwise_accuracy['PairWiseAccuracy']

# Find the row with the minimum PairwiseAccuracy
row_with_min_pairwise_accuracy = df.loc[df['PairWiseAccuracy'].idxmin()]

# Extract the Cluster1 and Cluster2 with the minimum PairwiseAccuracy
min_embedding = row_with_min_pairwise_accuracy['Embedding']
min_cluster = row_with_min_pairwise_accuracy['Clustering']
min_k = row_with_min_pairwise_accuracy['K']
min_exp = row_with_min_pairwise_accuracy['Experiment']
min_cluster1 = row_with_min_pairwise_accuracy['Cluster1']
min_cluster2 = row_with_min_pairwise_accuracy['Cluster2']
min_pw_acc = row_with_min_pairwise_accuracy['PairWiseAccuracy']

print(f"Minimum PairwiseAccuracy: Embedding: {max_embedding}, Clustering: {max_cluster}, K: {max_k}, Experiment: {max_exp}, Cluster1: {max_cluster1}, Cluster2: {max_cluster2}, PairWiseAccuracy: {max_pw_acc}")
print(f"Minimum PairwiseAccuracy: Embedding: {min_embedding}, Clustering: {min_cluster}, K: {min_k}, Experiment: {min_exp}, Cluster1: {min_cluster1}, Cluster2: {min_cluster2}, PairWiseAccuracy: {min_pw_acc}")

# Find distribution of pairwise accuracies
unique_pairwise_accuracies = np.unique(df['PairWiseAccuracy'])
unique_pairwise_accuracies = np.unique(np.around(unique_pairwise_accuracies, 2).tolist())
#print(unique_pairwise_accuracies)
#print(len(unique_pairwise_accuracies))

# Select random samples from pairwise accuracies
indices_to_sample = range(0, len(unique_pairwise_accuracies) - 1)

if len(unique_pairwise_accuracies) - 1 < 20:
    # Random selection with replacement
    random_sample_selection = rnd.choices(indices_to_sample, k=20)
else:
    # Random selection without replacement
    random_sample_selection = rnd.sample(indices_to_sample, 20)

# Select 10 elements whose accuracy is 1.0:
indices_for_this_accuracy = df[abs(df['PairWiseAccuracy'] - 1.0) == 0].index.tolist()
random_ten_samples_for_this_accuracy = rnd.sample(indices_for_this_accuracy, 10)

# Select 20 elements whose accuracy is not 1.0:
indices_for_these_accuracies = []
for elem in random_sample_selection:
    index_for_this_accuracy = df[(abs(df['PairWiseAccuracy'] - unique_pairwise_accuracies[elem]) <= 0.01) & (df['PairWiseAccuracy'] < 1.0)].index.tolist()
    sel_index_for_this_accuracy = rnd.sample(index_for_this_accuracy, 1)
    indices_for_these_accuracies.append(sel_index_for_this_accuracy)

# Create log file
file_out = open("pairwise_cluster_comparison.csv", "w")
file_out.write("Example,Filename,ClusterA,ClusterB,PairWiseAccuracy\n")
file_out.flush()


examples = 30

for example in range(0, examples):
    if example < 10:
        df_this_index = random_ten_samples_for_this_accuracy[example]
    else:
        df_this_index = indices_for_these_accuracies[example - 10][0]

    df_this_row = df.iloc[df_this_index]

    this_row_embedding = df_this_row['Embedding']
    this_row_clustering = df_this_row['Clustering']
    this_row_k = df_this_row['K']
    this_row_exp = df_this_row['Experiment']
    this_row_cluster1 = df_this_row['Cluster1']
    this_row_cluster2 = df_this_row['Cluster2']
    this_row_pw_acc = df_this_row['PairWiseAccuracy']

    # Load corresponding experiment data
    pickle_file_name = '../experiments_clusters/test_' + str(this_row_embedding) + '_' + str(this_row_clustering) + '_' + str(this_row_k) + '_experiment_' + str(this_row_exp)
    with open(pickle_file_name, 'rb') as f:
        loaded_data = pickle.load(f)

    file_out.write(str(example) + "," + pickle_file_name + "," + str(this_row_cluster1) + "," + str(this_row_cluster2) + "," + str(this_row_pw_acc) + "\n")
    file_out.flush()

    indices_m_c1 = [i for i, x in enumerate(loaded_data) if x == this_row_cluster1]
    indices_m_c2 = [i for i, x in enumerate(loaded_data) if x == this_row_cluster2]

    for iteration in range(0, 2):
        if iteration == 0:
            cluster_name = "Cluster A"
            # Select 20 random indices from cluster
            random_indices = np.random.choice(indices_m_c1, size=20, replace=False)
        else:
            cluster_name = "Cluster B"
            # Select 20 random indices from cluster
            random_indices = np.random.choice(indices_m_c2, size=20, replace=True)

        # Create a figure with subplots for each image
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))

        # Iterate through the random indices and plot the corresponding images
        for i, idx in enumerate(random_indices):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(x_test[idx].reshape(32, 32, 3))
            axes[row, col].axis('off')

        plt.tight_layout()
        fig_name = 'Example_' + str(example) + '_test_emb_' + str(this_row_embedding) + '_clust_' + str(this_row_clustering) + '_k_' + str(this_row_k) + '_experiment_' + str(this_row_exp) + "_" + str(cluster_name.replace(" ", "_"))
        plt.savefig('./imgs/' + fig_name + '.eps', format='eps', bbox_inches='tight')
        plt.savefig('./imgs/pngs/' + fig_name + '.png', format='png', bbox_inches='tight')
        plt.close()
        #plt.show()

        # Save the randomly chosen indices
        np.save('./indices/' + fig_name + ".npy", random_indices)
