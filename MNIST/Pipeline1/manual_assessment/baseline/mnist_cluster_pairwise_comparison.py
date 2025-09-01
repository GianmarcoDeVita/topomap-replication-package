from __future__ import print_function
import keras
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
import umap
from scipy.stats import mode
from sklearn.cluster import KMeans, Birch
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import class_weight
import statistics
import warnings
import pickle
import pandas as pd


"""#Download data"""


def get_data_mnist():
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    (img_rows, img_cols) = (28, 28)
    num_classes = 10

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_labels_train = y_train
    y_labels_test = y_test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test, y_labels_train, y_labels_test


(x_train, x_test, y_train, y_test, y_labels_train, y_labels_test) = get_data_mnist()


"""# Utils"""


def load_pseudolabels(experiment, embedding, clustering, K_, type):

    # Load cluster labels from results of the first pipeline
    clusters_labels_name = str(type) + "_" + str(embedding) + "_" + str(clustering) + "_" + str(K_) + "_" + "experiment" + "_" + str(experiment)
    clusters_labels_loc = os.path.join("../../experiments_clusters", clusters_labels_name)
    with open(clusters_labels_loc, 'rb') as file:
        y_cluster_labels_test = pickle.load(file)

    return y_cluster_labels_test


"""# Define DL Network"""


def nn(x_train, y_train, x_test, y_test, x_test_8, y_test_8, clust_a, clust_b, num_classes, embedding, clustering, experiment, verbose=0):

    model_save_filename = "mnist_binary_cluster_discriminator_" + str(embedding) + "_" + str(
        clustering) + "_" + str(clust_a) + "_" + str(clust_b) + "_" + str(num_classes) + "_" + str(experiment) + ".h5"
    model_path_name = os.path.join("trained_binary_models", model_save_filename)

    y_train_labels = np.argmax(y_train, axis=1)

    if not os.path.exists(model_path_name):
        batch_size = 128
        epochs = 12
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

        if verbose == 1:
            print(x_train.shape)
            print(y_train.shape)
            print(x_test.shape)
            print(y_test.shape)

        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        model.fit(x_train, y_train, class_weight=class_weight_dict, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test))
        model.save(model_path_name)
    else:
        model = keras.models.load_model(model_path_name)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(learning_rate=1.0),
                      metrics=['accuracy'])

    score = model.evaluate(x_test, y_test, verbose=0)
    score_8 = model.evaluate(x_test_8, y_test_8, verbose=0)

    del model

    return score[1], score_8[1]


num_experiments = 10

# Save cumulative results with report of cluster accuracy
file_out = open("mnist_cluster_pairwise_dnn_comparison.csv", "w")
file_out.write("Example,Embedding,Clustering,K,ClusterA,ClusterB,PairWiseAccuracy,AccuracyScore,AccuracyScore8\n")
file_out.flush()

# Load list of cluster pairs
df = pd.read_csv("../pairwise_cluster_comparison.csv", delimiter=',')

for index, row in df.iterrows():

    # Extract row containing the information for this iteration
    example_from_file = row['Example']
    pairwise_acc_from_file = row['PairWiseAccuracy']

    raw_labels_file_name = row['Filename']
    labels_file_name = raw_labels_file_name.split('_')

    # Extract settings from file name
    embedding_from_filename = labels_file_name[2]
    clustering_from_filename = labels_file_name[3]
    K_from_filename = labels_file_name[4]
    experiment_from_filename =  labels_file_name[6]

    # Labels of the two clusters
    cluster_a = row['ClusterA']
    cluster_b = row['ClusterB']

    # Load cluster labels
    y_cluster_labels_train = load_pseudolabels(experiment_from_filename, embedding_from_filename, clustering_from_filename, K_from_filename, 'train')
    y_cluster_labels_test = load_pseudolabels(experiment_from_filename, embedding_from_filename, clustering_from_filename, K_from_filename, 'test')

    # Select indices of the elements that are in clusters A and B
    train_elems_of_clust_a_and_b = np.where((y_cluster_labels_train == cluster_a) | (y_cluster_labels_train == cluster_b))[0]
    test_elems_of_clust_a_and_b = np.where((y_cluster_labels_test == cluster_a) | (y_cluster_labels_test == cluster_b))[0]

    x_train_clust_a_and_b = x_train[train_elems_of_clust_a_and_b]
    x_test_clust_a_and_b = x_test[test_elems_of_clust_a_and_b]

    y_train_clust_a_and_b_disc = y_cluster_labels_train[train_elems_of_clust_a_and_b]
    y_test_clust_a_and_b_disc = y_cluster_labels_test[test_elems_of_clust_a_and_b]

    y_train_clust_a_and_b_disc = (y_train_clust_a_and_b_disc == cluster_b).astype(int)
    y_test_clust_a_and_b_disc = (y_test_clust_a_and_b_disc == cluster_b).astype(int)

    y_train_clust_a_and_b = keras.utils.to_categorical(y_train_clust_a_and_b_disc, 2)
    y_test_clust_a_and_b = keras.utils.to_categorical(y_test_clust_a_and_b_disc, 2)

    # Test set and labels equal to the one administered to the users in the manual assessment task
    clust_a_sel_indices = 'Example_' + str(example_from_file) + '_test_emb_' + str(embedding_from_filename) + '_clust_' \
                          + str(clustering_from_filename) + '_k_' + str(K_from_filename) + '_experiment_' \
                          + str(experiment_from_filename) + "_Cluster_A.npy"
    clust_b_sel_indices = 'Example_' + str(example_from_file) + '_test_emb_' + str(embedding_from_filename) + '_clust_' \
                          + str(clustering_from_filename) + '_k_' + str(K_from_filename) + '_experiment_' \
                          + str(experiment_from_filename) + "_Cluster_B.npy"

    indices_cluster_a = np.load("../indices/" + clust_a_sel_indices)
    indices_cluster_b = np.load("../indices/" + clust_b_sel_indices)

    sel_cluster_a = indices_cluster_a[-5:]
    sel_cluster_b = indices_cluster_b[-5:]

    x_test_4_clust_a = x_test[sel_cluster_a]
    x_test_4_clust_b = x_test[sel_cluster_b]

    x_test_8_clust_a_and_b = np.concatenate((x_test_4_clust_a, x_test_4_clust_b), axis=0)

    y_test_8_clust_a_and_b_disc = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_test_8_clust_a_and_b = keras.utils.to_categorical(y_test_8_clust_a_and_b_disc, 2)

    dnn_pairwise_comparison_scores = []
    dnn_pairwise_comparison_scores8 = []

    for experiment in range(0, num_experiments):

        score, score_8 = nn(x_train_clust_a_and_b, y_train_clust_a_and_b, x_test_clust_a_and_b, y_test_clust_a_and_b,
                            x_test_8_clust_a_and_b, y_test_8_clust_a_and_b, cluster_a, cluster_b,
                            2, embedding_from_filename, clustering_from_filename, experiment)

        dnn_pairwise_comparison_scores.append(score)
        dnn_pairwise_comparison_scores8.append(score_8)

    file_out.write(str(example_from_file) + "," + str(embedding_from_filename) + "," + str(clustering_from_filename)
                   + "," + str(K_from_filename) + "," + str(cluster_a) + "," + str(cluster_b) + ","
                   + str(pairwise_acc_from_file) + "," + str(statistics.mean(dnn_pairwise_comparison_scores)) + ","
                   + str(statistics.mean(dnn_pairwise_comparison_scores8)) + "\n")
    file_out.flush()

file_out.close()

