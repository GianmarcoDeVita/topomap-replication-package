from __future__ import print_function
import keras
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
import umap
from scipy.stats import mode
from sklearn.cluster import KMeans, Birch
from sklearn import decomposition
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import class_weight
import statistics
import warnings
import pickle

WRITE_NCOMP = False

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
    clusters_labels_loc = os.path.join("experiments_clusters", clusters_labels_name)
    with open(clusters_labels_loc, 'rb') as file:
        y_cluster_labels_test = pickle.load(file)

    return y_cluster_labels_test


"""# Clustering techniques"""


def kmeans(data_train, data_test, num_clusters):
    # Data for training
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
    # Data for test
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))
    # Model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_train)
    cluster_labels_train = kmeans.labels_
    cluster_labels_test = kmeans.predict(data_test)
    return cluster_labels_train, cluster_labels_test, kmeans


def birch(data_train, data_test, num_clusters):
    # Data for training
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
    # Data for test
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))
    # Model
    brc = Birch(n_clusters=num_clusters)
    brc.fit(data_train)
    cluster_labels_train = brc.predict(data_train)
    cluster_labels_test = brc.predict(data_test)
    return cluster_labels_train, cluster_labels_test, brc


"""# Define DL Network"""


def nn(x_train, y_train, x_test, y_test, num_classes, embedding, clustering, experiment, verbose=0):

    model_save_filename = "mnist_cluster_discriminator_" + str(embedding) + "_" + str(
        clustering) + "_" + str(num_classes) + "_" + str(experiment) + ".h5"
    model_path_name = os.path.join("trained_models", model_save_filename)

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

    pred_labels = model.predict(x_test, verbose=0)
    y_test_labels = np.argmax(y_test, axis=1)

    predicted_class_labels = np.argmax(pred_labels, axis=1)
    clusters_in_test_set = np.unique(y_test_labels)

    weighted_accuracy_by_pair_nn = []

    for picked_cluster in clusters_in_test_set:
        for other_cluster in clusters_in_test_set:
            if picked_cluster != other_cluster:
                elements_in_picked_cluster = np.where(y_test_labels == picked_cluster)[0]
                elements_in_other_cluster = np.where(y_test_labels == other_cluster)[0]

                predictions_in_picked_cluster = predicted_class_labels[elements_in_picked_cluster]
                predictions_in_other_cluster = predicted_class_labels[elements_in_other_cluster]

                # Classifications in picked cluster
                correct_predictions_picked_cluster = len(np.where(predictions_in_picked_cluster == picked_cluster)[0])
                mispredictions_picked_cluster = len(np.where(predictions_in_picked_cluster == other_cluster)[0])

                # Classifications in other cluster
                correct_predictions_other_cluster = len(np.where(predictions_in_other_cluster == other_cluster)[0])
                mispredictions_other_cluster = len(np.where(predictions_in_other_cluster == picked_cluster)[0])

                # Compute weighted accuracy
                size_picked_cluster = correct_predictions_picked_cluster + mispredictions_picked_cluster
                size_other_cluster = correct_predictions_other_cluster + mispredictions_other_cluster

                n_AA = correct_predictions_picked_cluster  # Number of elements correctly predicted as belonging to cluster A
                n_BB = correct_predictions_other_cluster  # Number of elements correctly predicted as belonging to cluster B
                n_BA = mispredictions_picked_cluster  # Number of elements belonging to cluster A but predicted to belong to cluster B
                n_AB = mispredictions_other_cluster  # Number of elements belonging to cluster B but predicted to belong to cluster A

                if size_picked_cluster > size_other_cluster:
                    if size_other_cluster == 0:
                        continue
                    w = size_picked_cluster / size_other_cluster
                    wacc_pairs = (n_AA + w * n_BB) / (n_AA + w * n_BB + n_BA + w * n_AB)
                else:
                    if size_picked_cluster == 0:
                        continue
                    w = size_other_cluster / size_picked_cluster
                    wacc_pairs = (w * n_AA + n_BB) / (w * n_AA + n_BB + w * n_BA + n_AB)

                weighted_accuracy_by_pair_nn.append(wacc_pairs)

    del model

    min_weighted_accuracy_pair = np.min(weighted_accuracy_by_pair_nn)
    mean_weighted_accuracy_pair = np.mean(weighted_accuracy_by_pair_nn)

    return min_weighted_accuracy_pair, mean_weighted_accuracy_pair


"""# Prepare Input"""


def selection_of_K_cluster(K, experiment, embedding="none", clustering="kmeans"):

    clusters_labels_name = str(embedding) + "_" + str(clustering) + "_" + str(K) + "_" + "experiment" + "_" + str(experiment)
    clusters_train_labels_loc = os.path.join("experiments_clusters", ("train_" + clusters_labels_name))
    clusters_test_labels_loc = os.path.join("experiments_clusters", ("test_" + clusters_labels_name))

    x_train_0 = np.asarray(x_train)
    x_train_0 = np.reshape(x_train_0, [len(x_train_0), 28, 28, 1])

    x_test_0 = np.asarray(x_test)
    x_test_0 = np.reshape(x_test_0, [len(x_test_0), 28, 28, 1])

    if not os.path.exists(clusters_test_labels_loc):
        # Merge train and test set by concatenation
        # Obtain set of size len(x_train_0) + len(x_test_0)
        # First len(x_train_0) elements belong to the training set
        # Remaining len(x_test_0) elements belong to the test set
        x_train_test = np.concatenate((x_train_0, x_test_0), axis=0)
        data = np.asarray(x_train_test)
        data = data.reshape((len(data), np.prod(data.shape[1:])))
        num_components = 0

        ## Apply embedding (PCA)
        if embedding == "pca":
            # Find how many components express at least 90% of the variance
            pca = decomposition.PCA()
            pca.n_components = min(len(data), len(data[0]))
            pca_data = pca.fit_transform(data)
            percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
            cum_var_explained = np.cumsum(percentage_var_explained)
            for i in range(0, len(cum_var_explained)):
                if cum_var_explained[i] < 0.9:
                    num_components += 1
            pca.n_components = num_components
            data = pca.fit_transform(data)
            del pca

        # Apply embedding (UMap)
        if embedding == "umap":
            data = x_train_test.reshape((len(x_train_test), np.prod(x_train_test.shape[1:])))
            # Find through PCA how many components express at least 90% of the variance
            pca = decomposition.PCA()
            pca.n_components = min(len(data), len(data[0]))
            pca_data = pca.fit_transform(data)
            percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
            cum_var_explained = np.cumsum(percentage_var_explained)
            for i in range(0, len(cum_var_explained)):
                if cum_var_explained[i] < 0.9:
                    num_components += 1
            umap_embedding = umap.UMAP(n_components=num_components)
            data = umap_embedding.fit_transform(data)
            del pca, umap_embedding

        data_train = data[:len(x_train_0)]
        data_test = data[len(x_train_0):]

        # Apply clustering
        if clustering == "kmeans":
            y_train_clust, y_test_clust, model = kmeans(data_train, data_test, K)
        if clustering == "birch":
            y_train_clust, y_test_clust, model = birch(data_train, data_test, K)

        silhouette = silhouette_score(data_test, y_test_clust)
        db_index = davies_bouldin_score(data_test, y_test_clust)
        ch_index = calinski_harabasz_score(data_test, y_test_clust)

        y_train_0 = keras.utils.to_categorical(y_train_clust, K)
        y_test_0 = keras.utils.to_categorical(y_test_clust, K)

        # Save cluster labels for the training set
        savelabels = open(clusters_train_labels_loc, 'wb')
        pickle.dump(y_train_clust, savelabels)
        savelabels.close()

        # Save cluster labels for the test set
        savelabels = open(clusters_test_labels_loc, 'wb')
        pickle.dump(y_test_clust, savelabels)
        savelabels.close()

        # Save metrics
        metric_loc = os.path.join("metrics", ("silhouette_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(silhouette, savemetric)
        savemetric.close()

        metric_loc = os.path.join("metrics", ("db_index_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(db_index, savemetric)
        savemetric.close()

        metric_loc = os.path.join("metrics", ("ch_index_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(ch_index, savemetric)
        savemetric.close()

        file_ncomp.write(embedding + "," + clustering + "," + str(K) + "," + str(exp) + "," + str(
            num_components) + "\n")
        file_ncomp.flush()
    else:
        y_train_clust = load_pseudolabels(experiment, embedding, clustering, K, "train")
        y_train_0 = keras.utils.to_categorical(y_train_clust, K)

        y_test_clust = load_pseudolabels(experiment, embedding, clustering, K, "test")
        y_test_0 = keras.utils.to_categorical(y_test_clust, K)

        # Load metrics
        metric_loc = os.path.join("metrics", ("silhouette_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            silhouette = pickle.load(f)

        metric_loc = os.path.join("metrics", ("db_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            db_index = pickle.load(f)

        metric_loc = os.path.join("metrics", ("ch_index_" + clusters_labels_name))
        with open(metric_loc, 'rb') as f:
            ch_index = pickle.load(f)

    length = len(y_labels_test)
    labels = np.zeros(length)
    for i in range(0, K):
        mask = y_test_clust == i
        labels[mask] = mode(y_labels_test[mask])[0]
    corresponding_cluster_labels = labels

    cluster_accuracy = accuracy_score(y_labels_test, corresponding_cluster_labels)
    #print("Cluster accuracy: ", cluster_accuracy)

    return silhouette, db_index, ch_index, cluster_accuracy


"""# Experiments"""

# Experimental configuration
embeddings = ["pca", "umap"]
clusterings = ["kmeans", "birch"]
experiments = 10

experiment_silhouettes = []
experiment_db_scores = []
experiment_ch_scores = []

experiment_cluster_accuracy = []

# Ignore warning messages in output
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Save cumulative results with report of cluster accuracy
file_out = open("results/pipeline1_results_mnist.csv", "w")
file_out.write("Embedding,Clustering,K,Silhouette,DB_Scores,CH_Scores,ClusterAccuracy\n")
file_out.flush()

# Save cumulative results with report of pairwise weighted accuracy
file_sel = open("results/pipeline1_results_pairwise_mnist.csv", "w")
file_sel.write("Embedding,Clustering,K,PairWiseMinAccuracy,PairWiseMeanAccuracy\n")
file_sel.flush()


# Save number of components by experiment
if not os.path.exists("results/pipeline1_ncomp_mnist.csv"):
    WRITE_NCOMP = True
    file_ncomp = open("results/pipeline1_ncomp_mnist.csv", "w")
    file_ncomp.write("Embedding,Clustering,K,Experiment,Components\n")
    file_ncomp.flush()


for embedding in embeddings:
    for clustering in clusterings:
        minK = 10
        Ks = [minK]
        prevK_acc = 0
        prev_dev = 0
        increment = 10
        prev_K = 0

        derivatives = [0]

        experiment_pairwise_cluster_min_weighted_accuracy = []
        experiment_pairwise_cluster_weighted_accuracy = []

        for K in Ks:
            print("Values of K: ", Ks)
            for exp in range(0, experiments):
                keras.backend.clear_session()
                sil_score, db_score, ch_score, cluster_accuracy = selection_of_K_cluster(K, exp, embedding, clustering)

                experiment_silhouettes.append(sil_score)
                experiment_db_scores.append(db_score)
                experiment_ch_scores.append(ch_score)
                experiment_cluster_accuracy.append(cluster_accuracy)

            print("Embedding: ", embedding, " Clustering: ", clustering, " K: ", K, " Cluster accuracy: ", statistics.mean(experiment_cluster_accuracy))
            file_out.write(embedding + "," + clustering + "," + str(K) + "," + str(
                statistics.mean(experiment_silhouettes)) + "," + str(statistics.mean(experiment_db_scores)) + "," + str(statistics.mean(experiment_ch_scores)) + "," + str(statistics.mean(experiment_cluster_accuracy)) + "\n")
            file_out.flush()

            thisK_acc = statistics.mean(experiment_cluster_accuracy)
            this_der = (thisK_acc - prevK_acc)/(K - prev_K)
            derivatives.append(this_der)

            curr_dev = len(derivatives) - 1
            avg_dev = statistics.mean([derivatives[curr_dev], derivatives[curr_dev - 1]])

            print("This derivative: ", this_der)
            print("Average of the last two derivatives: ", avg_dev)
            print("--------------------------------------------------------------------")
            if avg_dev > 0.001:
                prevK_acc = thisK_acc
                prev_K = K
                Ks.append(K+increment)
            else:
                for experiment in range(0, experiments):

                    y_train_clusters = load_pseudolabels(experiment, embedding, clustering, K, "train")
                    y_train_0 = keras.utils.to_categorical(y_train_clusters, K)

                    y_test_clusters = load_pseudolabels(experiment, embedding, clustering, K, "test")
                    y_test_0 = keras.utils.to_categorical(y_test_clusters, K)

                    min_weighted_accuracy_pair, mean_weighted_accuracy_pair = nn(x_train, y_train_0, x_test, y_test_0, K, embedding, clustering, experiment)

                    experiment_pairwise_cluster_min_weighted_accuracy.append(min_weighted_accuracy_pair)
                    experiment_pairwise_cluster_weighted_accuracy.append(mean_weighted_accuracy_pair)

                file_sel.write(embedding + "," + clustering + "," + str(K) + "," + str(
                    statistics.mean(experiment_pairwise_cluster_min_weighted_accuracy)) + "," + str(
                    statistics.mean(experiment_pairwise_cluster_weighted_accuracy)) + "\n")
                file_sel.flush()

                metric_loc = os.path.join("experiments", ("mnist_" + str(embedding) + "_" + str(clustering) + "_" + "experiment_pairwise_cluster_min_weighted_accuracy"))
                save_acc = open(metric_loc, 'wb')
                pickle.dump(experiment_pairwise_cluster_min_weighted_accuracy, save_acc)
                save_acc.close()

                metric_loc = os.path.join("experiments", ("mnist_" + str(embedding) + "_" + str(clustering) + "_" + "experiment_pairwise_cluster_weighted_accuracy"))
                save_acc = open(metric_loc, 'wb')
                pickle.dump(experiment_pairwise_cluster_weighted_accuracy, save_acc)
                save_acc.close()

                experiment_pairwise_cluster_min_weighted_accuracy = []
                experiment_pairwise_cluster_weighted_accuracy = []

            experiment_silhouettes = []
            experiment_db_scores = []
            experiment_ch_scores = []

            experiment_cluster_accuracy = []

if WRITE_NCOMP:
    file_ncomp.close()
file_out.close()