import numpy as np
import os
import keras
import sklearn.metrics
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import class_weight

import tensorflow as tf
import umap
from sklearn import decomposition
from tensorflow import keras
from sklearn.cluster import KMeans, Birch
from scipy.stats import mode
import statistics
import warnings
import pickle

"""#Download data"""

WRITE_NCOMP = False
max_words = 10000

def get_all_data():
    (x_train_init, y_train_init), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2, seed=42)
    word_index = reuters.get_word_index(path="reuters_word_index.json")

    # of Test Samples: 2246
    # of Classes: 46
    index_to_word = {}
    for key, value in word_index.items():
        index_to_word[value] = key

    tokenizer = Tokenizer(num_words=max_words)
    x_train_init = tokenizer.sequences_to_matrix(x_train_init, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

    x_train, x_val, y_train, y_val = train_test_split(x_train_init, y_train_init, test_size=0.2, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test

# Download train, validation and test sets.
x_train, x_valid, x_test, y_train, y_valid, y_test = get_all_data()

class_names = ['cocoa','grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
   'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
   'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
   'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
   'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']
print("Our class names: {}".format(class_names, ))

"""# Utils"""


def load_pseudolabels(experiment, embedding, clustering, K_, type):

    # Load cluster labels from results of the first pipeline
    clusters_labels_name = str(type) + "_" + str(embedding) + "_" + str(clustering) + "_" + str(K_) + "_" + "experiment" + "_" + str(experiment)
    clusters_labels_loc = os.path.join("experiments_clusters", clusters_labels_name)
    with open(clusters_labels_loc, 'rb') as file:
        y_cluster_labels_test = pickle.load(file)

    return y_cluster_labels_test

"""# Clustering techniques"""


def kmeans(data_train, data_valid, data_test, num_clusters):
    # Data for training
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
    # Data for validation
    data_valid = np.asarray(data_valid)
    data_valid = data_valid.reshape((len(data_valid), np.prod(data_valid.shape[1:])))
    # Data for test
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))
    # Model
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data_train)
    cluster_labels_train = kmeans.labels_
    cluster_labels_valid = kmeans.predict(data_valid)
    cluster_labels_test = kmeans.predict(data_test)
    return cluster_labels_train, cluster_labels_valid, cluster_labels_test, kmeans


def birch(data_train, data_valid, data_test, num_clusters):
    # Data for training
    data_train = np.asarray(data_train)
    data_train = data_train.reshape((len(data_train), np.prod(data_train.shape[1:])))
    # Data for validation
    data_valid = np.asarray(data_valid)
    data_valid = data_valid.reshape((len(data_valid), np.prod(data_valid.shape[1:])))
    # Data for test
    data_test = np.asarray(data_test)
    data_test = data_test.reshape((len(data_test), np.prod(data_test.shape[1:])))
    # Model
    brc = Birch(n_clusters=num_clusters)
    brc.fit(data_train)
    cluster_labels_train = brc.predict(data_train)
    cluster_labels_valid = brc.predict(data_valid)
    cluster_labels_test = brc.predict(data_test)
    return cluster_labels_train, cluster_labels_valid, cluster_labels_test, brc

"""# Define DL Network"""


def nn(x_train, y_train, x_valid, y_valid, x_test, y_test, num_classes, embedding, clustering, experiment, verbose=0):

    # Define name for model to be saved.
    model_path_name = os.path.join('trained_models', 'reuters_cluster_discriminator_' + str(embedding) + '_' + str(clustering) + '_' + str(num_classes) + '_' + str(experiment) + '.h5')
    batch_size = 32

    # Convert labels into categorical matrices
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_valid_cat = keras.utils.to_categorical(y_valid, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    y_train_labels = np.argmax(y_train_cat, axis=1)

    if not os.path.exists(model_path_name):
        max_words = 10000
        # Model configuration
        model = Sequential()
        model.add(Dense(512, input_shape=(max_words,), activation ='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation ='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        epochs = 3

        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        history = model.fit(x_train, y_train_cat, class_weight=class_weight_dict, validation_data=(x_valid, y_valid_cat), batch_size=batch_size, epochs=epochs, verbose=0)
        model.save(os.path.join(model_path_name))

        del history
    else:
        model = keras.models.load_model(model_path_name)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # Generate generalization metrics
    # print(model.metrics)
    # print(model.metrics_names)
    # print(model.compiled_names)
    score = model.evaluate(x_test, y_test_cat, batch_size=batch_size, verbose=0)

    pred_labels = model.predict(x_test, batch_size=batch_size, verbose=0)

    predicted_class_labels = np.argmax(pred_labels, axis=1)
    y_test_labels = np.argmax(y_test_cat, axis=1)

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
    clusters_valid_labels_loc = os.path.join("experiments_clusters", ("valid_" + clusters_labels_name))
    clusters_test_labels_loc = os.path.join("experiments_clusters", ("test_" + clusters_labels_name))

    # Extract instances of a specific label
    x_train_0 = np.asarray(x_train)
    x_valid_0 = np.asarray(x_valid)
    x_test_0 = np.asarray(x_test)

    # print(clusters_labels_loc)
    if not os.path.exists(clusters_test_labels_loc):
        # Merge train, validation and test set by concatenation
        # Obtain set of size len(x_train_0) + len(x_valid_0) + len(x_test_0)
        # First len(x_train_0) elements belong to the training set
        # Next len(x_valid_0) elements belong to the validation set
        # Remaining len(x_test_0) elements belong to the test set
        x_train_test = np.concatenate((x_train_0, x_valid_0, x_test_0), axis=0)
        data = np.asarray(x_train_test)
        data = data.reshape((len(data), np.prod(data.shape[1:])))
        num_components = 0

        ## Apply embedding (PCA)
        if embedding == "pca":
            # Find how many components express at least 90% of the variance
            pca = decomposition.PCA()
            pca.n_components = min(len(data), max_words)
            pca_data = pca.fit_transform(data)
            percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)
            cum_var_explained = np.cumsum(percentage_var_explained)
            for i in range(0, len(cum_var_explained)):
                if cum_var_explained[i] < 0.9:
                    num_components += 1
            pca.n_components = num_components
            data = pca.fit_transform(data)
            del pca
        ## Apply embedding (UMap)
        if embedding == "umap":
            data = x_train_test.reshape((len(x_train_test), np.prod(x_train_test.shape[1:])))
            # Find through PCA how many components express at least 90% of the variance
            pca = decomposition.PCA()
            pca.n_components = min(len(data), max_words)
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
        data_val = data[len(x_train_0):(len(x_train_0) + len(x_valid_0))]
        data_test = data[(len(x_train_0) + (len(x_valid_0))):]

        # Apply clustering
        if clustering == "kmeans":
            y_train_clust, y_valid_clust, y_test_clust, model = kmeans(data_train, data_val, data_test, K)
        if clustering == "birch":
            y_train_clust, y_valid_clust, y_test_clust, model = birch(data_train, data_val, data_test, K)

        silhouette = silhouette_score(data_test, y_test_clust)
        db_index = davies_bouldin_score(data_test, y_test_clust)
        ch_index = calinski_harabasz_score(data_test, y_test_clust)

        y_train_0 = keras.utils.to_categorical(y_train_clust, K)
        y_valid_0 = keras.utils.to_categorical(y_valid_clust, K)
        y_test_0 = keras.utils.to_categorical(y_test_clust, K)

        # Save cluster labels for the training set
        savelabels = open(clusters_train_labels_loc, 'wb')
        pickle.dump(y_train_clust, savelabels)
        savelabels.close()

        # Save cluster labels for the test set
        savelabels = open(clusters_valid_labels_loc, 'wb')
        pickle.dump(y_valid_clust, savelabels)
        savelabels.close()

        # Save cluster labels for the test set
        savelabels = open(clusters_test_labels_loc, 'wb')
        pickle.dump(y_test_clust, savelabels)
        savelabels.close()

        # Save metrics
        metric_loc = os.path.join("metrics", ("silhouette_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(silhouette, savemetric)
        savelabels.close()

        metric_loc = os.path.join("metrics", ("db_index_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(db_index, savemetric)
        savelabels.close()

        metric_loc = os.path.join("metrics", ("ch_index_" + clusters_labels_name))
        savemetric = open(metric_loc, 'wb')
        pickle.dump(ch_index, savemetric)
        savelabels.close()

        file_ncomp.write(embedding + "," + clustering + "," + str(K) + "," + str(exp) + "," + str(num_components) + "\n")
        file_ncomp.flush()
    else:
        # Dummy variables for replication
        y_train_clust = load_pseudolabels(experiment, embedding, clustering, K, "train")
        y_train_0 = keras.utils.to_categorical(y_train_clust, K)

        y_valid_clust = load_pseudolabels(experiment, embedding, clustering, K, "valid")
        y_valid_0 = keras.utils.to_categorical(y_valid_clust, K)

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

    length = len(y_test)
    labels = np.zeros(length)
    for i in range(0, K):
        mask = y_test_clust == i
        labels[mask] = mode(y_test[mask])[0]
    corresponding_cluster_labels = labels

    cluster_accuracy = accuracy_score(y_test, corresponding_cluster_labels)
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
file_out = open("results/pipeline1_results_reuters.csv", "w")
file_out.write("Embedding,Clustering,K,Silhouette,DB_Scores,CH_Scores,ClusterAccuracy\n")
file_out.flush()

# Save cumulative results with report of pairwise weighted accuracy
file_sel = open("results/pipeline1_results_pairwise_reuters.csv", "w")
file_sel.write("Embedding,Clustering,K,PairWiseMinAccuracy,PairWiseMeanAccuracy\n")
file_sel.flush()


# Save number of components by experiment
if not os.path.exists("results/pipeline1_ncomp_reuters.csv"):
    WRITE_NCOMP = True
    file_ncomp = open("results/pipeline1_ncomp_reuters.csv", "w")
    file_ncomp.write("Embedding,Clustering,K,Experiment,Components\n")
    file_ncomp.flush()


for embedding in embeddings:
    for clustering in clusterings:
        minK = len(class_names)
        Ks = [minK]
        prevK_acc = 0
        prev_dev = 0
        increment = len(class_names)
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
                    y_valid_clusters = load_pseudolabels(experiment, embedding, clustering, K, "valid")
                    y_test_clusters = load_pseudolabels(experiment, embedding, clustering, K, "test")

                    min_weighted_accuracy_pair, mean_weighted_accuracy_pair = nn(x_train, y_train_clusters, x_valid, y_valid_clusters, x_test, y_test_clusters, K, embedding, clustering, experiment)

                    experiment_pairwise_cluster_min_weighted_accuracy.append(min_weighted_accuracy_pair)
                    experiment_pairwise_cluster_weighted_accuracy.append(mean_weighted_accuracy_pair)

                file_sel.write(embedding + "," + clustering + "," + str(K) + "," + str(
                    statistics.mean(experiment_pairwise_cluster_min_weighted_accuracy)) + "," + str(
                    statistics.mean(experiment_pairwise_cluster_weighted_accuracy)) + "\n")
                file_sel.flush()

                metric_loc = os.path.join("experiments", ("reuters_" + str(embedding) + "_" + str(
                    clustering) + "_" + "experiment_pairwise_cluster_min_weighted_accuracy"))
                save_acc = open(metric_loc, 'wb')
                pickle.dump(experiment_pairwise_cluster_min_weighted_accuracy, save_acc)
                save_acc.close()

                metric_loc = os.path.join("experiments", ("reuters_" + str(embedding) + "_" + str(
                    clustering) + "_" + "experiment_pairwise_cluster_weighted_accuracy"))
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