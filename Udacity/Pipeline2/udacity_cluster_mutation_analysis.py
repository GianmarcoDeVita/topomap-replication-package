from __future__ import print_function
import keras
import os

import numpy as np
import pandas as pd
import statistics
import warnings
import pickle
import json
import tensorflow
from tensorflow.keras.optimizers import Adam
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw
from scipy.stats import ranksums

"""#Download data"""

def get_test_data():

  dataset_folder = os.path.join('../Pipeline1')

  x_test_ = np.load(os.path.join(dataset_folder, 'dataset/udacity_test_x.npy'))
  y_test_ = np.load(os.path.join(dataset_folder, 'dataset/udacity_test_y.npy'))

  print("Test set: " + str(len(x_test_)) + " elements")

  return x_test_, y_test_

# Get test data
(x_test, y_test) =  get_test_data()


"""# Prepare Input"""


# Load cluster labels from the results of the cluster generation pipeline
def get_clusters(experiment, embedding, clustering, K_=None):

    y_cluster_labels_test = None
    clusters_labels_name = "test_"+ str(embedding) + "_" + str(clustering) + "_" + "experiment" + "_" + str(experiment)
    if K_:
        clusters_labels_name = "test_"+ str(embedding) + "_" + str(clustering) + "_" + str(K_) + "_" + "experiment" + "_" + str(experiment)
    clusters_labels_loc = os.path.join("../Pipeline1/experiments_clusters", clusters_labels_name)
    with open(clusters_labels_loc, 'rb') as file:
        y_cluster_labels_test = pickle.load(file)

    return y_cluster_labels_test


"""Evaluate model predictions"""

def evaluate_model_global_predictions(original_model_preds, mutated_model_preds):

    # Create matrix instances x runs:
    #   - 1 if instance i is predicted correctly
    #   - 0 if instance i is misclassified
    correct_predictions_original = []
    correct_predictions_mutated = []

    for run in range(0, runs):
        # Correct per run
        correct_at_this_run_original = []
        correct_at_this_run_mutated = []

        # Get accuracies from models
        original_correct_predictions = 0
        mutant_correct_predictions = 0

        for instance in range(0, len(y_test)):
            if np.abs(original_model_preds[run][instance] - y_test[instance]) < 0.3:
                original_correct_predictions = original_correct_predictions + 1
                correct_at_this_run_original.append(1)
            else:
                correct_at_this_run_original.append(0)
            if np.abs(mutated_model_preds[run][instance] - y_test[instance]) < 0.3:
                mutant_correct_predictions = mutant_correct_predictions + 1
                correct_at_this_run_mutated.append(1)
            else:
                correct_at_this_run_mutated.append(0)

        correct_predictions_original.append(correct_at_this_run_original)
        correct_predictions_mutated.append(correct_at_this_run_mutated)

    # Compute the mean in correct predictions by instance for this cluster
    correct_predictions_mean_original = np.mean(correct_predictions_original, axis=0)
    correct_predictions_mean_mutated = np.mean(correct_predictions_mutated, axis=0)

    # Get a vector of binary labels:
    #   - 1 if instance is a contributor
    #   - 0 if instance is not a contributor
    is_a_contributor = correct_predictions_mean_original > correct_predictions_mean_mutated

    return np.asarray(correct_predictions_original), np.asarray(correct_predictions_mutated), np.asarray(is_a_contributor)

def make_predictions(model):

    predicted_model = []

    for run in range(0, runs):
        keras.backend.clear_session()
        loaded_model = tensorflow.keras.models.load_model(model + str(run) + ".h5")
        loaded_model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

        # Get predicted labels from models
        pred_model = loaded_model.predict(x_test, verbose=0)
        pred_labels = np.asarray(np.concatenate(pred_model).ravel().tolist())
        #print(pred_labels)

        predicted_model.append(pred_labels)

    model_predictions = predicted_model
    model_name = model.split('/')
    np.save("./predictions/" + str(model_name[1]) + "predicted.npy", model_predictions)
    print(str(model_name[1] + " predictions saved"))

"""Analysis"""


def estimate_contributors(pred_labels_original, pred_labels_mutated):

    # Check if sizes are equal
    tot = len(pred_labels_original)
    if tot != len(pred_labels_mutated):
        return -1

    count = 0

    # A contributor is defined as an instance of the test set
    # which is predicted correctly by the original model, but
    # then is predicted incorrectly by the mutated model.
    for i in range(0, tot):
        if pred_labels_mutated[i] < pred_labels_original[i]:
            count = count + 1

    return count, tot


"""Statistical test"""

STAT_TEST = "WLX"


# calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


# calculates whether two accuracy arrays are statistically different according to GLM
def is_diff_sts(orig_accuracy_list, accuracy_list, threshold=0.05):
    # print("_"*50)
    # print(orig_accuracy_list)
    # print("_"*50)
    # print((accuracy_list))
    # print("_"*50)

    if STAT_TEST == "WLX":
        p_value = p_value_wilcoxon(orig_accuracy_list, accuracy_list)
    elif STAT_TEST == "GLM":
        p_value = p_value_glm(orig_accuracy_list, accuracy_list)

    effect_size = cohen_d(orig_accuracy_list, accuracy_list)

    # if properties.MODEL_TYPE == 'regression':
    #     is_sts = ((p_value < threshold) and effect_size <= -0.5)
    # else:
    #     is_sts = ((p_value < threshold) and effect_size >= 0.5)

    is_sts = ((p_value < threshold) and (effect_size <= -0.5 or effect_size >= 0.5))

    return is_sts, p_value, effect_size


def p_value_wilcoxon(orig_accuracy_list, accuracy_list):
    w, p_value_w = ranksums(orig_accuracy_list, accuracy_list)

    return p_value_w


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(effect_size=eff_size,
                                           nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow


"""Main"""

# Mutants
mutants_udacity = open('./udacity_mutants.json')
mutants_udacity_loaded = json.load(mutants_udacity)
mutants = []
for mutant in mutants_udacity_loaded:
    for mut_score in mutants_udacity_loaded[mutant]:
        if mutant in ["udacity_change_learning_rate"]:
            name_mutant = str(mutant) + str("_mutated0_MP_False_") + str(mut_score) + "_"
        elif mutant in ["udacity_change_epochs"]:
            name_mutant = str(mutant) + str("_mutated0_MP_50_") + str(mut_score) + "_"
        else:
            name_mutant = str(mutant) + str("_mutated0_MP_") + str(mut_score) + "_"
        if mut_score == "":
            name_mutant = str(mutant) + str("_mutated0_MP_")
        mutants.append(name_mutant)

experiments = 10
runs = 20  # Number of runs

# Ignore warning messages in output
warnings.filterwarnings("ignore")

# Save mutation analysis results
file_out = open("results/pipeline2_results_udacity.csv", "w")
file_out.write(
    "Mutant,Embedding,Clustering,KOfFilename,K,InputsInAggregation,ContributorsDensityInAggregation,GiniImpurityInAggregation,ContributorsShareInAggregation,KilledMutant\n")
file_out.flush()

# Load results obtained from first pipeline
df = pd.read_csv('../Pipeline1/results/pipeline1_results_pairwise_udacity.csv', sep=',')

"""Configuration selection criterion"""
selected_configuration = df.loc[df["PairWiseMinAccuracy"].idxmax()]

# Infer predictions from the original models and mutants
if not os.path.exists("predictions/" + "udacity_original_predicted.npy"):
    make_predictions("original/udacity_original_")
for model_ in mutants:
    if not os.path.exists("predictions/" + model_ + "predicted.npy"):
        make_predictions("mutants/" + model_)

for mutant in mutants:

    # Experiment configuration
    embedding = selected_configuration['Embedding']
    clustering = selected_configuration['Clustering']
    K = int(selected_configuration['K'])
    KofFilename = K

    print((str(mutant) + " " + str(embedding) + " " + str(clustering) + " " + str(K)))

    # List of actual values of K by experiment
    Klist = []

    # Gini Impurities by experiment
    gini_impurities_by_exp = []

    # Percentage of clusters in aggregation by experiment
    clusters_in_aggregation_by_exp = []
    # Share of contributors in aggregation by experiment
    contributors_share_in_aggregation_by_exp = []
    # Percentage of contributors in aggregation by experiment
    contributors_concentration_in_aggregation_by_exp = []
    # Raw number of contributors in aggregation by experiment
    contributors_in_aggregation_by_exp = []
    # Raw number of inputs in aggregation by experiment
    raw_inputs_in_aggregation_by_exp = []
    # Percentage of inputs in aggregation by experiment
    inputs_in_aggregation_by_exp = []
    # Gini Impurity of aggregation by experiment
    gini_impurities_in_aggregation_by_exp = []

    # Is aggregation discriminative at each experiment?
    discriminative_aggregation_by_exp = []

    # Load model predictions
    predictions_original_ = np.load("predictions/" + "udacity_original_predicted" + ".npy")
    predictions_mutated_ = np.load("predictions/" + mutant + "predicted.npy")

    # correct_predictions_original: Get list of binary labels:
    #   -   1 if the instance is correctly predicted by the original model
    #   -   0 if the instance is misclassified by the original model

    # correct_predictions_mutated: Get list of binary labels:
    #   -   1 if the instance is correctly predicted by the mutated model
    #   -   0 if the instance is misclassified by the mutated model

    # is_a_contributor: Get list of binary labels:
    #   -   1 if it is a contributor
    #   -   0 if it is not a contributor
    correct_predictions_original, correct_predictions_mutated, is_a_contributor = evaluate_model_global_predictions(
        predictions_original_, predictions_mutated_)

    # Total contributors in the test set
    total_contributors = np.sum(is_a_contributor)

    for exp in range(0, experiments):

        # Get cluster labels
        y_cluster_labels_test = get_clusters(exp, embedding, clustering, KofFilename)

        ## Fix number of clusters in case of inconsistencies
        # For example, it may be that some clusters are labelled with -1 (e.g. outliers' cluster in HDBScan)
        # or, for example, not all clusters are represented in the test set, hence rescale
        # Replace potential -1 cluster label (usually with HDBScan) with max+1
        y_cluster_labels_test[y_cluster_labels_test < 0] = np.max(y_cluster_labels_test) + 1

        # Extract actual number of clusters in test set:
        actual_cluster_labels_in_test = np.unique(y_cluster_labels_test)
        # Fix K with actual number of clusters
        K = len(actual_cluster_labels_in_test)
        Klist.append(K)
        # Update cluster labels so that if there are k < K clusters they are progressively enumerated from 0 to k-1
        for index in range(0, K):
            y_cluster_labels_test[y_cluster_labels_test == actual_cluster_labels_in_test[index]] = index

        # Statistics on contributors by cluster
        contributors_by_cluster = []
        contributors_concentration_by_cluster = []
        contributors_share_by_cluster = []
        gini_impurity_by_cluster = []
        inputs_share_by_cluster = []

        for k in range(0, K):

            # Get indices for elements of this cluster
            this_cluster_indices = np.where(y_cluster_labels_test == k)[0]

            contributors_by_cluster.append(np.sum(is_a_contributor[this_cluster_indices]))
            contributors_concentration_by_cluster.append(np.sum(is_a_contributor[this_cluster_indices]/np.sum(this_cluster_indices)))
            contributors_share_by_cluster.append(np.sum(is_a_contributor[this_cluster_indices]/np.sum(is_a_contributor)))
            inputs_share_by_cluster.append(np.sum(this_cluster_indices)/len(y_cluster_labels_test))

            # Compute Gini Impurity Index
            gini_impurity_index = 1 - (np.sum(is_a_contributor[this_cluster_indices]) / np.sum(this_cluster_indices)) ** 2 - (1 - np.sum(is_a_contributor[this_cluster_indices]) / np.sum(this_cluster_indices)) ** 2
            gini_impurity_by_cluster.append(gini_impurity_index)

        gini_impurities_by_exp.append(gini_impurity_by_cluster)

        # Save clusters data
        np.save("cluster_data/" + "_".join(["contributors_by_cluster", str(embedding), str(clustering), str(KofFilename),str(exp), str(mutant)])
                + ".npy", contributors_by_cluster)
        np.save("cluster_data/" + "_".join(["contributors_concentration_by_cluster", str(embedding), str(clustering), str(KofFilename),str(exp), str(mutant)])
                + ".npy", contributors_concentration_by_cluster)
        np.save("cluster_data/" + "_".join(["contributors_share_by_cluster", str(embedding), str(clustering), str(KofFilename),str(exp), str(mutant)])
                + ".npy", contributors_share_by_cluster)

        np.save("cluster_data/" + "_".join(["inputs_share_by_cluster", str(embedding), str(clustering), str(KofFilename),str(exp), str(mutant)])
                + ".npy", inputs_share_by_cluster)

        # We want to find the "essential" aggregation of clusters such that the aggregated elements are able to
        # kill the mutant. To that end, we implement the following steps:
        #   1   -   We sort the clusters (keeping track of the indices) by their contributors concentration in descending order.
        sorted_contributors_concentrations_indices = np.argsort(contributors_concentration_by_cluster)[::-1]
        sorted_contributors_concentrations = np.asarray(contributors_concentration_by_cluster)[sorted_contributors_concentrations_indices.astype(int)]

        #   2   -   We start from the first cluster and progressively aggregate subsequent clusters until the mutant is killed.
        aggregated_check = False
        last_aggregated_cluster = 0
        clusters_aggregation = [sorted_contributors_concentrations_indices[0]]
        aggregation_size = -1

        aggregated_clusters_by_exp = []
        is_in_aggregation = None

        while not aggregated_check and len(clusters_aggregation) < K:

            # Get indices of elements in the aggregation
            is_in_aggregation = np.in1d(y_cluster_labels_test, clusters_aggregation)
            # Get the number of elements in the aggregation
            aggregation_size = np.sum(is_in_aggregation)

            # Save accuracy by run for cluster k
            original_accuracies_this_aggregation = []
            mutant_accuracies_this_aggregation = []

            for run in range(0, runs):
                original_accuracy = np.sum(correct_predictions_original[run][is_in_aggregation])/aggregation_size
                original_accuracies_this_aggregation.append(original_accuracy)

                mutant_accuracy = np.sum(correct_predictions_mutated[run][is_in_aggregation])/aggregation_size
                mutant_accuracies_this_aggregation.append(mutant_accuracy)

            # Check if the aggregation does kill a mutant in a statistically significant way
            check = is_diff_sts(np.asarray(original_accuracies_this_aggregation), np.asarray(mutant_accuracies_this_aggregation))[0]

            # If the mutant is not killed, expand the aggregation by adding clusters in it
            if not check:

                if last_aggregated_cluster < K:
                    # Add the next cluster by contributors concentration
                    last_aggregated_cluster = last_aggregated_cluster + 1
                    clusters_aggregation.append(sorted_contributors_concentrations_indices[last_aggregated_cluster])

            # Update check
            aggregated_check = check

        # Save aggregation indices
        aggregated_clusters_by_exp.append(is_in_aggregation)

        # Compute stats on aggregation
        # Percentage of clusters in aggregation over total clusters
        clusters_in_aggregation = len(clusters_aggregation)/K
        clusters_in_aggregation_by_exp.append(clusters_in_aggregation)

        # Raw number of inputs in aggregation
        elements_in_aggregation_raw = np.sum(np.in1d(y_cluster_labels_test, clusters_aggregation))
        raw_inputs_in_aggregation_by_exp.append(elements_in_aggregation_raw)

        # Percentage of inputs in aggregation over total inputs
        elements_in_aggregation = elements_in_aggregation_raw/len(y_test)
        inputs_in_aggregation_by_exp.append(elements_in_aggregation)

        # Number of raw contributors in aggregation
        raw_contributors_in_aggregation = np.sum(np.in1d(y_cluster_labels_test[is_a_contributor], clusters_aggregation))
        contributors_in_aggregation_by_exp.append(raw_contributors_in_aggregation)

        # Percentage of contributors in aggregation
        contributors_concentration_in_aggregation_by_exp.append(raw_contributors_in_aggregation/aggregation_size)

        # Share of contributors in aggregation
        contributors_share_in_aggregation_by_exp.append(raw_contributors_in_aggregation/total_contributors)

        # Gini Impurity index in aggregation
        gini_impurity_aggregation_index = 1 - (raw_contributors_in_aggregation / aggregation_size) ** 2 - (1 - raw_contributors_in_aggregation / aggregation_size) ** 2
        gini_impurities_in_aggregation_by_exp.append(gini_impurity_aggregation_index)

        # Determine whether the obtained aggregation is discriminative
        if aggregated_check:
            discriminative_aggregation_by_exp.append(1)
        else:
            discriminative_aggregation_by_exp.append(0)

        print("Experiment " + str(exp) + ", Clusters in aggregation: " + str(len(clusters_aggregation)/K)
                + " over a total of " + str(K) + ", with " + str(raw_contributors_in_aggregation)
                + " contributors, Mutant is killed " + str(aggregated_check))

        # Create list of clusters selected to belong to the aggregation
        selected_clusters = [0] * K
        for this_cluster in clusters_aggregation:
            selected_clusters[this_cluster] = 1

        # Save selected clusters indices
        np.save("selected_clusters/" + "_".join(["selected_clusters", str(embedding), str(clustering), str(KofFilename), str(exp), str(mutant)])
                + ".npy", selected_clusters)

    # Save cluster data
    np.save("cluster_data/" + "_".join(["aggregated_clusters_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", aggregated_clusters_by_exp)

    # Save cluster data
    np.save("cluster_data/" + "_".join(["gini_impurities_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", gini_impurities_by_exp)


    # Save experiment data
    np.save("experiments_data/" + "_".join(["clusters_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", clusters_in_aggregation_by_exp)

    np.save("experiments_data/" + "_".join(["inputs_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", inputs_in_aggregation_by_exp)
    
    np.save("experiments_data/" + "_".join(["raw_inputs_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", raw_inputs_in_aggregation_by_exp)

    np.save("experiments_data/" + "_".join(["contributors_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", np.asarray(contributors_in_aggregation_by_exp))

    np.save("experiments_data/" + "_".join(["contributors_concentration_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", np.asarray(contributors_concentration_in_aggregation_by_exp))

    np.save("experiments_data/" + "_".join(["contributors_share_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", contributors_share_in_aggregation_by_exp)

    np.save("experiments_data/" + "_".join(["gini_impurities_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", gini_impurities_in_aggregation_by_exp)

    np.save("experiments_data/" + "_".join(["discriminative_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)])
            + ".npy", discriminative_aggregation_by_exp)

    print("Mutant " + str(mutant)
            + ", Embedding " + str(embedding)
            + ", Clustering " + str(clustering)
            + ", K " + str(statistics.mean(Klist))
            + ", Contributors " + str(statistics.mean(contributors_concentration_in_aggregation_by_exp))
            + ", Discriminative " + str(statistics.mean(discriminative_aggregation_by_exp)))

    file_out.write(mutant + "," + embedding + "," + clustering + "," + str(KofFilename) + ","
                    + str(statistics.mean(Klist)) + ","
                    + str(statistics.mean(inputs_in_aggregation_by_exp)) + ","
                    + str(statistics.mean(contributors_concentration_in_aggregation_by_exp)) + ","
                    + str(statistics.mean(gini_impurities_in_aggregation_by_exp)) + ","
                    + str(statistics.mean(contributors_share_in_aggregation_by_exp)) + ","
                    + str(statistics.mean(discriminative_aggregation_by_exp))
                    + "\n")
    file_out.flush()

file_out.close()