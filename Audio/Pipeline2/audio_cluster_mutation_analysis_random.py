from __future__ import print_function
import keras, sys, gc
import os

import numpy as np
import pandas as pd
import statistics
import warnings
import pickle
import json
from keras import backend as K
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw
from scipy.stats import wilcoxon, ranksums
from random import sample

"""Download data"""


def get_test_data():

    x_test = np.load('../Pipeline1/dataset/audio_test_x.npy')
    y_test = np.load('../Pipeline1/dataset/audio_test_y.npy')

    return x_test, y_test

(x_test, y_test) = get_test_data()

class_names = ['Julia_Gillard', 'Nelson_Mandela', 'Benjamin_Netanyahu', 'Margaret_Thatcher', 'Jens_Stoltenberg']
print("Our class names: {}".format(class_names, ))


"""Prepare Input"""


# Load number of inputs in aggregation
def get_raw_number_of_inputs_in_aggregation(embedding, clustering, KofFilename, mutant):
    inputs_in_aggregation = np.load("experiments_data/" + "_".join(["raw_inputs_in_aggregation_by_exp", str(embedding), str(clustering), str(KofFilename), str(mutant)]) + ".npy")
    return inputs_in_aggregation


"""Evaluate model predictions"""


def evaluate_model_global_predictions(original_model_preds, mutated_model_preds):

    # Create matrix instances x runs:
    #   - 1 if instance i is predicted correctly
    #   - 0 if instance i is misclassified
    correct_predictions_original = []
    correct_predictions_mutated = []

    for run in range(0, runs):
        correct_at_this_run_original = (np.asarray(original_model_preds[run]) == np.asarray(y_test))
        correct_at_this_run_mutated = (np.asarray(mutated_model_preds[run]) == np.asarray(y_test))

        correct_predictions_original.append(correct_at_this_run_original)
        correct_predictions_mutated.append(correct_at_this_run_mutated)

    # Compute the mean in correct predictions by instance for this cluster
    correct_predictions_mean_original = np.mean(correct_predictions_original, axis=0)
    correct_predictions_mean_mutated = np.mean(correct_predictions_mutated, axis=0)

    # Get a vector of binary labels:
    #   - 1 if instance is a contributor
    #   - 0 if instance is not a contributor
    is_a_contributor = correct_predictions_mean_original > correct_predictions_mean_mutated

    return correct_predictions_original, correct_predictions_mutated, is_a_contributor


def make_predictions(model):

    predicted_model = []

    for run in range(0, runs):
        keras.backend.clear_session()
        loaded_model = keras.models.load_model(model + str(run) + ".h5")

        # Get predicted labels from models
        pred_model = loaded_model.predict(x_test, verbose=0)
        pred_labels = np.argmax(pred_model, axis=1)

        # experiments_model.append(score_model)
        predicted_model.append(pred_labels)

    model_predictions = predicted_model
    model_name = model.split('/')
    np.save("predictions/" + str(model_name[1]) + "predicted.npy", model_predictions)
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
    # Unpaired
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
mutants_audio = open('audio_mutants.json')
mutants_audio_loaded = json.load(mutants_audio)
mutants = []
for mutant in mutants_audio_loaded:
    for mut_score in mutants_audio_loaded[mutant]:
        if mutant in ["audio_change_learning_rate", "audio_change_epochs"]:
            name_mutant = str(mutant) + str("_mutated0_MP_False_") + str(mut_score) + "_"
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
file_out = open("results/pipeline2_results_audio_random_sample.csv", "w")
file_out.write(
    "Mutant,Embedding,Clustering,KOfFilename,Killed,ContributorsDensity,MutantKilledOverExp\n")
file_out.flush()

# Load results obtained from first pipeline
df = pd.read_csv('../Pipeline1/results/pipeline1_results_pairwise_audio.csv', sep=',')

"""Configuration selection criterion"""
selected_configuration = df.loc[df["PairWiseMinAccuracy"].idxmax()]

# Infer predictions from the original models and mutants
if not os.path.exists("predictions/" + "audio_original_predicted.npy"):
    make_predictions("original/audio_original_")
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

    # Percentage of contributors in aggregation by experiment
    contributors_concentration_in_aggregation_by_exp = []

    # Is aggregation discriminative at each experiment?
    discriminative_aggregation_by_exp = []

    # Load model predictions
    predictions_original_ = np.load("predictions/" + "audio_original_predicted" + ".npy")
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

    # Get number of inputs in aggregation
    tot_raw_inputs_in_aggregation = get_raw_number_of_inputs_in_aggregation(embedding, clustering, KofFilename, mutant)

    for exp in range(0, experiments):

        # Get number of inputs in aggregation for this experiment
        this_raw_inputs_in_aggregation = tot_raw_inputs_in_aggregation[exp]
        indices_to_select = range(0, len(y_test))

        # Get indices of elements in the random sample
        random_sample_selection = sample(indices_to_select, this_raw_inputs_in_aggregation)

        sampled_x_test = x_test[random_sample_selection]
        sampled_y_test = y_test[random_sample_selection]

        # Save accuracy by run
        original_accuracies_this_aggregation = []
        mutant_accuracies_this_aggregation = []

        for run in range(0, runs):
            original_accuracy = np.sum(correct_predictions_original[run][random_sample_selection])/this_raw_inputs_in_aggregation
            original_accuracies_this_aggregation.append(original_accuracy)

            mutant_accuracy = np.sum(correct_predictions_mutated[run][random_sample_selection])/this_raw_inputs_in_aggregation
            mutant_accuracies_this_aggregation.append(mutant_accuracy)

        # Number of raw contributors in aggregation
        raw_contributors_in_random_aggregation = np.sum(is_a_contributor[random_sample_selection])

        # Percentage of contributors in aggregation
        contributors_concentration_in_aggregation_by_exp.append(
            raw_contributors_in_random_aggregation / this_raw_inputs_in_aggregation)

        # Check if the aggregation does kill a mutant in a statistically significant way
        check = is_diff_sts(np.asarray(original_accuracies_this_aggregation), np.asarray(mutant_accuracies_this_aggregation))[0]

        # Determine whether the obtained aggregation is discriminative
        if check:
            discriminative_aggregation_by_exp.append(1)
        else:
            discriminative_aggregation_by_exp.append(0)

    print("Mutant " + str(mutant)
          + ", Embedding " + str(embedding)
          + ", Clustering " + str(clustering)
          + ", ContributorsDensity " + str(statistics.mean(contributors_concentration_in_aggregation_by_exp))
          + ", Discriminative " + str(statistics.mean(discriminative_aggregation_by_exp)))

    file_out.write(mutant + "," + embedding + "," + clustering + "," + str(KofFilename) + ","
                   + str(statistics.mean(discriminative_aggregation_by_exp) > 0) + ","
                   + str(statistics.mean(contributors_concentration_in_aggregation_by_exp)) + ","
                   + str(statistics.mean(discriminative_aggregation_by_exp))
                   + "\n")
    file_out.flush()

file_out.close()