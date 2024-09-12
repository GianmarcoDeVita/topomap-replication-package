from __future__ import print_function
import keras, sys, gc
import os

import numpy as np
import pandas as pd
import statistics
import warnings
import pickle
import json
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw
from scipy.stats import wilcoxon, ranksums

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
y_test = np.asarray(y_test).reshape(-1)

"""Evaluate model predictions"""


def evaluate_models_pred(original_model_preds, mutated_model_preds, run):

    # Get accuracies from models
    original_correct_predictions = 0
    mutant_correct_predictions = 0
    for instance in range(0, len(y_test)):
        if original_model_preds[run][instance] == y_test[instance]:
            original_correct_predictions = original_correct_predictions + 1
        if mutated_model_preds[run][instance] == y_test[instance]:
            mutant_correct_predictions = mutant_correct_predictions + 1

    # Store prediction scores by model
    score_original = original_correct_predictions / len(y_test)
    score_mutated = mutant_correct_predictions / len(y_test)

    # Store predicted labels models
    pred_labels_original = original_model_preds[run]
    pred_labels_mutated = mutated_model_preds[run]

    return score_original, score_mutated, pred_labels_original, pred_labels_mutated


"""Analysis"""


def estimate_contributors(pred_labels_original, pred_labels_mutated):

    # Check if sizes are equal
    tot = len(pred_labels_original)
    if tot != len(pred_labels_mutated):
        return -1

    count = 0

    misclassifications = []

    # A contributor is defined as an instance of the test set
    # which is predicted correctly by the original model, but
    # then is predicted incorrectly by the mutated model.
    for i in range(0, tot):
        if pred_labels_mutated[i] < pred_labels_original[i]:
            count = count + 1
            misclassifications.append(1)
        else:
            misclassifications.append(0)

    return count, tot, misclassifications


"""Statistical test"""

STAT_TEST = ""

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

    orig_accuracy_list = list(orig_accuracy_list)
    accuracy_list = list(accuracy_list)

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


def make_predictions(model):

    predicted_model = []

    for run in range(0, runs):
        keras.backend.clear_session()
        loaded_model = keras.models.load_model(model + str(run) + ".h5")

        # Get predicted labels from models
        pred_model = loaded_model.predict(x_test, verbose=0)
        pred_labels = np.argmax(pred_model, axis=1)

        predicted_model.append(pred_labels)

    model_predictions = predicted_model
    model_name = model.split('/')
    np.save("../predictions/" + str(model_name[1]) + ".npy", model_predictions)
    print(str(model_name[1] + " predictions saved"))


# Mutants
mutants_reuters = open('../reuters_mutants.json')
mutants_reuters_loaded = json.load(mutants_reuters)
mutants = []
for mutant in mutants_reuters_loaded:
    for mut_score in mutants_reuters_loaded[mutant]:
        if mutant in ["reuters_change_learning_rate"]:
            name_mutant = str(mutant) + str("_mutated0_MP_False_") + str(mut_score) + "_"
        else:
            name_mutant = str(mutant) + str("_mutated0_MP_") + str(mut_score) + "_"
        if mut_score == "":
            name_mutant = str(mutant) + str("_mutated0_MP_")
        mutants.append(name_mutant)

runs = 20  # Number of runs

# Ignore warning messages in output
warnings.filterwarnings("ignore")

# Infer predictions from the original models and mutants
if not os.path.exists("../predictions/" + "reuters_original_predicted.npy"):
    make_predictions("../original/reuters_original_")
for model_ in mutants:
    if not os.path.exists("../predictions/" + model_ + "predicted.npy"):
        make_predictions("../mutants/" + model_)

# Save mutation analysis results
file_out = open("reuters_misclassification_distribution_and_mutant_killing.csv", "w")
file_out.write("Mutant," + ",".join(class_names) +
               "Total,Killed_WLX,Killed_GLM\n")
file_out.flush()

for mutant in mutants:

    # Create data structures to save results
    original_accuracies = []
    mutated_accuracies = []

    original_predictions = []
    mutated_predictions = []

    predictions_original_ = np.load("../predictions/" + "reuters_original_predicted" + ".npy")
    predictions_mutated_ = np.load("../predictions/" + mutant + "predicted.npy")

    for run in range(0, runs):
        experiments_orig, experiments_mutated, predicted_original, predicted_mutated = evaluate_models_pred(predictions_original_, predictions_mutated_, run)

        original_accuracies.append(experiments_orig)
        mutated_accuracies.append(experiments_mutated)

        original_predictions.append(predicted_original)
        mutated_predictions.append(predicted_mutated)

    predictions_on_original = []
    predictions_on_mutant = []

    for run in range(0, runs):

        predictions_at_this_run_original = []
        predictions_at_this_run_mutant = []

        # Create matrix instances x runs:
        #   - 1 if instance i is predicted correctly
        #   - 0 if instance i is misclassified
        for instance in range(0, len(y_test)):
            if original_predictions[run][instance] == y_test[instance]:
                predictions_at_this_run_original.append(1)
            else:
                predictions_at_this_run_original.append(0)
            if mutated_predictions[run][instance] == y_test[instance]:
                predictions_at_this_run_mutant.append(1)
            else:
                predictions_at_this_run_mutant.append(0)

        # Add predictions of r-th run
        predictions_on_original.append(np.asarray(predictions_at_this_run_original))
        predictions_on_mutant.append(np.asarray(predictions_at_this_run_mutant))

    predictions_on_original = np.asarray(predictions_on_original)
    predictions_on_mutant = np.asarray(predictions_on_mutant)

    # Compute the mean in correct predictions by instance for this cluster
    prediction_stats_mean_original = np.mean(predictions_on_original, axis=0)
    prediction_stats_mean_mutated = np.mean(predictions_on_mutant, axis=0)

    # Contributors data
    total_misclassifications = 0

    # Estimate number of potential contributors
    num_cont, tot, misclassifications = estimate_contributors(prediction_stats_mean_original,
                                          prediction_stats_mean_mutated)

    # Count total misclassifications
    total_misclassifications = total_misclassifications + num_cont

    misclassifications = np.asarray(misclassifications)

    misclassifications_by_class = []
    for label in range(0, len(class_names)):
        indices_for_this_label = np.where(y_test == label)[0]
        misclassifications_for_this_label = np.sum(misclassifications[indices_for_this_label.astype(int)])
        misclassifications_by_class.append(misclassifications_for_this_label)

    # Check if cluster does kill a mutant in a statistically significant way
    checks = []

    for stat_test in ["WLX", "GLM"]:
        STAT_TEST = stat_test
        check = is_diff_sts(np.asarray(original_accuracies), np.asarray(mutated_accuracies))[0]

        # Filter out false positives
        if total_misclassifications == 0:
            check = False

        checks.append(check)

    print("Mutant " + str(mutant) + ", Killed WLX: " + str(checks[0]) + ", Killed GLM: " + str(checks[1]))
    file_out.write(mutant + ","
                   + ",".join(map(str, misclassifications_by_class)) + ","
                   + str(total_misclassifications) + "," + str(checks[0]) + "," + str(checks[1]) + "\n")
    file_out.flush()

file_out.close()