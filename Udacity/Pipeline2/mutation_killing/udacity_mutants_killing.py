from __future__ import print_function
import keras, sys, gc
import os

import numpy as np
import pandas as pd
import warnings
import json
import tensorflow
from tensorflow.keras.optimizers import Adam
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw
from scipy.stats import ranksums


"""#Download data"""


def get_test_data():

  dataset_folder = os.path.join('../../Pipeline1')

  x_test_ = np.load(os.path.join(dataset_folder, 'dataset/udacity_test_x.npy'))
  y_test_ = np.load(os.path.join(dataset_folder, 'dataset/udacity_test_y.npy'))

  print("Test set: " + str(len(x_test_)) + " elements")

  return x_test_, y_test_

# Get test data
(x_test, y_test) =  get_test_data()


"""Evaluate model predictions"""


def evaluate_models_pred(original_model_preds, mutated_model_preds, run):

    # Get accuracies from models
    original_correct_predictions = 0
    mutant_correct_predictions = 0
    for instance in range(0, len(y_test)):
        if np.abs(original_model_preds[run][instance] - y_test[instance]) < 0.3:
            original_correct_predictions = original_correct_predictions + 1
        if np.abs(mutated_model_preds[run][instance] - y_test[instance]) < 0.3:
            mutant_correct_predictions = mutant_correct_predictions + 1

    # Store prediction scores by model
    score_original = original_correct_predictions / len(y_test)
    score_mutated = mutant_correct_predictions / len(y_test)

    # Store predicted values models
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

    mispredictions = []

    # A contributor is defined as an instance of the test set
    # which is predicted correctly by the original model, but
    # then is predicted incorrectly by the mutated model.
    for i in range(0, tot):
        if pred_labels_mutated[i] < pred_labels_original[i]:
            count = count + 1
            mispredictions.append(1)
        else:
            mispredictions.append(0)

    return count, tot, mispredictions


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
        loaded_model = tensorflow.keras.models.load_model(model + str(run) + ".h5")
        loaded_model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-4))

        # Get predicted labels from models
        pred_model = loaded_model.predict(x_test, verbose=0)
        pred_labels = np.asarray(np.concatenate(pred_model).ravel().tolist())
        #print(pred_labels)

        predicted_model.append(pred_labels)

    model_predictions = predicted_model
    model_name = model.split('/')
    np.save("../predictions/" + str(model_name[2]) + "predicted.npy", model_predictions)
    print(str(model_name[2] + " predictions saved"))


# Mutants
mutants_udacity = open('../udacity_mutants.json')
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

runs = 20  # Number of runs

# Ignore warning messages in output
warnings.filterwarnings("ignore")

# Infer predictions from the original models and mutants
if not os.path.exists("../predictions/" + "udacity_original_predicted.npy"):
    make_predictions("../original/udacity_original_")
for model_ in mutants:
    if not os.path.exists("../predictions/" + model_ + "predicted.npy"):
        make_predictions("../mutants/" + model_)

# Save mutation analysis results
file_out = open("udacity_mutant_killing.csv", "w")
file_out.write("Mutant,Total,Killed_WLX,Killed_GLM\n")
file_out.flush()

for mutant in mutants:

    # Create data structures to save results
    original_accuracies = []
    mutated_accuracies = []

    original_predictions = []
    mutated_predictions = []

    predictions_original_ = np.load("../predictions/" + "udacity_original_predicted" + ".npy")
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
        #   - 0 if instance i is mispredicted
        for instance in range(0, len(y_test)):
            if np.abs(original_predictions[run][instance] - y_test[instance]) < 0.3:
                predictions_at_this_run_original.append(1)
            else:
                predictions_at_this_run_original.append(0)
            if np.abs(mutated_predictions[run][instance] - y_test[instance]) < 0.3:
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
    total_mispredictions = 0

    # Estimate number of potential contributors
    num_cont, tot, mispredictions = estimate_contributors(prediction_stats_mean_original,
                                          prediction_stats_mean_mutated)

    # Count total mispredictions
    total_mispredictions = total_mispredictions + num_cont

    mispredictions = np.asarray(mispredictions)

    # Check if cluster does kill a mutant in a statistically significant way
    checks = []

    for stat_test in ["WLX", "GLM"]:
        STAT_TEST = stat_test
        check = is_diff_sts(np.asarray(original_accuracies), np.asarray(mutated_accuracies))[0]

        # Filter out false positives
        if total_mispredictions == 0:
            check = False

        checks.append(check)

    print("Mutant " + str(mutant) + ", Killed WLX: " + str(checks[0]) + ", Killed GLM: " + str(checks[1]))
    file_out.write(mutant + "," + str(total_mispredictions) + "," + str(checks[0])  + "," + str(checks[1]) + "\n")
    file_out.flush()

file_out.close()