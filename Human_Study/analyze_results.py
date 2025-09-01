import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import itertools

def extract_response_data(questionnaire_responses, num_tasks, num_questions):

    questionnaire_classifications = []
    questionnaire_difficulty = []

    for task in range(0, num_tasks):

        if task + 1 == 17 or task + 1 == 18:
            # Skip attention check
            continue

        if task % 2 == 0:

            column_task_name = '.'.join(("Q2", str(task + 1)))
            task_answers = []

            for question in range(0, num_questions):
                column_name = '_'.join((column_task_name, str(question + 1)))
                task_answers.append(questionnaire_responses[column_name].values.tolist())

            questionnaire_classifications.append(task_answers)

        else:
            column_difficulty_name = '.'.join(("Q2", str(task + 1)))
            questionnaire_difficulty.append(questionnaire_responses[column_difficulty_name].values.tolist())

    questionnaire_classifications = np.transpose(np.array(questionnaire_classifications) - 1)
    questionnaire_difficulty = np.transpose(1 - np.array(questionnaire_difficulty) / 5)

    return questionnaire_classifications, questionnaire_difficulty

def merge_questionnaires_data(lst1, lst2):
    return [sub[item] for item in range(len(lst2))
                      for sub in [lst1, lst2]]

def get_classification_criteria(questionnaire_responses_a, questionnaire_responses_b):

    classification_criteria_a = questionnaire_responses_a["Q3.1"].values.tolist()
    classification_criteria_b = questionnaire_responses_b["Q3.1"].values.tolist()

    classification_criteria_ = classification_criteria_a + classification_criteria_b

    classification_criteria = [x.split(",") for x in classification_criteria_]
    classification_criteria = list(itertools.chain(*classification_criteria))
    classification_criteria = [int(x) for x in classification_criteria]

    classification_criteria_stats = np.unique(classification_criteria, return_counts=True)[1]

    return classification_criteria_stats


def get_5_5_bias(questionnaire_responses_a, questionnaire_responses_b):

    split_bias_a = questionnaire_responses_a["Q3.2"].values.tolist()
    split_bias_b = questionnaire_responses_b["Q3.2"].values.tolist()

    split_bias = split_bias_a + split_bias_b

    yes_no = np.unique(split_bias, return_counts=True)[1]

    return yes_no

def compute_responses_accuracy(responses_classifications, ground_truth):

    accuracies = []

    for user in range(0, len(responses_classifications)):

        user_accuracy = np.mean(responses_classifications[user] == np.transpose(ground_truth), axis = 0)

        accuracies.append(user_accuracy)

    accuracies = np.array(accuracies)

    return accuracies

def analyze_results(data):

    loaded_responses_a = pd.read_csv(os.path.join("data", "_".join((data, "questionnaire_a.csv"))))
    loaded_responses_b = pd.read_csv(os.path.join("data", "_".join((data, "questionnaire_b.csv"))))

    human_classification_q_a, human_difficulty_q_a = extract_response_data(loaded_responses_a, 32, 10)
    human_classification_q_b, human_difficulty_q_b = extract_response_data(loaded_responses_b, 32, 10)

    ground_truth_q_a = np.load(os.path.join("data", "_".join((data, "a_ground_truth.npy"))))
    ground_truth_q_a = ground_truth_q_a[:15]
    ground_truth_q_b = np.load(os.path.join("data", "_".join((data, "b_ground_truth.npy"))))
    ground_truth_q_b = ground_truth_q_b[:15]

    accuracies_responses_q_a = compute_responses_accuracy(human_classification_q_a, ground_truth_q_a)
    accuracies_responses_q_b = compute_responses_accuracy(human_classification_q_b, ground_truth_q_b)

    print(len(accuracies_responses_q_a))
    print(len(accuracies_responses_q_b))

    total_participants = len(accuracies_responses_q_a) + len(accuracies_responses_q_b)

    yes_no_num = get_5_5_bias(loaded_responses_a, loaded_responses_b)

    print("Yes: ", yes_no_num[0]/total_participants)
    print("No: ", yes_no_num[1]/total_participants)

    print("-------------------------")

    classification_criteria_stats = get_classification_criteria(loaded_responses_a, loaded_responses_b)

    if data == "mnist":
        print("Digit Orientation: ", classification_criteria_stats[0]/total_participants)
        print("Digit Boldness: ", classification_criteria_stats[1]/total_participants)
        print("Digit Value: ", classification_criteria_stats[2]/total_participants)

    if data == "cifar10":
        print("Image Color: ", classification_criteria_stats[0]/total_participants)
        print("Image Subject: ", classification_criteria_stats[1]/total_participants)


    print("-------------------------")

    human_mean_accuracy_q_a = np.mean(accuracies_responses_q_a, axis=0)
    human_mean_accuracy_q_b = np.mean(accuracies_responses_q_b, axis=0)

    human_mean_difficulty_q_a = np.mean(human_difficulty_q_a, axis=0)
    human_mean_difficulty_q_b = np.mean(human_difficulty_q_b, axis=0)

    # Merge questionnaires data
    human_mean_accuracy = merge_questionnaires_data(human_mean_accuracy_q_a, human_mean_accuracy_q_b)
    human_mean_difficulty = merge_questionnaires_data(human_mean_difficulty_q_a, human_mean_difficulty_q_b)

    # Load DNN performance data
    dnn_performance_data = pd.read_csv(os.path.join("data", "_".join((data, "cluster_pairwise_dnn_comparison.csv"))))

    # Extract the specified columns as arrays
    pairwise_accuracy = dnn_performance_data['PairWiseAccuracy'].values
    accuracy_score = dnn_performance_data['AccuracyScore'].values
    accuracy_score10 = dnn_performance_data['AccuracyScore10'].values

    # Compute Pearson correlation
    res_pwacc = pearsonr(human_mean_accuracy, pairwise_accuracy)
    print(f"Pearson correlation between human_accuracy and pairwise_accuracy: {res_pwacc[0]}")
    print(f"P-value between human_accuracy and pairwise_accuracy: {res_pwacc[1]}")

    res_acc = pearsonr(human_mean_accuracy, accuracy_score)
    print(f"Pearson correlation between human_accuracy and accuracy_score: {res_acc[0]}")
    print(f"P-value between human_accuracy and accuracy_score: {res_acc[1]}")

    res_acc10 = pearsonr(human_mean_accuracy, accuracy_score10)
    print(f"Pearson correlation between human_accuracy and accuracy_score10: {res_acc10[0]}")
    print(f"P-value between human_accuracy and accuracy_score10: {res_acc10[1]}")

    print("Overall mean accuracy: ", np.mean(human_mean_accuracy))
    print("Overall mean accuracy: ", np.std(human_mean_accuracy))
    print("Overall mean difficulty: ", np.mean(human_mean_difficulty))
    print("Overall mean difficulty: ", np.std(human_mean_difficulty))

    print("-------------------------")

    # Compute Pearson correlation with difficulty
    res_pwacc = pearsonr(human_mean_difficulty, pairwise_accuracy)
    print(f"Pearson correlation between human_difficulty and pairwise_accuracy: {res_pwacc[0]}")
    print(f"P-value between human_difficulty and pairwise_accuracy: {res_pwacc[1]}")

    res_acc = pearsonr(human_mean_difficulty, accuracy_score)
    print(f"Pearson correlation between human_difficulty and accuracy_score: {res_acc[0]}")
    print(f"P-value between human_difficulty and accuracy_score: {res_acc[1]}")

    res_acc10 = pearsonr(human_mean_difficulty, accuracy_score10)
    print(f"Pearson correlation between human_difficulty and accuracy_score10: {res_acc10[0]}")
    print(f"P-value between human_difficulty and accuracy_score10: {res_acc10[1]}")

    print("-------------------------")

    res_hacc = pearsonr(human_mean_accuracy, human_mean_difficulty)
    print(f"Pearson correlation between human_accuracy and human_difficulty: {res_hacc[0]}")
    print(f"P-value between human_difficulty and accuracy_score10: {res_hacc[1]}")


if __name__ == "__main__":
  ds = sys.argv[1]
  if ds != "mnist" and ds != "cifar10":
      print("Please write 'mnist' or 'cifar10'.")
  else:
    analyze_results(ds)

