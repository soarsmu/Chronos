import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from tabulate import tabulate
from io import StringIO
from tqdm import tqdm
from utils import *


def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()

    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

dataset = sys.argv[1]
RES_DIR = f'Results/{dataset}'
DATA_DIR = f'GZXML-Datasets/{dataset}'

# print(_c("Loading files", attr="yellow"))
print("Loading files")
trn_X_Y = read_sparse_mat('%s/trn_X_Y.txt'%DATA_DIR, use_xclib=False)
tst_X_Y = read_sparse_mat('%s/tst_X_Y.txt'%DATA_DIR, use_xclib=False)

score_mat = _filter(read_bin_spmat(f'{RES_DIR}/score_mat.bin').copy(), None)
# Shape should be:
# nrows = number of test data
# ncols = scores for possible labels
x = score_mat.toarray()

# getting the set of seen labels in training dataset
seen_labels = set()
train_label = []
with open(f'{DATA_DIR}/trn_X_Y.txt', "r", encoding="utf-8") as re:
    train_label = re.readlines()[1:]
for text in train_label:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        seen_labels.add(int(label_num))


# loop through the score matrix
text_labels = []
with open(f'{DATA_DIR}/tst_X_Y.txt', "r", encoding="utf-8") as re:
    text_labels = re.readlines()[1:]
actuals = []
for text in text_labels:
    list_labels = []
    split = text.split(" ")
    for label in split:
        label_num = label.split(":")[0]
        list_labels.append(int(label_num))
    actuals.append(list_labels)
sum_recall_1 = 0
sum_recall_2 = 0
sum_recall_3 = 0
sum_precision_1 = 0
sum_precision_2 = 0
sum_precision_3 = 0
num_test_data = 0
total_labels = 0

prediction_not_seen = {}
prediction_not_seen_correct = {}

for i, rows in enumerate(x):
    predictions = np.argpartition(rows, -3)[-3:]
    predictions = predictions[::-1]
    temp_pred = rows[predictions[:3]]
    new_sort = np.argsort(temp_pred)
    ns = list(new_sort)
    
    num_test_data += 1
    local_correct_prediction = 0
    labels = actuals[i]
    # get only the top k prediction

    total_labels += len(labels)
    correct_prediction = 0
    
    # print("Prediction labels")
    # #print("SCORES")
    # #print(temp_pred)
    # #print("BEFORE")
    # #print(predictions)
    # #print("AFTER")
    # #print(new_sort)
    # print([predictions[ns[len(ns)-1]],predictions[ns[len(ns)-2]],predictions[ns[len(ns)-3]]])
    # print(labels)
    # print()
    
    # K = 1
    if predictions[ns[len(ns)-1]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-1]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-1]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-1]], 0) + 1
    if predictions[ns[len(ns)-1]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-1]]] = prediction_not_seen.get(predictions[ns[len(ns)-1]], 0) + 1
    sum_precision_1 += (correct_prediction / min(1, len(labels)))
    # sum_precision_1 += (correct_prediction / 1)
    sum_recall_1 += (correct_prediction / len(labels))

    # K = 2
    if predictions[ns[len(ns)-2]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-2]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-2]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-2]], 0) + 1
    if predictions[ns[len(ns)-2]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-2]]] = prediction_not_seen.get(predictions[ns[len(ns)-2]], 0) + 1
    sum_precision_2 += (correct_prediction / (min(2, len(labels))))
    # sum_precision_2 += (correct_prediction / 2)
    sum_recall_2 += (correct_prediction / len(labels))

    # K = 3
    if predictions[ns[len(ns)-3]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-3]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-3]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-3]], 0) + 1
    if predictions[ns[len(ns)-3]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-3]]] = prediction_not_seen.get(predictions[ns[len(ns)-3]], 0) + 1
    sum_precision_3 += (correct_prediction / min(3, len(labels)))
    # sum_precision_3 += (correct_prediction / 3)
    sum_recall_3 += (correct_prediction / len(labels))

precision_1 = sum_precision_1 / num_test_data
precision_2 = sum_precision_2 / num_test_data
precision_3 = sum_precision_3 / num_test_data
recall_1 = sum_recall_1 / num_test_data
recall_2 = sum_recall_2 / num_test_data
recall_3 = sum_recall_3 / num_test_data
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)

print("K = 1")
print("P@1 = " + precision_1.__str__())
print("R@1 = " + recall_1.__str__())
print("F@1 = " + f1_1.__str__())

print("K = 2")
print("P@2 = " + precision_2.__str__())
print("R@2 = " + recall_2.__str__())
print("F@2 = " + f1_2.__str__())

print("K = 3")
print("P@3 = " + precision_3.__str__())
print("R@3 = " + recall_3.__str__())
print("F@3 = " + f1_3.__str__())
print("TOTAL LABELS: " + total_labels.__str__())

# print(prediction_not_seen)
print("How many unseen labels:")
print(len(prediction_not_seen))
print("How many unseen labels usage")
sum = 0
print(prediction_not_seen)
for key, items in prediction_not_seen.items():
    sum += items
print(sum)
print()
print()
# print(prediction_not_seen_correct)
print("How many unseen labels correct:")
print(len(prediction_not_seen_correct))
print("How many unseen labels used correctly")
sum = 0
for key, items in prediction_not_seen_correct.items():
    sum += items
print(sum)
