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
inc_unseen = 0
if len(sys.argv) > 2:
    if sys.argv[2] == 'inc_unseen':
        inc_unseen = 1
    else:
        inc_unseen = 0


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

sum_recall_1_not_seen = 0
sum_recall_2_not_seen = 0
sum_recall_3_not_seen = 0
sum_precision_1_not_seen = 0
sum_precision_2_not_seen = 0
sum_precision_3_not_seen = 0
num_test_data_not_seen = 0
total_labels_not_seen = 0

sum_recall_1_seen = 0
sum_recall_2_seen = 0
sum_recall_3_seen = 0
sum_precision_1_seen = 0
sum_precision_2_seen = 0
sum_precision_3_seen = 0
num_test_data_seen = 0
total_labels_seen = 0

prediction_not_seen = {}
prediction_seen = {}
prediction_not_seen_correct = {}
prediction_seen_correct = {}
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

    #if there is atleast one not seen label in actual label, include current in the not seen (inc_unseen = 1)
    #if all actual label is not seen, include current in the not seen (inc_unseen = 0)
    not_seen_check = 0
    for jj in labels:
        if jj not in seen_labels:
            not_seen_check += 1
            #break
    correct_prediction_seen = 0
    correct_prediction_not_seen = 0
    if (inc_unseen == 1 and not_seen_check == 0) or (inc_unseen == 0 and not_seen_check != len(labels)):
        total_labels_seen += len(labels)
        num_test_data_seen += 1
        if predictions[ns[len(ns)-1]] in labels:
            correct_prediction_seen += 1
        sum_precision_1_seen += (correct_prediction_seen / min(1, len(labels)))
        # sum_precision_1 += (correct_prediction / 1)
        sum_recall_1_seen += (correct_prediction_seen / len(labels))

        # K = 2
        if predictions[ns[len(ns)-2]] in labels:
            correct_prediction_seen += 1
        sum_precision_2_seen += (correct_prediction_seen / (min(2, len(labels))))
        # sum_precision_2 += (correct_prediction / 2)
        sum_recall_2_seen += (correct_prediction_seen / len(labels))

        # K = 3
        if predictions[ns[len(ns)-3]] in labels:
            correct_prediction_seen += 1
        sum_precision_3_seen += (correct_prediction_seen / min(3, len(labels)))
        # sum_precision_3 += (correct_prediction / 3)
        sum_recall_3_seen += (correct_prediction_seen / len(labels))
    else:
        total_labels_not_seen += len(labels)
        num_test_data_not_seen += 1
        if predictions[ns[len(ns)-1]] in labels:
            correct_prediction_not_seen += 1
        sum_precision_1_not_seen += (correct_prediction_not_seen / min(1, len(labels)))
        # sum_precision_1 += (correct_prediction / 1)
        sum_recall_1_not_seen += (correct_prediction_not_seen / len(labels))

        # K = 2
        if predictions[ns[len(ns)-2]] in labels:
            correct_prediction_not_seen += 1
        sum_precision_2_not_seen += (correct_prediction_not_seen / (min(2, len(labels))))
        # sum_precision_2 += (correct_prediction / 2)
        sum_recall_2_not_seen += (correct_prediction_not_seen / len(labels))

        # K = 3
        if predictions[ns[len(ns)-3]] in labels:
            correct_prediction_not_seen += 1
        sum_precision_3_not_seen += (correct_prediction_not_seen / min(3, len(labels)))
        # sum_precision_3 += (correct_prediction / 3)
        sum_recall_3_not_seen += (correct_prediction_not_seen / len(labels))

    '''for kk in [0,1,2]:
        if predictions[kk] not in seen_labels:
            #NOT SEEN
        else:
            #SEEN '''
    
    print("Prediction labels")
    #print("SCORES")
    #print(temp_pred)
    #print("BEFORE")
    #print(predictions)
    #print("AFTER")
    #print(new_sort)
    print([predictions[ns[len(ns)-1]],predictions[ns[len(ns)-2]],predictions[ns[len(ns)-3]]])
    print(labels)
    print()

    # K = 1

    if predictions[ns[len(ns)-1]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-1]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-1]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-1]], 0) + 1
        else:
            prediction_seen_correct[predictions[ns[len(ns)-1]]] = prediction_seen_correct.get(predictions[ns[len(ns)-1]], 0) + 1
    if predictions[ns[len(ns)-1]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-1]]] = prediction_not_seen.get(predictions[ns[len(ns)-1]], 0) + 1
    else:
        prediction_seen[predictions[ns[len(ns)-1]]] = prediction_seen.get(predictions[ns[len(ns)-1]], 0) + 1
    sum_precision_1 += (correct_prediction / min(1, len(labels)))
    # sum_precision_1 += (correct_prediction / 1)
    sum_recall_1 += (correct_prediction / len(labels))

    # K = 2
    if predictions[ns[len(ns)-2]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-2]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-2]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-2]], 0) + 1
        else:
            prediction_seen_correct[predictions[ns[len(ns)-2]]] = prediction_seen_correct.get(predictions[ns[len(ns)-2]], 0) + 1
    if predictions[ns[len(ns)-2]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-2]]] = prediction_not_seen.get(predictions[ns[len(ns)-2]], 0) + 1
    else:
        prediction_seen[predictions[ns[len(ns)-2]]] = prediction_seen.get(predictions[ns[len(ns)-2]], 0) + 1
    sum_precision_2 += (correct_prediction / (min(2, len(labels))))
    # sum_precision_2 += (correct_prediction / 2)
    sum_recall_2 += (correct_prediction / len(labels))

    # K = 3
    if predictions[ns[len(ns)-3]] in labels:
        correct_prediction += 1
        if predictions[ns[len(ns)-3]] not in seen_labels:
            prediction_not_seen_correct[predictions[ns[len(ns)-3]]] = prediction_not_seen_correct.get(predictions[ns[len(ns)-3]], 0) + 1
        else:
            prediction_seen_correct[predictions[ns[len(ns)-3]]] = prediction_seen_correct.get(predictions[ns[len(ns)-3]], 0) + 1
    if predictions[ns[len(ns)-3]] not in seen_labels:
        prediction_not_seen[predictions[ns[len(ns)-3]]] = prediction_not_seen.get(predictions[ns[len(ns)-3]], 0) + 1
    else:
        prediction_seen[predictions[ns[len(ns)-3]]] = prediction_seen.get(predictions[ns[len(ns)-3]], 0) + 1
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
print("TEST DATA TOTAL: " + num_test_data.__str__())

print()
print()

print("SEEN")
if inc_unseen == 0:
    print("(At least 1 seen label in entry)")
else:
    print("(All labels have been seen)")
precision_1_seen= sum_precision_1_seen / num_test_data_seen
precision_2_seen = sum_precision_2_seen / num_test_data_seen
precision_3_seen = sum_precision_3_seen / num_test_data_seen
recall_1_seen = sum_recall_1_seen / num_test_data_seen
recall_2_seen = sum_recall_2_seen / num_test_data_seen
recall_3_seen = sum_recall_3_seen / num_test_data_seen
f1_1_seen = 2 * precision_1_seen * recall_1_seen / (precision_1_seen + recall_1_seen)
f1_2_seen = 2 * precision_2_seen * recall_2_seen / (precision_2_seen + recall_2_seen)
f1_3_seen = 2 * precision_3_seen * recall_3_seen / (precision_3_seen + recall_3_seen)

print("K = 1")
print("P@1 = " + precision_1_seen.__str__())
print("R@1 = " + recall_1_seen.__str__())
print("F@1 = " + f1_1_seen.__str__())

print("K = 2")
print("P@2 = " + precision_2_seen.__str__())
print("R@2 = " + recall_2_seen.__str__())
print("F@2 = " + f1_2_seen.__str__())

print("K = 3")
print("P@3 = " + precision_3_seen.__str__())
print("R@3 = " + recall_3_seen.__str__())
print("F@3 = " + f1_3_seen.__str__())
print("TOTAL LABELS: " + total_labels_seen.__str__())
print("TEST DATA TOTAL (SEEN): " + num_test_data_seen.__str__())

print()
print()

print("NOT SEEN")
if inc_unseen == 1:
    print("(At least 1 unseen label in entry)")
else:
    print("(All labels are unseen)")
precision_1_not_seen= sum_precision_1_not_seen / num_test_data_not_seen
precision_2_not_seen = sum_precision_2_not_seen / num_test_data_not_seen
precision_3_not_seen = sum_precision_3_not_seen / num_test_data_not_seen
recall_1_not_seen = sum_recall_1_not_seen / num_test_data_not_seen
recall_2_not_seen = sum_recall_2_not_seen / num_test_data_not_seen
recall_3_not_seen = sum_recall_3_not_seen / num_test_data_not_seen
f1_1_not_seen = 2 * precision_1_not_seen * recall_1_not_seen / (precision_1_not_seen + recall_1_not_seen)
f1_2_not_seen = 2 * precision_2_not_seen * recall_2_not_seen / (precision_2_not_seen + recall_2_not_seen)
f1_3_not_seen = 2 * precision_3_not_seen * recall_3_not_seen / (precision_3_not_seen + recall_3_not_seen)

print("K = 1")
print("P@1 = " + precision_1_not_seen.__str__())
print("R@1 = " + recall_1_not_seen.__str__())
print("F@1 = " + f1_1_not_seen.__str__())

print("K = 2")
print("P@2 = " + precision_2_not_seen.__str__())
print("R@2 = " + recall_2_not_seen.__str__())
print("F@2 = " + f1_2_not_seen.__str__())

print("K = 3")
print("P@3 = " + precision_3_not_seen.__str__())
print("R@3 = " + recall_3_not_seen.__str__())
print("F@3 = " + f1_3_not_seen.__str__())
print("TOTAL LABELS: " + total_labels_not_seen.__str__())
print("TEST DATA TOTAL (NOT SEEN): " + num_test_data_not_seen.__str__())

print()
print()

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


# print(prediction_seen)
print("How many seen labels:")
print(len(prediction_seen))
print("How many seen labels usage")
sum = 0
print(prediction_seen)
for key, items in prediction_seen.items():
    sum += items
print(sum)
print()
print()
# print(prediction_seen_correct)
print("How many seen labels correct:")
print(len(prediction_seen_correct))
print("How many seen labels used correctly")
sum = 0
for key, items in prediction_seen_correct.items():
    sum += items
print(sum)
