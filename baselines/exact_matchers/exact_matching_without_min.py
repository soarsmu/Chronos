import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle as pkl
from spiral import ronin

NON_LABEL_COLUMNS = ["cve_id", "cleaned", "matchers", "merged", "reference", "description_and_reference", "year"]
FEATURE_NAME = "merged"

def split_labels(test_dataset, labels):
    splitted_labels = {}
    for i in range(len(labels)):
            splitted_labels[i] = []
            sub_labels = labels[i].split(";")
            tmp = 0
            for sl in sub_labels:
                splitted_sub_labels = ronin.split(sl)
                for ssl in splitted_sub_labels:
                    splitted_labels[i].append(ssl)
    return splitted_labels 


                
def main():
    test_dataset = pd.read_csv("../../zero_shot_dataset/zero_shot_test_cleaned.csv")
    labels = [i for i in test_dataset.columns if i not in NON_LABEL_COLUMNS]
    description = test_dataset[FEATURE_NAME]
    pd_labels_test = test_dataset[labels].to_numpy()
    
    ground_truth = {}
    for idx in tqdm(range(pd_labels_test.shape[0])):
        ground_truth[idx] = list(np.where(pd_labels_test[idx] == 1)[0])

    sum_recall_1 = 0
    sum_recall_2 = 0
    sum_recall_3 = 0
    sum_precision_1 = 0
    sum_precision_2 = 0
    sum_precision_3 = 0
    num_test_data = 0
    total_labels = 0

    splitted_labels = split_labels(test_dataset, labels)
    for idx, desc in tqdm(enumerate(description)):
        num_test_data += 1
        splitted_desc = desc.split()
        freq = {}
        for w in splitted_desc:
            if w in freq:
                freq[w] += 1
            else:
                freq[w] = 1
        
        count = []
        for i in range(len(labels)):
            sub_labels = labels[i].split(";")
            tmp = 0
            for sl in sub_labels:
                 if sl in freq:
                     tmp += freq[sl]
            count.append(tmp)
        count = np.array(count)
        preds = np.argpartition(count, -3)[-3:]
        temp_pred = count[preds[:3]]
        new_sort = np.argsort(temp_pred)
        ns = list(new_sort)
        correct_prediction = 0
        if preds[ns[len(ns)-1]] in ground_truth[idx] and temp_pred[len(ns)-1] != 0:
            correct_prediction += 1
        sum_precision_1 += (correct_prediction / 1)
        sum_recall_1 += (correct_prediction / len(ground_truth[idx]))

        if preds[ns[len(ns)-2]] in ground_truth[idx] and temp_pred[len(ns)-2] != 0:
            correct_prediction += 1
        sum_precision_2 += (correct_prediction / 2)
        sum_recall_2 += (correct_prediction / len(ground_truth[idx]))

        if preds[ns[len(ns)-3]] in ground_truth[idx] and temp_pred[len(ns)-3] != 0:
            correct_prediction += 1
        sum_precision_3 += (correct_prediction / 3)
        sum_recall_3 += (correct_prediction / len(ground_truth[idx]))

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
    print(precision_1.__str__())
    print(recall_1.__str__())
    print(f1_1.__str__())

    print("K = 2")
    print(precision_2.__str__())
    print(recall_2.__str__())
    print(f1_2.__str__())

    print("K = 3")
    print(precision_3.__str__())
    print(recall_3.__str__())
    print(f1_3.__str__())






if __name__ == "__main__":
    # split_labels()
    main()
