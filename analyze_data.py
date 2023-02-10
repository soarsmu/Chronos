import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle as pkl
from helper import combine_description_and_reference_data
import matplotlib.pyplot as plt

BASE_PATH = "dataset/description_data/"
DESCRIPTION_DATA_PATH = f"{BASE_PATH}/dataset_merged_cleaned.csv"

NON_LABEL_COLUMNS = ["cve_id", "cleaned", "matchers", "merged", "reference", "description_and_reference", "year"]


def read_dataset() -> pd.DataFrame:
    description_data = pd.read_csv(DESCRIPTION_DATA_PATH)
    # reference_data = pd.read_csv(REFERENCE_DATA_PATH, index_col = 0)
    # combined_data = combine_description_and_reference_data(desc=description_data, ref=reference_data)
    return description_data

                

def count_datapoint_for_each_year():
    df = read_dataset()
    os.makedirs("zero_shot_dataset", exist_ok=True)
    labels = [i for i in df.columns if i not in NON_LABEL_COLUMNS]
    df['year'] = df['cve_id'].str.split('-').str[1]
    # print(df['year'])
    df['year'] = df['year'].astype('int')
    print(df['year'].unique())
    print(labels[1590])
    print(np.where(df[labels].to_numpy().sum(axis=0).flatten() > 0)[0].shape[0])
    label_per_year = {}
    for threshold in [2014, 2015, 2016, 2017, 2018, 2019]:
        data = df[df['year'] == threshold].copy()
        label_per_year[threshold] = np.where(data[labels].to_numpy().sum(axis=0).flatten() > 0)[0]

    for threshold in [2015, 2016, 2017, 2018, 2019]:
        print("============================")
        print(threshold)
        data = df[df['year'] == threshold].copy()
        np_data = data[labels].to_numpy()
        unseen_label = []
        seen_label = []
        for i in label_per_year[threshold]:
            if i not in label_per_year[threshold-1]:
                unseen_label.append(i)
            else:
                seen_label.append(i)
        
        print("seen: {}".format(len(seen_label)))
        print("unseen: {}".format(len(unseen_label)))
        count_unseen = 0
        count_seen = 0
        for i in range(len(data['cve_id'])):
            is_unseen = True
            is_seen = True

            for idx in np.where(np_data[i] == 1)[0]:
                if idx in seen_label:
                    is_unseen = False
                if idx in unseen_label:
                    is_seen = False
            if is_unseen:
                count_unseen += 1

            if is_seen:
                count_seen += 1
        remain = len(data['cve_id']) - count_seen - count_unseen
        print("unseen dp: {}/{} = {}".format(count_unseen, len(data['cve_id']), count_unseen/len(data['cve_id'])))
        print("seen dp: {}/{} = {}".format(count_seen, len(data['cve_id']), count_seen/len(data['cve_id'])))
        print("remain dp: {}/{} = {}".format(remain, len(data['cve_id']), remain/len(data['cve_id'])))

if __name__ == "__main__":
    count_datapoint_for_each_year()
