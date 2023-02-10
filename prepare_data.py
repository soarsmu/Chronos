import pandas as pd
import numpy as np
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from skmultilearn.model_selection import iterative_train_test_split
import json
import sklearn
import pickle
import os
from helper import combine_description_and_reference_data
import sys
from spiral import ronin

# REF_NAME = sys.argv[1]

BASE_PATH = "dataset/description_data"
DESCRIPTION_DATA_PATH = f"{BASE_PATH}/dataset_merged_cleaned.csv"
REFERENCE_DATA_PATH = f"dataset/reference_data/reference_data_raw_0.5_15.csv"

# print(f"We use: reference_data_raw_{REF_NAME}.csv")

DATASET_PATH = "zero_shot_dataset/zero_shot_train_cleaned.csv"
NON_LABEL_COLUMNS = ["cve_id", "cleaned", "matchers", "merged", "reference", "description_and_reference", "year"]
FEATURE_NAME = sys.argv[1]
LABEL_SPLITTING = sys.argv[2]
# FEATURE_NAME = "description_and_reference"
# FEATURE_NAME = "cleaned"
# FEATURE_NAME = "merged"

DESCRIPTION_FIELDS = ["cve_id", FEATURE_NAME]

def read_dataset() -> pd.DataFrame:
    description_data = pd.read_csv(DESCRIPTION_DATA_PATH)
    reference_data = pd.read_csv(REFERENCE_DATA_PATH, index_col = 0)
    combined_data = combine_description_and_reference_data(desc=description_data, ref=reference_data)
    return combined_data

XF_ROW = 0
YF_ROW = 0


def get_row_Xf():
    num_lines = sum(1 for line in open('zero_shot_dataset/zestxml/Xf.txt'))
    return str(num_lines)

def get_row_Yf(): 
    num_lines = sum(1 for line in open('zero_shot_dataset/zestxml/Yf.txt'))
    return str(num_lines)

# Split the data for zero shot training
# The dataset is split chronologically
# 75% training, 25% testing just like Chen et al. paper
# The idea is that the test data is newer and will contain new libraries that are not seen in the training data
def zero_shot_data_splitting():
    df = read_dataset()
    os.makedirs("zero_shot_dataset", exist_ok=True)

    df['year'] = df['cve_id'].str.split('-').str[1]
    # print(df['year'])
    df['year'] = df['year'].astype('int')
    threshold_down = 2016
    threshold_up = 2018
    '''trian_data'''
    temp_train = df[df['year'] <= threshold_down].copy()
    # temp_train = temp_train[temp_train['year'] <= threshold_down].copy()

    test = df[df['year'] >= threshold_up].copy()
    temp_train['id_after_year'] = temp_train['cve_id'].str.split('-').str[2]
    temp_train['id_after_year'] = temp_train['id_after_year'].astype('int')
    number_val = 5 * (len(temp_train) // 100)
    number_train = len(temp_train) 
    train = temp_train.sort_values(by=["year", "id_after_year"]).head(number_train)
    val = temp_train.sort_values(by=["year", "id_after_year"]).tail(number_val)
    print(train)
    print(val)
    print(test)
    del train["year"]
    del train["id_after_year"]
    del val["year"]
    del val["id_after_year"]
    del test["year"]

    train.drop(train.filter(regex="Unname"), axis=1, inplace=True)
    test.drop(test.filter(regex="Unname"), axis=1, inplace=True)
    val.drop(val.filter(regex="Unname"), axis=1, inplace=True)

    print("train size: ", len(train))
    print("test size: ", len(test))
    print("val size: ", len(val))

    train.to_csv("zero_shot_dataset/zero_shot_train_cleaned.csv", index=False)
    test.to_csv("zero_shot_dataset/zero_shot_test_cleaned.csv", index=False)
    val.to_csv("zero_shot_dataset/zero_shot_val_cleaned.csv", index=False)


# this function is used to save the splitted dataset as numpy file
# use this function if the csv files are already splitted into the test and train dataset
def save_splitted_zero_shot_dataset_validation_as_numpy():
    TRAIN_PATH = "zero_shot_dataset/zero_shot_train_cleaned.csv"
    TEST_PATH = "zero_shot_dataset/zero_shot_test_cleaned.csv"
    VAL_PATH = "zero_shot_dataset/zero_shot_val_cleaned.csv"
    description_fields = DESCRIPTION_FIELDS
    # Initiate the dataframe containing the CVE ID and its description
    # Change the "FEATURE_NAME" field in the description_fields variable to use other text feature such as reference

    # Process the training dataset

    df = pd.read_csv(TRAIN_PATH, usecols=description_fields)
    df[FEATURE_NAME] = df[FEATURE_NAME].astype(str)
    # Read column names from file
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE

    pd_labels = pd.read_csv(TRAIN_PATH,
                            usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels = [i for i in cols if i not in NON_LABEL_COLUMNS]

    # Convert to numpy for splitting
    train = df.to_numpy()
    label_train = pd_labels.to_numpy()

    df_test = pd.read_csv(TEST_PATH, usecols=description_fields)
    df_test[FEATURE_NAME] = df_test[FEATURE_NAME].astype(str)
    pd_labels_test = pd.read_csv(TEST_PATH,
                                 usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    test = df_test.to_numpy()
    label_test = pd_labels_test.to_numpy()

    df_val = pd.read_csv(VAL_PATH, usecols=description_fields)
    df_val[FEATURE_NAME] = df_val[FEATURE_NAME].astype(str)
    pd_labels_val = pd.read_csv(VAL_PATH,
                                usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    val = df_val.to_numpy()
    label_val = pd_labels_val.to_numpy()

    # Save the splitted data to files
    os.makedirs("zero_shot_dataset/splitted_val", exist_ok=True)
    np.save("zero_shot_dataset/splitted_val/splitted_train_x.npy", train, allow_pickle=True)
    np.save("zero_shot_dataset/splitted_val/splitted_train_y.npy", label_train, allow_pickle=True)
    np.save("zero_shot_dataset/splitted_val/splitted_test_x.npy", test, allow_pickle=True)
    np.save("zero_shot_dataset/splitted_val/splitted_test_y.npy", label_test, allow_pickle=True)
    np.save("zero_shot_dataset/splitted_val/splitted_val_x.npy", test, allow_pickle=True)
    np.save("zero_shot_dataset/splitted_val/splitted_val_y.npy", label_test, allow_pickle=True)



# get additional training data in the form of the label string
def get_label_training_data():
    TRAIN_PATH = "zero_shot_dataset/zero_shot_train_cleaned.csv"
    TEST_PATH = "zero_shot_dataset/zero_shot_test_cleaned.csv"
    VAL_PATH = "zero_shot_dataset/zero_shot_val_cleaned.csv"
    cols = list(pd.read_csv(TRAIN_PATH, nrows=1))
    # Initiate the dataframe containing the labels for each CVE

    pd_labels = pd.read_csv(TRAIN_PATH,
                            usecols=[i for i in cols if i not in NON_LABEL_COLUMNS])
    # Initiate a list which contain the list of labels considered in te dataset
    list_labels = [i for i in cols if i not in NON_LABEL_COLUMNS]

    new_df = pd.read_csv(TRAIN_PATH, nrows=1).copy()
    list_new_df = []
    for i, label in enumerate(list_labels):
        vals = [i, label, label, label]
        for label_2 in list_labels:
            if label_2 == label:
                vals.append(1)
            else:
                vals.append(0)
        list_new_df.append(vals)

    result = pd.DataFrame(list_new_df, columns=cols)
    result.to_csv("zero_shot_dataset/labels_string.csv", index=False)


# THERE ARE SO MANY THINGS TO PREPARE FOR ZESTXML:
# DONE Xf.txt: all features used in tf-idf representation of documents ((trn/tst/val)_X_Xf), ith line denotes ith feature in the tf-idf representation. In particular, for datasets used in the paper, it's the stemmed bigram and unigram features of documents but you can choose to have any set of features depending on your application.
# DONE Yf.txt: similar to Xf.txt it represents features of all labels. In addition to unigrams and bigrams, we also add a unique feature specific to each label (represented by __label__<i>__<label-i-text>, this feature will only be present in ith label's features), this allows the model to have label specific parameters and helps it to do well on many-shot labels. Features with __parent__ in them are only specific to the GZ-EURLex-4.3K dataset because raw labels in this dataset have some additional information about parent concepts of each label, you can safely choose to ignore these features for any other/new dataset.
# DONE (trn/tst/val)_X_Xf.txt: sparse matrix (documents x document-features) representing tf-idf feature matrix of (trn/tst/val) input documents.
# DONE Y_Yf.txt: similar to (trn/tst/val)_X_Xf.txt but for labels, this is the sparse matrix (labels x label-features) representing tf-idf feature matrix of labels.
# trn_Y_Yf.txt: similar to Y_Yf.txt but contains features for only the seen labels (can be interpreted as Y_Yf[seen-labels])
# DONE (trn/tst/val)_X_Y.txt: sparse matrix (documents x labels) representing (trn/tst/val) document-label relevance matrix.


# helper function for trn_Y_Yf
# get the list of seen labels from a csv file
TRAINING_DATA_PATH = "zero_shot_dataset/zero_shot_train_cleaned.csv"
def get_list_labels(csv_file_path):
    seen_label = []
    df = pd.read_csv(csv_file_path, usecols=LABEL_COLUMNS)
    for label in LABEL_COLUMNS:
        sum = df[label].sum()
        if sum > 0:
            seen_label.append(label)
    return seen_label


# should be similar with the regular Y_Yf
# just need to find out which labels are seen in the training data
import regex as re


def prepare_zest_trn_Y_Yf(vectorizer):
    list_labels = get_list_labels(TRAINING_DATA_PATH)
    # add the unique label features
    for i in range(0, len(list_labels)):
        label = list_labels[i]
        s = re.sub(r"[^\w\s]", '_', label)
        # formatted = "__label__" + i.__str__() + "__" + label.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(";", "_")
        if LABEL_SPLITTING == "splitting":
            list_labels[i] = list_labels[i] 
        else:
            formatted = "__label__" + i.__str__() + "__" + s.replace(" ", "_")
            list_labels[i] = list_labels[i]  + " " + formatted

    with open("zero_shot_dataset/zestxml/trn_Y_Yf.txt", "w", encoding="utf-8") as wr:
        # header is number of labels SPACE number of features (i.e., numrows of Yf.txt)
        wr.write(len(list_labels).__str__() + " "+YF_ROW+"\n")
        for label in list_labels:
            sparse_mat = vectorizer.transform([label])
            value = sparse_mat.data
            indices = sparse_mat.indices
            sorted_value = [x for _, x in sorted(zip(indices, value))]
            sorted_indices = sorted(indices)
            # printing the tfidf values
            to_print = ""
            for i in range(0, len(sorted_value)):
                to_print = to_print + sorted_indices[i].__str__() + ":" + sorted_value[i].__str__() + " "
            to_print = to_print[:-1] + "\n"
            wr.write(to_print)


# make use of the list of labels and the vectorizer created during the Yf.txt creation
# potentially buggy as we did not consider the __label__ features
# BUG IS FIXED
def prepare_zest_Y_Yf(label_column, vectorizer):
    with open("zero_shot_dataset/zestxml/Y_Yf.txt", "w", encoding="utf-8") as wr:
        # header is number of labels SPACE number of features (i.e., numrows of Yf.txt)
        wr.write("2817 "+YF_ROW+"\n")
        for label in label_column:
            sparse_mat = vectorizer.transform([label])
            value = sparse_mat.data
            indices = sparse_mat.indices
            sorted_value = [x for _, x in sorted(zip(indices, value))]
            sorted_indices = sorted(indices)
            # printing the tfidf values
            to_print = ""
            for i in range(0, len(sorted_value)):
                to_print = to_print + sorted_indices[i].__str__() + ":" + sorted_value[i].__str__() + " "
            to_print = to_print[:-1] + "\n"
            wr.write(to_print)


# process from the svmlight file
# generates a total of 6 files, which are the three X_Xf files
# and the three X_Y files
def prepare_zest_X_Xf_and_X_Y():
    with open("zero_shot_dataset/zestxml/trn_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("zero_shot_dataset/zestxml/trn_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("zero_shot_dataset/zestxml/trn_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " "+XF_ROW+"\n")
        xy_wr.write(num_rows.__str__() + " 2817\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()

    with open("zero_shot_dataset/zestxml/tst_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("zero_shot_dataset/zestxml/tst_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("zero_shot_dataset/zestxml/tst_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " "+XF_ROW+"\n")
        xy_wr.write(num_rows.__str__() + " 2817\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()

    with open("zero_shot_dataset/zestxml/val_svmlight.txt", "r", encoding="utf-8") as re:
        lines = re.read().splitlines()
        num_rows = len(lines)
        xf_wr = open("zero_shot_dataset/zestxml/val_X_Xf.txt", "w", encoding="utf-8")
        xy_wr = open("zero_shot_dataset/zestxml/val_X_Y.txt", "w", encoding="utf-8")
        # write the header: num_rows num_cols
        # num_cols is taken from the Xf.txt and from the number of labels in the dataset respectively
        xf_wr.write(num_rows.__str__() + " "+XF_ROW+"\n")
        xy_wr.write(num_rows.__str__() + " 2817\n")
        for line in lines:
            line = line.strip()
            # split into 2, the [0] is labels, [1] is TfIdf features
            split = line.split(" ", 1)
            xf_wr.write(split[1] + "\n")
            # for the labels, split further based on comma
            label_text = ""
            for label in split[0].split(","):
                label_text = label_text + label + ":1.00000 "
            label_text = label_text[:-1] + "\n"
            xy_wr.write(label_text)
        xf_wr.close()
        xy_wr.close()
        re.close()


# Possibly for this one is similar to SVMLight format without the labels at the beginning
# Use the Vectorizer created during the Xf.txt creation
# this function will generate the svmlight first
# which will then be processed into X_Xf.txt and trn/tst/val_X_Y.txt
def prepare_zest_svmlight(vectorizer):
    # Load the splitted dataset files
    train = np.load("zero_shot_dataset/splitted_val/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("zero_shot_dataset/splitted_val/splitted_train_y.npy", allow_pickle=True)
    test = np.load("zero_shot_dataset/splitted_val/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("zero_shot_dataset/splitted_val/splitted_test_y.npy", allow_pickle=True)
    val = np.load("zero_shot_dataset/splitted_val/splitted_val_x.npy", allow_pickle=True)
    label_val = np.load("zero_shot_dataset/splitted_val/splitted_val_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    val_corpus = val[:, 1].tolist()
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]

    vectorizer = vectorizer

    train_X = vectorizer.transform(train_corpus)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    test_Y = label_test
    val_X = vectorizer.transform(val_corpus)
    val_Y = label_val

    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "zero_shot_dataset/zestxml/trn_svmlight.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "zero_shot_dataset/zestxml/tst_svmlight.txt", multilabel=True)
    dump_svmlight_file(val_X, val_Y, "zero_shot_dataset/zestxml/val_svmlight.txt", multilabel=True)
    #


# Prepare the Xf.txt, which contains all features used in tf-idf representation of documents
# Therefore I assume it would be
# 1. Create TfIdfVectorizer using all the text dataset
# 2. The TfIdfVectorizer uses Unigram and Bigram
# 3. Then, get the vocabulary dictionary (i.e., TfIdfVectorizer.vocabulary
# return the TfIdfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def prepare_zestxml_Xf():
    df = read_dataset()
    text_corpus = df[FEATURE_NAME].values.astype("U")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(text_corpus)
    os.makedirs("zero_shot_dataset/zestxml/", exist_ok=True)
    with open("zero_shot_dataset/zestxml/Xf.txt", "w", encoding="utf-8") as wr:
        for key in sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get):
            wr.write(key + "\n")
    global XF_ROW
    XF_ROW = get_row_Xf() 
    return vectorizer


# Simply the list of labels, in the form of unigram, bigram, and unique __label__ format
def prepare_zestxml_Yf():
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]
    label_labels = []

    # solve tokenize with Spiral
    if LABEL_SPLITTING == "splitting":
        for i in range(len(label_columns)):
            splitted_label = ronin.split(label_columns[i])
            for word in splitted_label:
                label_columns[i] = label_columns[i] + " " + word

    for i, label in enumerate(label_columns):
        import regex as re
        s = re.sub(r"[^\w\s]", '_', label)
        # formatted = "__label__" + i.__str__() + "__" + label.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace(";", "_")
        if LABEL_SPLITTING != "splitting":
            formatted = "__label__" + i.__str__() + "__" + s.replace(" ", "_")
            label_labels.append(formatted)
    # print(label_labels)
    if LABEL_SPLITTING == "splitting":
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit((label_columns + label_labels))
    with open("zero_shot_dataset/zestxml/Yf.txt", "w", encoding="utf-8") as wr:
        for key in sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get):
            wr.write(key + "\n")
        # for label in label_labels:
        #     wr.write(label + "\n")
    for i in range(0, len(label_columns)):
        label_columns[i] = label_columns[i] 
        if LABEL_SPLITTING == "splitting":
            label_columns[i] = label_columns[i] 
        else:
            label_columns[i] = label_columns[i] + " " + label_labels[i]
    global YF_ROW
    YF_ROW = get_row_Yf() 
    return label_columns, vectorizer


def prepare_zestxml_dataset():
    tfidf_vectorizer = prepare_zestxml_Xf()
    label_column, label_vectorizer = prepare_zestxml_Yf()
    prepare_zest_svmlight(tfidf_vectorizer)
    prepare_zest_X_Xf_and_X_Y()
    prepare_zest_Y_Yf(label_column, label_vectorizer)
    prepare_zest_trn_Y_Yf(label_vectorizer)

# the test and train data are the same with omikuji
# however, you need to create the train/test_labels.txt and train/test_texts.txt
# with each row contains the text and labels for the train/test data
def prepare_lightxml_dataset():
    # Load the splitted dataset files
    train = np.load("zero_shot_dataset/splitted_val/splitted_train_x.npy", allow_pickle=True)
    label_train = np.load("zero_shot_dataset/splitted_val/splitted_train_y.npy", allow_pickle=True)
    test = np.load("zero_shot_dataset/splitted_val/splitted_test_x.npy", allow_pickle=True)
    label_test = np.load("zero_shot_dataset/splitted_val/splitted_test_y.npy", allow_pickle=True)
    train_corpus = train[:, 1].tolist()
    test_corpus = test[:, 1].tolist()
    cols = list(pd.read_csv(DATASET_PATH, nrows=1))
    label_columns = [i for i in cols if i not in NON_LABEL_COLUMNS]
    num_labels = len(label_columns)

    vectorizer = TfidfVectorizer().fit(train_corpus)

    idx_zero_train = np.argwhere(np.all(label_train[..., :] == 0, axis=0))
    idx_zero_test = np.argwhere(np.all(label_test[..., :] == 0, axis=0))

    train_X = vectorizer.transform(train_corpus)
    # train_Y = np.delete(label_train, idx_zero_train, axis=1)
    train_Y = label_train
    test_X = vectorizer.transform(test_corpus)
    # test_Y = np.delete(label_test, idx_zero_test, axis=1)
    test_Y = label_test

    num_features = len(vectorizer.get_feature_names())
    num_row_train = train_X.shape[0]
    num_row_test = test_X.shape[0]

    # Dump the standard svmlight file
    dump_svmlight_file(train_X, train_Y, "zero_shot_dataset/lightxml/train.txt", multilabel=True)
    dump_svmlight_file(test_X, test_Y, "zero_shot_dataset/lightxml/test.txt", multilabel=True)

    train_text = []
    train_label = []
    test_text = []
    test_label = []

    cve_labels = pd.read_csv("dataset/description_data/cve_labels.csv")

    train_data = pd.read_csv("zero_shot_dataset/zero_shot_train_cleaned.csv")
    # process the label and text here
    for index, row in train_data.iterrows():
        train_text.append(row[FEATURE_NAME].lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        train_label.append(label_string)

    test_data = pd.read_csv("zero_shot_dataset/zero_shot_test_cleaned.csv")
    for index, row in test_data.iterrows():
        test_text.append(row[FEATURE_NAME].lstrip().rstrip())
        # for label below
        label = cve_labels[cve_labels["cve_id"] == row.cve_id]
        label_unsplit = label.labels.values[0]
        label_array = label_unsplit.split(",")
        label_string = ""
        for label in label_array:
            label_string = label_string + label + " "
        label_string = label_string.rstrip()
        # print(label_string)
        test_label.append(label_string)

    with open("zero_shot_dataset/lightxml/train_texts.txt", "w", encoding="utf-8") as wr:
        for line in train_text:
            wr.write(line + "\n")

    with open("zero_shot_dataset/lightxml/train_labels.txt", "w", encoding="utf-8") as wr:
        for line in train_label:
            wr.write(line + "\n")

    with open("zero_shot_dataset/lightxml/test_texts.txt", "w", encoding="utf-8") as wr:
        for line in test_text:
            wr.write(line + "\n")

    with open("zero_shot_dataset/lightxml/test_labels.txt", "w", encoding="utf-8") as wr:
        for line in test_label:
            wr.write(line + "\n")


if __name__ == "__main__":
    zero_shot_data_splitting()
    print("Data splitting finished")
    COLUMNS = list(pd.read_csv(TRAINING_DATA_PATH, nrows=1))
    LABEL_COLUMNS = [i for i in COLUMNS if i not in NON_LABEL_COLUMNS]
    save_splitted_zero_shot_dataset_validation_as_numpy()
    print("Transform to numpy finished")
    prepare_zestxml_dataset()
    print("ZestXML data preparation finished")
