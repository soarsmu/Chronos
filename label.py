import sys
import numpy as np
from utils import *

PREDICTION_PATH = 'zestxml/prediction_no_ref.txt'
NON_LABEL_COLUMNS = ["cve_id", "cleaned", "matchers",
                     "merged", "reference", "description_and_reference", "year"]
TRAINING_DATA_PATH = "zero_shot_dataset/zero_shot_train_cleaned.csv"
COLUMNS = list(pd.read_csv(TRAINING_DATA_PATH, nrows=1))
LABEL_COLUMNS = [i for i in COLUMNS if i not in NON_LABEL_COLUMNS]

for label in LABEL_COLUMNS:
    print(label)