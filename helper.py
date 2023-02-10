import unicodedata
import pandas as pd
import numpy as np

import os

def normalize_non_asci(text: str) -> str:
    return str(unicodedata.normalize('NFKD', text).encode('ascii', 'ignore'))

def remove_non_alphanumeric(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ")

def preprocess_text(text: str) -> str:
    return normalize_non_asci(remove_non_alphanumeric(text))

def strip(s: str) -> str:
    return s.strip()

def get_year_from_cve_id(cve_id: str) -> int:
    return int(cve_id.split("-")[1])

def combine_description_and_reference_data(desc: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame :

    # fill any NaN in the reference text column with an empty string
    desc.fillna("", inplace=True)

    # fill any NaN in the reference text column with an empty string
    ref.fillna("", inplace=True)

    # inner join : merge description data and reference data by cve id
    df = pd.merge(ref, desc, on="cve_id")

    # drop unused columns
    df.drop(columns=["labels", "cpe", "description",
                     "references"], inplace=True)

    # rename column to get more intuitive name
    df.rename(columns={"consolidated_strings": "reference"}, inplace=True)

    # preprocess reference text
    df["reference"] = df["reference"].apply(preprocess_text)

    # concat description and reference
    df["description_and_reference"] = df["merged"] + " " + df["reference"]
    df["description_and_reference"].apply(strip)

    # remove empty text
    df["merged"].replace("", np.nan, inplace=True)
    df["description_and_reference"].replace("", np.nan, inplace=True)
    df = df.dropna(subset=["merged"])
    df = df.dropna(subset=["description_and_reference"])
    
    df["year"] = df["cve_id"].apply(get_year_from_cve_id)

    return df








