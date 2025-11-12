# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:34:49 2025

@author:Kwangho Baek baek0040@umn.edu; dptm22203@gmail.com
"""

import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import OrderedDict
import itertools

def load_and_preprocess_data(config: dict, data_dir: Path):
    """
    Loads, preprocesses, and splits the data for the DCM-SEAL model.

    Args:
        config (dict): A dictionary defining the data structure. Expected keys:
            - 'core_vars' (list[str]): Columns for core utility variables.
            - 'embedding_vars' (list[str]): A list of column names to be
              treated as embedding variables.
            - 'segmentation_vars_categorical' (list[str]): Categorical columns
              for the segmentation network (will be one-hot encoded).
            - 'segmentation_vars_continuous' (list[str]): Continuous columns
              for the segmentation network (will be standardized).
            - 'test_size' (float): Proportion for the test set.
            - 'random_state' (int): Seed for the train/test split.
        data_dir (Path): The path to the main 'data' directory.

    Returns:
        tuple: A tuple containing:
            - train_df (pd.DataFrame): The processed training dataframe.
            - test_df (pd.DataFrame): The processed testing dataframe.
            - train_x_emb (torch.Tensor): The integer tensor for embedding vars (training).
            - test_x_emb (torch.Tensor): The integer tensor for embedding vars (testing).
            - embedding_dims (OrderedDict): The discovered dimensions of the embedding variables.
    """
    # --- Unpack config ---
    core_vars = config.get("core_vars", [])
    emb_vars = config.get("embedding_vars", [])
    seg_vars_cat = config.get("segmentation_vars_categorical", [])
    seg_vars_cont = config.get("segmentation_vars_continuous", [])
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 5723588)

    #Validating variable list exclusivity
    all_lists = {
        "embedding_vars": emb_vars,
        "segmentation_vars_categorical": seg_vars_cat,
        "segmentation_vars_continuous": seg_vars_cont,
        "core_vars": core_vars
    }
    for (name1, list1), (name2, list2) in itertools.combinations(all_lists.items(), 2):
        # Find the intersection of the two sets
        overlap = set(list1) & set(list2)
        if overlap:
            raise ValueError(
                f"Configuration Error: Variable lists '{name1}' and '{name2}' are not mutually exclusive. "
                f"Overlapping variable(s): {list(overlap)}"
            )

    # --- 1. Load data ---
    print(f"Loading data from {data_dir}")
    data_file = data_dir / "dfIn.csv"
    conv_file = data_dir / "dfConv.csv"
    
    try:
        df_in = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Main data file {data_file} not found.")

    # Validate that all specified variables exist in the dataframe ---
    all_specified_vars = set(emb_vars + seg_vars_cat + seg_vars_cont + core_vars)
    # Find which variables are specified but not available
    missing_vars = all_specified_vars - set(df_in.columns)

    if missing_vars:
        raise ValueError(f"Missing variables in config but not in the input: {list(missing_vars)}")
    
    # --- 2. Apply contextual conversion rules, if a conversion file exists ---
    if conv_file.is_file():
        print(f"Conversion file {conv_file} found. Applying rules...")
        df_conv = pd.read_csv(conv_file)
        for field in df_conv['field'].unique():
            if field in df_in.columns:
                mapping = df_conv[df_conv['field'] == field].set_index('orilevel')['newlevel'].to_dict()
                df_in[field] = df_in[field].replace(mapping)
    #df_in.iloc[1::config['n_alternatives']].to_csv('converted.csv',index=False)

    # --- 3. Discover Embedding Dims & Integer (label) Encode ---
    embedding_dims = OrderedDict() #need this to get trained weight matrix 
    embedding_labels = OrderedDict()
    if emb_vars: # if exists
        for var in emb_vars:
            # Treat the column as categorical first
            df_in[var] = df_in[var].astype(str)
            # Now, apply the LabelEncoder to the string-based categories
            le = LabelEncoder()
            df_in[var] = le.fit_transform(df_in[var])
            labels = [f'{var}_{level}' for level in list(le.classes_)]  # e.g., ['month_Jan', 'month_Feb',...] 
            count  = len(labels)
            # Discover the number of unique categories for this variable
            embedding_dims[var] = count
            embedding_labels[var] = (count, labels)
    embedding_dict={'dims' : embedding_dims, 'labels':embedding_labels}
    print(f"Discovered embedding dimensions and integer encoding: {embedding_dims}")

    # --- 4. One-Hot Encode Segmentation (categorical) Variables ---
    if seg_vars_cat:
        cols_to_encode = [col for col in seg_vars_cat if col in df_in.columns]
        df_processed = pd.get_dummies(df_in, columns=cols_to_encode, dummy_na=False)
        # pd.get_dummies() creates boolean columns by default. We convert them here.
        encoded_cols = [col for col in df_processed.columns if df_processed[col].dtype == 'bool']
        df_processed[encoded_cols] = df_processed[encoded_cols].astype(int)
    else:
        df_processed = df_in.copy()

    # --- 5. Split Data by Group (by 'chid') ---
    print("Splitting training/test data by 'chid'; Need to do this before standardization to prevent data leakage")
    unique_ids = df_processed['chid'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)
    train_df = df_processed[df_processed['chid'].isin(train_ids)].copy()
    train_df.sort_values(['chid', 'alt'], inplace=True)
    test_df = df_processed[df_processed['chid'].isin(test_ids)].copy()
    test_df.sort_values(['chid', 'alt'], inplace=True)

    # --- 6. Create Embedding Tensors ---
    if emb_vars: # if exists
        train_x_emb = torch.tensor(train_df[emb_vars].values, dtype=torch.long)
        test_x_emb = torch.tensor(test_df[emb_vars].values, dtype=torch.long)
    else:
        train_x_emb = torch.empty((len(train_df), 0), dtype=torch.long)
        test_x_emb = torch.empty((len(test_df), 0), dtype=torch.long)

    # --- 7. Standardize Numerical Columns ---
    if seg_vars_cont:
        print(f"Standardizing numerical columns, excluding cores: {core_vars}")
        scaler = StandardScaler()
        cols_to_scale = [v for v in seg_vars_cont if v in train_df.columns]

        if cols_to_scale:
            train_df[cols_to_scale] = train_df[cols_to_scale].astype('float64')
            test_df[cols_to_scale] = test_df[cols_to_scale].astype('float64')
            scaler.fit(train_df[cols_to_scale])
            train_df.loc[:, cols_to_scale] = scaler.transform(train_df[cols_to_scale])
            test_df.loc[:, cols_to_scale] = scaler.transform(test_df[cols_to_scale])
            print(f"Standardized columns: {cols_to_scale}")

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, test_df, train_x_emb, test_x_emb, embedding_dict


if __name__ == '__main__':
    #integrity check
    import numpy as np
    df = pd.read_csv('./data/SwissMetro/dfIn.csv')
    match_sum_per_chid = df.groupby('chid')['match'].sum()
    is_match_sum_always_one = np.all(match_sum_per_chid == 1)
    gapped_chids = []
    for chid, group in df.groupby('chid'):
        alt_ids = sorted(group['alt'].unique())
        if alt_ids != list(range(alt_ids[0], alt_ids[0] + len(alt_ids))):
            gapped_chids.append(chid)
    if gapped_chids:
        print(gapped_chids)


