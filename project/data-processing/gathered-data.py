# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:55:13 2021

@author: Yasel
"""

import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
import re
from nltk.tokenize import WordPunctTokenizer
from pathlib import Path
import bz2

## Data Extraction 
def get_data(file_name):
    reviews = bz2.BZ2File(file_name).readlines()
    reviews = [review.decode("utf-8") for review in reviews]
    target = {'1':0, '2':1}

    label = [target[label[9]] for label in reviews]
    reviews = [review[11:] for review in reviews]
    df = pd.DataFrame(data = {"label":label, "reviews": reviews})
    return df

## Handling Missing Values
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def missing_zero_values_table(df,df_name):
    '''
    Inputs:
        df: pandas table
        df_name: string of the pandas table name
    Output:
        "df_name has columns_nb columns and rows_nb Rows. There are columns_nb columns that have missing values."
    '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
    mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
    mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[mz_table.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    print (df_name + " has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str(mz_table.shape[0]) + " columns that have missing values.")
    return mz_table
    
missing_zero_values_table(train_df,"train_df")
missing_zero_values_table(test_df,"test_df")

