# -*- coding: utf-8 -*-
"""
Created on Thu May  7 00:25:03 2020

@author: enriq
"""

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

df1 = pd.DataFrame({'Key':['Apple', 'Banana', 'Orange', 'Strawberry']})
df2 = pd.DataFrame({'Key':['Aple', 'Mango', 'Orag', 'Straw', 'Bannanna', "Tangananica", "Im a Banana", 'Berry']})

def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=2):
    """
    key1 column of the left table
    key2 column of the right table
    threshold is how close the matches should be to return a match, based on Levenshtein distance
    limit is the amount of matches that will get returned, these are sorted high to low
    """
    s = df_2[key2].tolist()

    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
    df_1['matches'] = m

    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2

    return df_1


fuzzy_merge(df1, df2, 'Key', 'Key', threshold=80)
