# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:23:47 2019

@author: SRIJAN
"""

# import os 
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns


df = pd.read_csv("./dataset/spam.csv", encoding = 'latin-1')
df.head()

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
df = df.rename(columns = {'v1':'label', 'v2':'message'})
df.head()

sns.countplot(df.label)
plt.xlabel('Label')
plt.ylabel('Number of ham and spam messages')

df.groupby('label').describe()

df['lenght'] = df['message'].apply(len)


mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
df.hist(column = 'lenght', by = 'label', bins = 50, figsize = (11,5))