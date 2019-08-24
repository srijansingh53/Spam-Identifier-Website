# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:56:02 2019

@author: SRIJAN
"""
import sys
import re
import pickle
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer
from nltk.stem import WordNetLemmatizer 

df = pd.read_csv('./dataset/spam.csv', encoding='latin-1')[['v1', 'v2']]
df = df.rename(columns = {'v1':'label', 'v2':'message'})

# nltk.download('stopwords') #only if you have not downloaded the stopwords of nltk
def preprocess_text(text):
    # remove all punctuation
    text = re.sub(r'[^\w\d\s]', ' ', text)
    # collapse all white spaces
    text = re.sub(r'\s+', ' ', text)
    # convert to lower case
    text = re.sub(r'^\s+|\s+?$', '', text.lower())
    # remove stop words and perform stemming
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer() 
    return ' '.join(
        lemmatizer.lemmatize(term) 
        for term in text.split()
        if term not in set(stop_words)
    )

df['processed_text'] = df.message.apply(lambda row : preprocess_text(row))

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1,2))
tfidf_data = tfidf_vec.fit_transform(df.processed_text)
tfidf_data = pd.DataFrame(tfidf_data.toarray())

spam_filter = pickle.load(open('./model/MNB.sav','rb'))


s = sys.argv[1]

text = [preprocess_text(s)]
t = tfidf_vec.transform(text)
p = np.array(t.toarray())
print(spam_filter.predict(p)[0])



