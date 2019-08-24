# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:47:28 2019

@author: SRIJAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv('./dataset/spam.csv', encoding='latin-1')[['v1', 'v2']]
df = df.rename(columns = {'v1':'label', 'v2':'message'})
# df.head()

"""-------------------Some text preprocessing--------------------"""

import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer as Stemmer


# nltk.download('stopwords') #only if you have not downloaded the stopwords of nltk

def process(text):
    text = text.lower()
    text = ''.join([t for t in text if t not in string.punctuation])
    text = [t for t in text.split() if t not in stopwords.words('english')]
    st = Stemmer()
    text = [st.stem(t) for t in text]
    return text

# example 
process ('It\'s holiday and we are playing cricket. Jeff is playing very well!!!')

"""------------------Vectorizing words--------------------------"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer(analyzer=process)
data = tfidfv.fit_transform(df['message'])
tfidf_data = pd.DataFrame(data.toarray())
tfidf_data.head()
"""-----viewing TFID Vectorizer results----------"""
# testing TFID vectorization
mess = df.iloc[2]['message']
print(mess)
print(tfidfv.transform([mess]))

# A better view
j = tfidfv.transform([mess]).toarray()[0]
print('index\tidf\ttfidf\tterm')
for i in range(len(j)):
    if j[i] != 0:
        print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')


"""----------------------------Training with Multinomial Naive Bayes----------------------"""
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)), # messages to weighted TFIDF score
    ('classifier', MultinomialNB())                    # train on TFIDF vectors with Naive Bayes
])



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size = 0.2, random_state = 42)

spam_filter.fit(x_train, y_train)

from joblib import dump
dump(spam_filter, 'model.joblib')






pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(spam_filter, file)
pickle.dump(spam_filter, open('MNB.sav', 'wb'))

"""----------------prediction and inference-------------------"""
predictions = spam_filter.predict(x_test)


count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)
x_test[y_test != predictions]

from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))

"""-----------------------Testing-------------------"""
def detect_spam(s):
    return spam_filter.predict([s])[0]
detect_spam("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info")

