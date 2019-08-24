# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 21:40:40 2019

@author: SRIJAN
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:47:28 2019

@author: SRIJAN
"""
import re
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
from nltk.stem import WordNetLemmatizer 

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
# example 
preprocess_text ('It\'s holiday and we are playing cricket. Jeff is playing very well!!!')

df['processed_text'] = df.message.apply(lambda row : preprocess_text(row))
df.head()

"""------------------Vectorizing words--------------------------"""

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1,2))
tfidf_data = tfidf_vec.fit_transform(df.processed_text)
tfidf_data = pd.DataFrame(tfidf_data.toarray())
tfidf_data.head()


df['message_length'] = df.message.apply(lambda row : len(row))
df.head()

# =============================================================================
# """-----viewing TFID Vectorizer results----------"""
# # testing TFID vectorization
# mess = df.iloc[2]['message']
# print(mess)
# print(tfidfv.transform([mess]))
# 
# # A better view
# j = tfidfv.transform([mess]).toarray()[0]
# print('index\tidf\ttfidf\tterm')
# for i in range(len(j)):
#     if j[i] != 0:
#         print(i, format(tfidfv.idf_[i], '.4f'), format(j[i], '.4f'), tfidfv.get_feature_names()[i],sep='\t')
# 
# 
# =============================================================================

"""----------------------------Training with Multinomial Naive Bayes----------------------"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_data, df['label'], test_size=.2, random_state = 42)

spam_filter = MultinomialNB(alpha=0.2)
spam_filter.fit(X_train, y_train)


"""----------------prediction and inference-------------------"""
predictions = spam_filter.predict(X_test).tolist()
wrong = []
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        print (predictions[i])
        count += 1
        wrong.append(i)

      
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)

from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))

"""-----------------------Testing-------------------"""
text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
text = [preprocess_text(text)]
t = tfidf_vec.transform(text)
p = np.array(t.toarray())
spam_filter.predict(p)[0]

pickle.dump(spam_filter, open('./model/MNB.sav', 'wb'))