# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 01:56:02 2019

@author: SRIJAN
"""

import pickle


model = pickle.load(open('MNB.sav', 'rb'))

def detect_spam(s):
    return spam_filter.predict([s])[0]
detect_spam("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info")
