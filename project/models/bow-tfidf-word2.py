# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:05:44 2021

@author: Yasel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from time import time 


X_train, X_validation, y_train, y_validation = train_test_split(train_df['review_stemmed'],train_df['label'], test_size = 0.2, random_state = 1234)

## Bag Of Words 

## BoW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
X_train = X_train.dropna()
countV = CountVectorizer() # Bag Of Words
countV.fit_transform(X_train) # Fit the dictionnary

## TF-IDF 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfV = TfidfVectorizer()
tfidfV.fit_transform(X_train)

## Word2Vec 
import gensim

tweets = X_train.values

model_w2v = gensim.models.Word2Vec(
            tweets,
            size=200, # desired no. of features/independent variables
            window=8, # context window size
            min_count=2, # Ignores all words with total frequency lower than 2.                                  
            sg = 5, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 32, # no.of cores
            seed = 34
) 

model_w2v.train(tweets, total_examples= len(tweets), epochs=20)

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


train_size = len(tweets)
train_arrays = np.zeros((train_size, 200))

def wordLists(data) :
    size = len(data)
    output = np.zeros((size, 200))
    for i in range(size):
        output[i,:] = word_vector(data[i], 200)
        print(output[i,:])
    return output 