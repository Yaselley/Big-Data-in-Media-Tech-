# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:09:50 2021

@author: Yasel
"""

#%% Naive Bayes
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

train_df = X_train
test_df = X_validation
y = y_train
y_test = y_validation 
train_dfW2V = wordLists(X_train.values)
test_dfW2V = wordLists(X_validation.values)

NB1B,NB2B,NB3B,NB4B = test_model_BoW(train_df,test_df,y,y_test,[MultinomialNB()],["Naïve Bayes"])
NB1T,NB2T,NB3T,NB4T = test_model_TFIDF(train_df,test_df,y,y_test,[MultinomialNB()],["Naïve Bayes"])

#%% Random Forest 
models,names = [RandomForestClassifier(n_estimators=2000)],["Random Forest"]
model,model_name = models[0],names[0]
RDF1B,RDF2B,RDF3B,RDF4B = test_model_BoW(train_df,test_df,y,y_test,models,names)
RDF1T,RDF2T,RDF3T,RDF4T = test_model_TFIDF(train_df,test_df,y,y_test,models,names)
RDF1W,RDF2W,RDF3W,RDF4W = train_test_model_W2V(model,train_dfW2V,test_dfW2V,y,y_test,model_name)


#%% Linear SVM 
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
C = 1.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C,probability=True)

models = [svc]
names = ["SVC"]
model = models[0]
model_name = names[0]

SVM1B,SVM2B,SVM3B,SVM4B = test_model_BoW(train_df,test_df,y,y_test,models,names)
SVM1T,SVM2T,SVM3T,SVM4T = test_model_TFIDF(train_df,test_df,y,y_test,models,names)
SVM1W,SVM2W,SVM3W,SVM4W = train_test_model_W2V(model,train_dfW2V,test_dfW2V,y,y_test,model_name)

#%% SVM Kernel 

svcrbf = svm.SVC(kernel='rbf', gamma=0.7, C=C,probability=True)

models = [svcrbf]
names = ["SVM Kernel"]
model = models[0]
model_name = names[0]

SVM1B,SVM2B,SVM3B,SVM4B = test_model_BoW(train_df,test_df,y,y_test,models,names)
SVM1T,SVM2T,SVM3T,SVM4T = test_model_TFIDF(train_df,test_df,y,y_test,models,names)
SVM1W,SVM2W,SVM3W,SVM4W = train_test_model_W2V(model,train_dfW2V,test_dfW2V,y,y_test,model_name)

#%% 

