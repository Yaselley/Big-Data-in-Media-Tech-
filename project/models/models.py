# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:08:47 2021

@author: Yasel
"""

from nltk.collections import LazyMap
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import xgboost as xgb

def test_model_BoW(train_df,test_df,y,y_test,models,names) :
    L,S = [[] for i in range(len(names))],[[] for i in range(len(names))]
    L2,S2 = [],[]
    for i in range(len(models)):
        model_bow = Pipeline([('countV_bayes',countV),('bayes_classifier',models[i])])
        model_bow.fit(train_df,y)
        y_pred_train = model_bow.predict(train_df)
        y_pred_test = model_bow.predict(test_df)
        probas_train = model_bow.predict_proba(train_df)
        probas_test = model_bow.predict_proba(test_df)
        accuracy_train = np.mean(y_pred_train == y)
        accuracy_test = np.mean(y_pred_test == y_test)
        f1_train = f1_score(y_pred_train,y, average='macro')
        f1_test = f1_score(y_pred_test,y_test, average='macro')
        L[i].append(accuracy_train)
        S[i].append(accuracy_test)
        L[i].append(f1_train)
        S[i].append(f1_test)
        L2.append(probas_train)
        S2.append(probas_test)

        print('\n\n\n')
        print("#################Start Train and Test BoW with {}#################".format(names[i]))
        print("For training score Using BoW We reached {} as accuracy".format(accuracy_train))
        print("For testing score Using BoW We reached {} as accuracy".format(accuracy_test))
        print("For training score Using BoW We reached {} as f1-score".format(f1_train))
        print("For testing score Using BoW We reached {} as f1-score".format(f1_test))
        print("##################End Train and Test BoW with {}##################".format(names[i]))
        print('\n\n\n')
    
    return L,S,L2,S2
    
    
def test_model_TFIDF(train_df,test_df,y,y_test,models,names) : 
    L,S = [[] for i in range(len(names))],[[] for i in range(len(names))]
    L2,S2 = [],[]
    for i in range(len(models)):
        model_TFIDF = Pipeline([('tfidfv_bayes',tfidfV),('bayes_classifier',models[i])])
        model_TFIDF.fit(train_df,y)
        y_pred_train = model_TFIDF.predict(train_df)
        y_pred_test = model_TFIDF.predict(test_df)
        accuracy_train = np.mean(y_pred_train == y)
        accuracy_test = np.mean(y_pred_test == y_test)
        probas_train = model_TFIDF.predict_proba(train_df)
        probas_test = model_TFIDF.predict_proba(test_df)
        f1_train = f1_score(y_pred_train,y,average='macro')
        f1_test = f1_score(y_pred_test,y_test,average='macro')
        L[i].append(accuracy_train)
        S[i].append(accuracy_test)
        L[i].append(f1_train)
        S[i].append(f1_test)
        print('\n\n\n')
        print("#################Start Train and Test TF-IDF with {}#################".format(names[i]))
        print("For training score Using TFIDF We reached {} as accuracy".format(accuracy_train))
        print("For testing score Using TFIDF We reached {} as accuracy".format(accuracy_test))
        print("For training score Using TFIDF We reached {} as f1-score".format(f1_train))
        print("For testing score Using TFIDF We reached {} as f1-score".format(f1_test))
        print("#################End Train and Test TF-IDF with {}###################".format(names[i]))
        print('\n\n\n')
        L2.append(probas_train)
        S2.append(probas_test)
    
    return L,S,L2,S2

def train_test_model_W2V(model,train_df,test_df_,y,y_test,model_name) : 
    
    model_w2v = model
    model_w2v.fit(train_df,y)
    print("trained Word2vec")
    probas_train = model_w2v.predict_proba(train_df)
    probas_test = model_w2v.predict_proba(test_df_)
    y_pred_train = model_w2v.predict(train_df)
    y_pred_test = model_w2v.predict(test_df_)
    auc_train = roc_auc_score(pd.get_dummies(y).values, probas_train, multi_class="ovo")
    auc_test = roc_auc_score(pd.get_dummies(y_test).values, probas_test, multi_class="ovo")
    accuracy_train = np.mean(y_pred_train == y)
    accuracy_test = np.mean(y_pred_test == y_test)
    f1_train = f1_score(y_pred_train,y,average='macro')
    f1_test = f1_score(y_pred_test,y_test,average='macro')

    print("For training score Using "+ model_name +" we have {} as auc score, {} as accuracy".format(auc_train,accuracy_train))
    print("For testing score Using " + model_name +" we have {} as auc score, {} as accuracy".format(auc_test,accuracy_test))
    print("END##############################")
    return f1_train,f1_test,y_pred_train,y_pred_test

#%%