############################################################################
# Python Code for Feature Extractions, Model Testing, and Model Extraction
############################################################################
# Importing dictionaries

import pandas as pd 
import numpy as np
import scipy as sp
import os
import gc
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def Model_WorldCupTweet(data,labels,test):
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb

    train_text = data[['tweet','tweet_chars']].copy()
    test_text = test[['tweet','tweet_chars']].copy()

    #Label Ecoding
    lb = LabelEncoder()
    train_target = lb.fit_transform(data[labels])

    # TF-IDF Char
    char_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='char',ngram_range=(1,     8),min_df=5,max_df=0.9)
    char_vectorizer.fit(pd.concat([train_text,test_text]).tweet_chars)

    char_train_feat = char_vectorizer.transform(train_text.tweet_chars)
    char_test_feat = char_vectorizer.transform(test_text.tweet_chars)

    # TF-IDF word
    word_vectorizer = TfidfVectorizer(sublinear_tf=True,analyzer='word',ngram_range=(1, 3),min_df=5,max_df=0.9)
    word_vectorizer.fit(pd.concat([train_text,test_text]).tweet)
    word_train_feat = word_vectorizer.transform(train_text.tweet)
    word_test_feat = word_vectorizer.transform(test_text.tweet)
    # Merge Vector(TF-IDF Char,TF-IDF word)
    train_features = hstack([char_train_feat , word_train_feat ]).tocsr()
    test_features = hstack([char_test_feat , word_test_feat ]).tocsr()

    # Save Cross-validation Step's Result 
    sk_train = np.zeros((len(train_target), 3))
    sk_test = np.zeros((len(test), 3))
    sk2_train = np.zeros((len(train_target), 3))
    sk2_test = np.zeros((len(test), 3))

    # Cross-validation
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=521)
    scores = []

    for i, (train_idx, valid_idx) in enumerate(kf.split(train_target, train_target)):
        print('Round: ',i)
        train_x, valid_x = train_features[train_idx], train_features[valid_idx]
        train_y, valid_y = train_target[train_idx], train_target[valid_idx]

    # XGBoost 
        xgb_train = xgb.DMatrix(train_x, train_y)
        xgb_valid = xgb.DMatrix(valid_x, valid_y)

        print("start train")

        params = {
                          'booster': 'gbtree',
                          'objective': 'multi:softprob',
                          'n_estimators' : 2000,
                          'learning_rate' : 0.1,
                          'eta': 0.06,
                          'max_depth': 5,
                          'subsample': 0.85,
                          'colsample_bytree': 0.85,
                          'min_child_weight': 10,
                          'nthread': 8,
                          'seed': 2018,
                          'silent': 1,
                          'num_class': 3
                }

        evallist = [(xgb_valid, 'eval')]
        xgb_model = xgb.train(params, xgb_train, 1000, evallist, early_stopping_rounds=30, verbose_eval=10)
        sk_train[valid_idx] = xgb_model.predict(xgb.DMatrix(valid_x))
        sk_test += xgb_model.predict(xgb.DMatrix(test_features))
        # LogisticRegression
        print('LogisticRegression...')
        classifier = LogisticRegression(dual=True, class_weight='balanced', C=5)
        classifier.fit(train_x, train_y)
        sk_test += classifier.predict_proba(test_features)

        # LightGBM
        print('LightGBM...')
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y)
            
        lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric':'multi_logloss',
         'learning_rate': 0.06,
        'num_leaves': 32, 
        'subsample': 0.85, 
        'colsample_bytree': 0.85, 
        'nthread': 8,
        'seed': 2018,
        'verbose':1,
        'min_data_in_leaf': 50,
        'num_class': 3
        }

        lgb_model = lgb.train(lgb_params, lgb_train, 
        valid_sets=[lgb_train, lgb_valid], 
        valid_names=['train', 'valid'], 
        num_boost_round=2000,
        verbose_eval=10,
        early_stopping_rounds=30
        )

        sk_test += lgb_model.predict(test_features)

        sk_test /= 30

        # Get prob
        test_pred = lb.inverse_transform(np.argmax(sk_test, axis=1))
        submit = test[['timetweet']]
        submit[labels] = test_pred
        return submit



#Training Model for Each Variable_England vs. Colombia
res_eng_colo_efans = Model_WorldCupTweet(train_england_colombia,'England Team Identification',test_england_colombia)

res_eng_colo_cfans = Model_WorldCupTweet(train_england_colombia,'Colombia Team Identification',test_england_colombia)

res_eng_colo_countries = Model_WorldCupTweet(train_england_colombia,'National Identifictaion',test_england_colombia)

res_eng_colo_sentiment = Model_WorldCupTweet(train_england_colombia,'Sentiment',test_england_colombia)

result_england_colombia = pd.concat([res_eng_colo_efans,res_eng_colo_cfans[['Colombia Team Identification']],res_eng_colo_countries[['National Identification']],res_eng_colo_sentiment['Sentiment']],axis=1)

#Training Model for Each Variable_England vs. Croatia
res_eng_croa_efans = Model_WorldCupTweet(train_england_croatia,'England Team Identification',test_england_croatia)

res_eng_croa_cfans = Model_WorldCupTweet(train_england_croatia,'Croatia Team Identification',test_england_croatia)

res_eng_croa_countries = Model_WorldCupTweet(train_england_croatia,'National Identifictaion',test_england_croatia)

res_eng_croa_sentiment = Model_WorldCupTweet(train_england_croatia,'Sentiment',test_england_croatia)

#Print Results
result_england_colombia.head()

result_england_croatia.head()


