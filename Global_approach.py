#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
The code is divided into 3 parts.

PART 1: Testing several models with weighted class
PART 2: Testing several models with weighted class and by adding new features
        - The best model is the random forest model in PART 2 
        
PART 3: Testing several models with undersampling technique and by adding new features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import read_csv

from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn import ensemble


#Function that split each account into dict
def split_accounts_into_dict(trans):

    unique_accounts = pd.unique(trans['accountNumber'])
    account_story = dict()
    
    for a in unique_accounts:
        d = trans.query('accountNumber == @a') 
        d.sort_values(by = 'transactionTime')
        account_story[a] = d
        
    return account_story

#Function that plot amount of healthy and fraudulent transactions per account 
def plot_transactions(d, cols = np.array(['blue','red'])):
    x = np.arange(0, len(d), 1)
    y = d['transactionAmount'].tolist()
    c = cols[d['Fraud'].tolist()]
    
    plt.figure(figsize = (20,10))
    plt.bar(x,y, color = c)
    plt.title('Healthy/Fraudulant Transactions for account ' + a)

#Plot the most important features in the model
def plot_feature_importance(importance,names,model_type):

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    fi_df = fi_df.head(10)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

#one hot encoding
def compile_dataset(trans, cat_predictors, num_predictors):
    data_formatted = pd.DataFrame(index=trans.index)
    for c in cat_predictors:
        data_formatted = pd.concat([data_formatted, pd.get_dummies(trans[c], prefix = c)], axis = 1)
    
    for n in num_predictors:
        
        data_formatted = pd.concat([data_formatted, trans[n]], axis = 1)
        
    return data_formatted

## Importing data from CSV
fraud_fear = pd.read_csv("///Users/hocine/Desktop/feature_space/data-new/labels_obf.csv")
trans = pd.read_csv("///Users/hocine/Desktop/feature_space/data-new/transactions_obf.csv")





"""
                       ++++ PART 1 ++++
"""
################################################
                                               #
            # Exploratory Analysis             #
################################################  


   
set_fraud_ids = set(fraud_fear.eventId)
trans['Fraud'] = trans.eventId.apply(lambda x: 1 if x in set_fraud_ids else 0)
trans['transactionTime'] = pd.to_datetime(trans['transactionTime'])


#Fraud proportions
fraud_prop = trans['Fraud'].value_counts(normalize = True)#%Fraud = 0.007%
fraud_nbrs = trans['Fraud'].value_counts(normalize = False)#Fraud = 875


#Nbr of accounts
nbr_acc = len(pd.unique(trans['accountNumber']))#766 unique accounts


#Nbr of unique fraudulant accounts
nbr_unique_acc = len(pd.unique(trans.query('Fraud == 1')['accountNumber']))#167

"""
Format Data: split data per accountNumber associated to a time series of spendings & fraud flags

"""
fraud_accounts = pd.unique(trans.query('Fraud == 1')['accountNumber'])
healthy_accounts = pd.unique(trans.query('Fraud == 0')['accountNumber'])

split_accounts = split_accounts_into_dict(trans)



#Visualize some account transaction history
for a in np.random.choice(fraud_accounts,50):
    d = split_accounts[a]
    print('plotting accounts ', a, ' with ', len(d), ' transactions ...')
    plot_transactions(d)




################################################
                                               #
         # One Hot Encoding  and Split         #
################################################     

cat_predictors = ['mcc', 'merchantCountry', 'posEntryMode']
num_predictors = ['transactionAmount', 'availableCash']

# x_nor is containing the normalized values of num_predictors, this is only used for LR
Feature_cat = trans[cat_predictors]
Feature_num = trans[num_predictors]
X_nor = preprocessing.StandardScaler().fit(Feature_num).transform(Feature_num)
X_nor = pd.DataFrame(X_nor, columns=num_predictors)

#Get dummies 
X_nor = pd.concat([X_nor,pd.get_dummies(trans['posEntryMode'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['mcc'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['merchantZip'])], axis=1)

y = trans['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X_nor, y, test_size = 0.3)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

        


################################################
                                               #
           # MODELS                            #
################################################  
#Logistic Regression***********************************************************
#class weight 1:10  Precision =  0.15 | Recall =  0.27
#class weight 1:20  Precision =  0.09 | Recall =  0.38

lr = LogisticRegression(class_weight = {0:1,
                                        1:20})
#class_weight better than undersampling cause it keeps ALL the given data
lr.fit(X_train, y_train)

y_hat_lr = lr.predict(X_test)
print('Logistic Regression \n',
      'Precision = ', precision_score(y_test, y_hat_lr), '\n',
      'Recall = ', recall_score(y_test, y_hat_lr), '\n')
#******************************************************************************



#Get dummies for  non normalized values
trans.set_index('eventId', inplace=True)
data_pred = compile_dataset(trans, cat_predictors, num_predictors)

X= data_pred 
y = trans['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)




#Random Forest***********************************************************
#balanced: Random Forest  Precision =  0.61| Recall =  0.38
#class weight 1:10  Precision =  0.57 | Recall =  0.42
#class weight 1:20 Precision = 0.47 | Recall = 0.43
#class weight 1:30 Precision = 0.47 | Recall = 0.39
rf = RandomForestClassifier(class_weight = {0:1,
                                            1:30})
rf.fit(X_train, y_train)

y_hat_rf = rf.predict(X_test)
print('Random Forest \n',
      'Precision = ', precision_score(y_test, y_hat_rf), '\n',
      'Recall = ', recall_score(y_test, y_hat_rf), '\n')

#feat_import = pd.DataFrame({'feature':rf.feature_names_in_,
 #                           'coef':rf.feature_importances_})

#Classes seem to be unseperable (completely). In Other words, there is BIG overlapping between Fraudulent 
#and non fraudulent (see plots of transactions per account number)
#One solution would be to predict according to specefic account Number (RNN for example...)
#******************************************************************************
plot_feature_importance(
    rf.feature_importances_,
    X_train.columns,
    'RANDOM FOREST')


#SVM with NON LINEAR KERNELRS (RBF kernel)*************************************

svm_rbf = svm.SVC(kernel='rbf',class_weight = {1:15})
svm_rbf.fit(X_train, y_train)
yhat_rbf = svm_rbf.predict(X_test)
print('SVM \n',
      'Precision = ', precision_score(y_test, yhat_rbf), '\n',
      'Recall = ', recall_score(y_test, yhat_rbf), '\n')
#******************************************************************************



#ADABOOST *******************************************************************
ada = ensemble.AdaBoostClassifier()
ada.fit(X_train,y_train)
yhat_ada = ada.predict(X_test)

print('AdaBoost \n',
      'Precision = ', precision_score(y_test, yhat_ada), '\n',
      'Recall = ', recall_score(y_test, yhat_ada), '\n')
#******************************************************************************


"""
                       ++++ PART 2 ++++
"""

#Models with new features and weight class

# amountMinusAvrg contains for each account the amount spent
# minus the avreage of the spent amount per account
dict_avrg_amount = trans.groupby('accountNumber')['transactionAmount'].mean().to_frame().to_dict()['transactionAmount']
trans['amountMinusAvrg'] = trans.apply(lambda x: dict_avrg_amount[x['accountNumber']] - x['transactionAmount'] , axis=1)
           

#avrgSpentWeekday
trans['transWeekDay'] = trans.transactionTime.apply(lambda x: "Day_%d"%pd.Timestamp.to_pydatetime(x).weekday())
avrgSpentWeekday = trans.groupby(['accountNumber', 'transWeekDay'])['transactionAmount'].mean()
dict_avrgSpentWeekday = avrgSpentWeekday.to_dict()

trans['amountMinusAvrgSpentWeekday'] =  trans.apply(
    lambda x: x['transactionAmount'] - dict_avrgSpentWeekday[tuple((x['accountNumber'], x['transWeekDay']))]  ,
    axis=1)



set_fraud_ids = set(fraud_fear.eventId)
trans['Fraud'] = [ (x in set_fraud_ids) for x in trans.index]
#trans.eventId.apply(lambda x: 1 if x in set_fraud_ids else 0)

trans['transactionTime'] = pd.to_datetime(trans['transactionTime'])
trans['Date'] = trans['transactionTime'].apply(lambda x: datetime.strftime(x, '%d-%m-%Y') )
trans['Time'] = trans['transactionTime'].apply(lambda x: datetime.strftime(x, '%H:%M') )
trans['hours'] = trans['transactionTime'].apply(lambda x: datetime.strftime(x, "H_%H"))



################################################
                                               #
         # One Hot Encoding and Split         #
################################################   

cat_predictors = ['mcc', 'merchantCountry', 'posEntryMode','hours']
num_predictors = ['availableCash','amountMinusAvrg','amountMinusAvrgSpentWeekday']

data_pred = compile_dataset(trans, cat_predictors, num_predictors)

# x_nor is containing the normalized values of num_predictors, this is only used for LR
# x_nor is containing the normalized values of num_predictors, this is only used for LR
Feature_cat = trans[cat_predictors]
Feature_num = trans[num_predictors]
X_nor = preprocessing.StandardScaler().fit(Feature_num).transform(Feature_num)

X_nor = pd.DataFrame(X_nor, columns=num_predictors, index = trans.index)
#Get dummies 
X_nor = pd.concat([X_nor,pd.get_dummies(trans['posEntryMode'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['mcc'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['merchantCountry'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['hours'])], axis=1)


y = trans['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X_nor, y, test_size = 0.2)



################################################
                                               #
           # MODELS                            #
################################################  

#Logistic Regression***********************************************************


lr = LogisticRegression(class_weight = {0:1,
                                        1:40})
#class_weight better than undersampling cause it keeps ALL the given data
lr.fit(X_train, y_train)

y_hat_lr = lr.predict(X_test)
y_hat_lr = lr.predict(X_test)
print('Logistic Regression \n',
      'Precision = ', precision_score(y_test, y_hat_lr), '\n',
      'Recall = ', recall_score(y_test, y_hat_lr), '\n')
#******************************************************************************



X= data_pred 
y = trans['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



#Random Forest***********************************************************
rf = RandomForestClassifier(class_weight = {0:1,
                                            1:40})
rf.fit(X_train, y_train)

y_hat_rf = rf.predict(X_test)
print('Random Forest \n',
      'Precision = ', precision_score(y_test, y_hat_rf), '\n',
      'Recall = ', recall_score(y_test, y_hat_rf), '\n')

feat_import = pd.DataFrame({'feature':rf.feature_names_in_,
                            'coef':rf.feature_importances_})
#******************************************************************************

plot_feature_importance(
    rf.feature_importances_,
    X_train.columns,
    'RANDOM FOREST')

#SVM with NON LINEAR KERNELRS (RBF kernel)*************************************

svm_rbf = svm.SVC(kernel='rbf',class_weight = {1:10})
svm_rbf.fit(X_train, y_train)
yhat_rbf = svm_rbf.predict(X_test)
print('SVM \n',
      'Precision = ', precision_score(y_test, yhat_rbf), '\n',
      'Recall = ', recall_score(y_test, yhat_rbf), '\n')
#******************************************************************************


#ADABOOST *********************************************************************

ada = ensemble.AdaBoostClassifier()
ada.fit(X_train,y_train)
yhat_ada = ada.predict(X_test)
print('AdaBoost \n',
      'Precision = ', precision_score(y_test, yhat_ada), '\n',
      'Recall = ', recall_score(y_test, yhat_ada), '\n')
#******************************************************************************



#Random Fores is having the best Recall and precision parameters


# It is calculating the different values of Recall and Precision when changing the values of
# the threshold
predict_prob = rf.predict_proba(X_test)
y_predict_prob =  pd.Series([x[1] for x in predict_prob], index=X_test.index)

for thresh in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
    y_predict =  y_predict_prob > thresh

    true_positives = len(
        set(y_test[y_test == 1].index).intersection(set(y_predict[y_predict == 1 ].index))
    )

    true_negatives = len(
        set(y_test[y_test == 0].index).intersection(set(y_predict[y_predict == 0 ].index))
    )

    false_positives = len(
        set(y_test[y_test == 0].index).intersection(set(y_predict[y_predict == 1 ].index))
    )

    false_negatives = len(
        set(y_test[y_test == 1].index).intersection(set(y_predict[y_predict == 0 ].index))
    )

    precision =  round(100*true_positives / (true_positives + false_positives), 0) 
    recall    =  round(100*true_positives / (true_positives + false_negatives), 0)

    print(f'{thresh} {true_positives} {false_positives} {true_negatives}  {false_negatives} {precision} {recall}')          
       

#Returning the set of the first 400 transactions that are the most likely to be fraudulant
y_predict_prob.sort_values(ascending=False, inplace=True)
set_most_fraudulant = set(y_predict_prob[:400].index)



"""
                       ++++ PART 3 ++++
"""

# Undersampling technique
# This will make the training set small
from imblearn.under_sampling import RandomUnderSampler

cat_predictors = ['mcc', 'merchantCountry', 'posEntryMode','hours']
num_predictors = ['transactionAmount', 'availableCash','amountMinusAvrg','amountMinusAvrgSpentWeekday']

data_pred = compile_dataset(trans, cat_predictors, num_predictors)

# x_nor is containing the normalized values of num_predictors, this is only used for LR
Feature_cat = trans[cat_predictors]
Feature_num = trans[num_predictors]
X_nor = preprocessing.StandardScaler().fit(Feature_num).transform(Feature_num)

X_nor = pd.DataFrame(X_nor, columns=num_predictors, index = trans.index)
#Get dummies 
X_nor = pd.concat([X_nor,pd.get_dummies(trans['posEntryMode'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['mcc'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['merchantCountry'])], axis=1)
X_nor = pd.concat([X_nor,pd.get_dummies(trans['hours'])], axis=1)


rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
X_res, y_res = rus.fit_resample(X_nor, y)
ax = y_res.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under sampling classes distribution")

X_res_train, X_res_test, y_res_train, y_res_test = train_test_split( X_res, y_res, test_size=0.2, random_state=4)



#Logistic Regression *********************************************************

lr = LogisticRegression(C=0.01, solver='liblinear').fit(X_res_train,y_res_train)
yhat_lr_res = lr.predict(X_res_test)
yhat_lr_prob = lr.predict_proba(X_res_test)
print('Logistic Regression \n',
      'Precision = ', precision_score(y_test, y_hat_lr), '\n',
      'Recall = ', recall_score(y_test, y_hat_lr), '\n')
#******************************************************************************


# Undersampling without using normalized X
X= data_pred 
y = trans['Fraud']
rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
X_res, y_res = rus.fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct='%.2f')


#Random Forest*****************************************************************
#balanced: Random Forest  Precision =  0.61| Recall =  0.38

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_hat_rf = rf.predict(X_test)
print('Random Forest \n',
      'Precision = ', precision_score(y_test, y_hat_rf), '\n',
      'Recall = ', recall_score(y_test, y_hat_rf), '\n')

feat_import = pd.DataFrame({'feature':rf.feature_names_in_,
                            'coef':rf.feature_importances_})
#******************************************************************************
plot_feature_importance(
    rf.feature_importances_,
    X_train.columns,
    'RANDOM FOREST')

#SVM with NON LINEAR KERNELRS (RBF kernel)*************************************

svm_rbf = svm.SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
yhat_rbf = svm_rbf.predict(X_test)
print('SVM \n',
      'Precision = ', precision_score(y_test, yhat_rbf), '\n',
      'Recall = ', recall_score(y_test, yhat_rbf), '\n')
#******************************************************************************



#ADABOOST *********************************************************************

ada = ensemble.AdaBoostClassifier()
ada.fit(X_train,y_train)
yhat_ada = ada.predict(X_test)
print('AdaBoost \n',
      'Precision = ', precision_score(y_test, yhat_ada), '\n',
      'Recall = ', recall_score(y_test, yhat_ada), '\n')
#******************************************************************************
