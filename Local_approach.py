import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

def compile_dataset(data, cat_predictors, num_predictors):
    data_formatted = pd.DataFrame()
    for c in cat_predictors:
        data_formatted = pd.concat([data_formatted, pd.get_dummies(data[c], prefix = c)], axis = 1)
    
    for n in num_predictors:
        
        data_formatted = pd.concat([data_formatted, data[n]], axis = 1)
        
    return data_formatted

def train_on_fraud_accounts(data, data_pred, test_size = 0.3, min_nbr_samples = 5):

    fraud_accounts = pd.unique(data.query('Fraud == 1')['accountNumber'])
    
    fraud_accounts_models = dict()
    
    fraud_accounts_ids = []
    fraud_accounts_nbr_transacs = []
    fraud_accounts_nbr_fraud_transacs = []
    fraud_accounts_precisions = []
    fraud_accounts_recall = []
    fraud_accounts_acc = []
    
    for a in fraud_accounts:
        print('Training on accountNumber ' + a +'...')
        
        #get relevent data to accountNumber a
        d = data_pred.loc[data.query('accountNumber == @a').index]
        
        if len(d) >= min_nbr_samples:
            fraud_accounts_ids.append(a)
            f = data.query('accountNumber == @a')['Fraud'].tolist()
            fraud_accounts_nbr_transacs.append(len(d))
            fraud_accounts_nbr_fraud_transacs.append(np.array(f).sum())
            
            #split it into train and test
            X_train, X_test, y_train, y_test = train_test_split(d, f, test_size = test_size)
        
            #train model on it
            model = RandomForestClassifier()
            
            model.fit(X_train, y_train)
            fraud_accounts_models[a] = model
            
            #Test and metrics computation
            y_pred = model.predict(X_test)
            
            fraud_accounts_precisions.append(precision_score(y_test, y_pred))
            fraud_accounts_recall.append(recall_score(y_test, y_pred))
            fraud_accounts_acc.append(accuracy_score(y_test, y_pred))
        
    fraud_accounts_perfs = pd.DataFrame({'accountNumber':fraud_accounts_ids,
                                         'nbr_transactions': fraud_accounts_nbr_transacs,
                                         'nbr_fraud_transactions' : fraud_accounts_nbr_fraud_transacs,
                                         'precision': fraud_accounts_precisions,
                                         'recall': fraud_accounts_recall,
                                         'accuracy': fraud_accounts_acc})
    
    return fraud_accounts_perfs, fraud_accounts_models
    





#Load and normalize numerical data
data_path = os.path.join('FeaturesSpace_Homework', 'data', )
fraud_fear = pd.read_csv(os.path.join(data_path, 'labels_obf.csv'))
data = pd.read_csv(os.path.join(data_path, 'transactions_obf.csv'))
data[['transactionAmount', 'availableCash']] = StandardScaler().fit_transform(data[['transactionAmount', 'availableCash']])


#Add fraud column
set_fraud_ids = set(fraud_fear.eventId)
data['Fraud'] = data.eventId.apply(lambda x: 1 if x in set_fraud_ids else 0)
data['transactionTime'] = pd.to_datetime(data['transactionTime'])


cat_predictors = ['mcc', 'merchantCountry', 'posEntryMode']
num_predictors = ['transactionAmount', 'availableCash']
data_pred = compile_dataset(data, cat_predictors, num_predictors)


fraud_accounts_perfs, fraud_accounts_models = train_on_fraud_accounts(data, data_pred)
fraud_accounts_perfs.to_csv('featurespace_accountNumber_model_RF_perfs.csv', index = False)

"""
*Global Approach:
    All accounts share the SAME model. Low Performances 
* Local Approach:
    Model by numberAccount provide better performances (recall & precision)
    cause they take into account the specific habits of the account holders.
    However it requires fraudulent & non fraudulent transactions to be trained
"""