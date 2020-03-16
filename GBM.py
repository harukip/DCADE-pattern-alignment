#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from xgboost import XGBClassifier
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[ ]:


def GBM_predict(data_name, model_name):
    import pickle
    X = pd.read_csv(os.path.join(".", "GBM", data_name+".csv", index_col=0))
    ignore_cols = ['encode', 'rep_time', 'ign_len']
    X.drop(['label'], axis=1, inplace=True)
    with open(os.path.join(".", "GBM", model_name+".dat", 'rb')) as f:
        model = pickle.load(f)
    numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64'] and cname not in ignore_cols]
    X_test = X[numeric_cols].copy()
    y_test = model.predict(X_test)
    X['good_encode'] = y_test
    isgood = X['good_encode'] == 1
    filter_X = X[isgood]
    best = filter_X.sort_values(by='similarity', ascending=False).iloc[0]
    return int(best['encode']), int(best['ign_len'])


# In[ ]:


def save_model(model, model_name):
    import pickle
    with open(os.path.join(".", "GBM", model_name+".dat", 'wb')) as f:
        pickle.dump(model, f)


# In[ ]:


if __name__ == "__main__":
    X = pd.read_csv("./GBM/train_N17.csv", index_col=0)
    y = X.label
    X.drop(['label'], axis=1, inplace=True)
    X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    ignore_cols = ['encode', 'rep_time', 'ign_len']
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64'] and cname not in ignore_cols]   
    X_train = X_train_full[numeric_cols].copy()
    X_valid = X_val_full[numeric_cols].copy()
    
    model = XGBClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    
    print("MAE:", mean_absolute_error(y_true=y_val, y_pred=y_pred))
    
    prediction = model.predict(X[numeric_cols])
    X['good_encode'] = prediction
    isgood = X['good_encode'] == 1
    filter_X = X[isgood]
    print(filter_X.sort_values(by='similarity', ascending=False).iloc[0])


# In[ ]:




