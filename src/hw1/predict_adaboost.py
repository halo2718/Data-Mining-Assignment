import numpy as np
import pandas as pd
from models.decision_tree import DecisionTreeClassifier
from models.adaboost import AdaBoostClassifier
from utililities.io import get_dataframe
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve, f1_score
df = get_dataframe("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv", "drop")

shares = df.iloc[:, -1]
popular   = shares >= 1400
inpopular = shares < 1400
df.loc[popular, ' shares']   = 1
df.loc[inpopular, ' shares'] = 0

def get_split(df_d, proportion=0.8):
    train_set   = df_d.iloc[:int(df_d.shape[0] * proportion), :-1]
    train_label = df_d.iloc[:int(df_d.shape[0] * proportion), -1]
    test_set    = df_d.iloc[int(df_d.shape[0] * proportion):, :-1]
    test_label  = df_d.iloc[int(df_d.shape[0] * proportion):, -1]
    return train_set, train_label, test_set, test_label

train_set, train_label, test_set, test_label = get_split(df, proportion=0.8)

clf = AdaBoostClassifier().fit(train_set, train_label)
score = clf.score(test_set, test_label) 
result = clf.predict(test_set)
f1 = f1_score(test_label, result)
print(score)
print(f1)
print(roc_auc_score(test_label, result))


