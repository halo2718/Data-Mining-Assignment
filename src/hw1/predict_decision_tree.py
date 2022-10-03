from matplotlib import test
import numpy as np
import seaborn as sns
from utils import read_csv, get_title_and_content, remove_url
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from models.linear_regressor import LinearRegressor
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)

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

clf = DecisionTreeClassifier(criterion="entropy").fit(train_set, train_label)
score = clf.score(test_set, test_label) 
print(score)




