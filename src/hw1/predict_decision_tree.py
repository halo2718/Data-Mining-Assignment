import re
import numpy as np
import pandas as pd
from models.decision_tree import DecisionTreeClassifier
from utililities.io import get_dataframe

from sklearn.tree import plot_tree ,export_graphviz
import graphviz

# TODO: 归一化，直接把MLP的代码复制进来就好。这个模型影响不大

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

clf = DecisionTreeClassifier(criterion="entropy", max_depth=12).fit(train_set, train_label)
result = clf.predict(test_set)
score = clf.score(test_set, test_label) 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve, f1_score
import matplotlib.pyplot as plt

# plot_tree(clf, 
#           filled = True, 
#           rounded = True)
 
# plt.savefig('tree_visualization.png')

# print("here")
# print(roc_auc_score(test_label, result))
# plot_roc_curve(clf, test_set, test_label)
# plt.show()

f1 = f1_score(test_label, result)
print(f1)


dot_data = export_graphviz(clf, feature_names=df.columns[:-1], class_names = ['low','high'], out_file=None, special_characters=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render("OnlineNews")