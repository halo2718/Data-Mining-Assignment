from matplotlib import test
import numpy as np
import seaborn as sns
from utils import read_csv, get_title_and_content, remove_url
import pandas as pd

from models.linear_regressor import LinearRegressor


df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)

# label = df.loc[:," shares"].to_numpy()
def get_split(df_d, proportion=0.8):
    train_set   = df_d.iloc[:int(df_d.shape[0] * proportion), :-1]
    train_label = df_d.iloc[:int(df_d.shape[0] * proportion), -1]
    test_set    = df_d.iloc[int(df_d.shape[0] * proportion):, :-1]
    test_label  = df_d.iloc[int(df_d.shape[0] * proportion):, -1]
    return train_set, train_label, test_set, test_label

train_set, train_label, test_set, test_label = get_split(df, proportion=0.8)

# std_normalized_train_set_df       =   (df - df.mean()) / df.std()
data     =   (train_set - train_set.min()) / (train_set.max() - train_set.min())
label    =   (train_label - train_label.min()) / (train_label.max() - train_label.min())
data  = data.to_numpy()
label = label.to_numpy()

test_data  = (test_set - train_set.min()) / (train_set.max() - train_set.min())
test_label = (test_label - train_label.min()) / (train_label.max() - train_label.min())

mid_val = (1400 - train_label.min()) / (train_label.max() - train_label.min())

print(data.shape)
print(label.shape)

def metric(pred, gt):
    delta = (pred - mid_val) * (gt.to_numpy() - mid_val)
    a = np.where(delta >= 0, 1, 0)
    print(np.sum(a) / a.shape[0])

lr = LinearRegressor(data, label)
lr.solve()
pred = lr.predict(test_data.to_numpy())

metric(pred, test_label)

test_delta = (test_label.to_numpy() - mid_val)
a = np.where(test_delta >= 0, 1, 0)
print(np.sum(a) / a.shape[0])




