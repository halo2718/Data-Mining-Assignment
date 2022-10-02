import numpy as np
import seaborn as sns
from utils import read_csv, get_title_and_content, remove_url
import pandas as pd

from models.linear_regressor import LinearRegressor


df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)

# label = df.loc[:," shares"].to_numpy()
std_normalized_df       =   (df - df.mean()) / df.std()
minmax_normalized_df    =   (df - df.min()) / (df.max() - df.min())
print((1400 - df.min()) / (df.max() - df.min()))
data = minmax_normalized_df.iloc[:, :-1]
print(data)
