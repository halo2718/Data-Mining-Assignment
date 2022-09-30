from lib2to3.pgen2.token import N_TOKENS
import numpy as np
import seaborn as sns
from utils import read_csv, get_title_and_content, remove_url
import pandas as pd

df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
# df.iloc[0] = df.iloc[0].str.strip()

print(df.shape)
df.dropna()
print(df.shape)
df.drop_duplicates()
print(df.shape)
print(df.describe())
sns.relplot(data=df, x=' n_tokens_title', y=' shares')
