import pandas as pd

df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)

df.describe()
print(df.describe())