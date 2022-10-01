import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../../data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
print(df.describe())

temp_data = df[df[' shares'] <= 100000]

print(temp_data)
fig, axes = plt.subplots(figsize=(10,10))
sns.scatterplot(x=' title_sentiment_polarity', y=' shares',hue=' n_tokens_title', data=temp_data, ax=axes)
plt.show()

'''
数据分析：
0. 通过drop_na(), drop_duplicate()等方法可知，原始数据质量较好，没有空数据or重复数据。
1. 不同属性的数据尺度不同，例如n_tokens_content的取值范围在0到8474之间而title_sentiment_polarity仅在-1到1之间；为此，在应用到模型之前应当进行归一化。
2. 
'''