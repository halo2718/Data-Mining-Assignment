import pandas as pd

# Get the dataframe given a path to .csv file.
def get_dataframe(path, condition="drop"):
    '''
    Input:
        path:       (str)   The path of target .csv file, e.g. data\OnlineNewsPopularity\OnlineNewsPopularity.csv.
        condition:  (str)   The type of preprocessing to decide how to preprocess the dataframe.
    Output:
        df:         (pandas.Dataframe)  The dataframe of corresponding .csv file.
    '''
    df = pd.read_csv(path)
    if condition == "drop":
        df.drop(labels=['url', ' timedelta'], axis = 1, inplace=True)
    return df
        