import pandas as pd

def load_csv(file_path):
    df = pd.read_csv(file_path, nrows=50)
    df = df.astype(str)
    df = df.where(pd.notnull(df), None)
    df = df.map(lambda x: None if pd.isna(x) else x)
    # pd.set_option('display.max_columns', None)
    # print(df.head)
    print("read completed")
    return df

# load_csv('ctgov_split/ctgov_0.csv')