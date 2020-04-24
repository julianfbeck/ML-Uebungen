import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import random
DATA_FILE = './Data/original_titanic.csv'
# TODO : implement
df = pd.read_csv(DATA_FILE, header=0)

def prepareData(df):
    df.loc[(((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==1)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==1), "Age"].mean()
    df.loc[(((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==2)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==2), "Age"].mean()
    df.loc[(((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==3)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 0)) & (df.Pclass==3), "Age"].mean()
    
    df.loc[(((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==1)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==1), "Age"].mean()
    df.loc[(((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==2)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==2), "Age"].mean()
    df.loc[(((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==3)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "male")   &(df.Survived == 1)) & (df.Pclass==3), "Age"].mean()
    
    
    df.loc[(((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==1)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==1), "Age"].mean()
    df.loc[(((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==2)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==2), "Age"].mean()
    df.loc[(((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==3)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 0)) & (df.Pclass==3), "Age"].mean()
    
    df.loc[(((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==1)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==1), "Age"].mean()
    df.loc[(((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==2)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==2), "Age"].mean()
    df.loc[(((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==3)) & df.Age.isnull(),"Age"] =  df.loc[((df.Sex == "female") &(df.Survived == 1)) & (df.Pclass==3), "Age"].mean()
    return df
    
df = prepareData(df)
df_shuffled = df.sample(frac=1) 
df_train = df_shuffled[:int(len(df_shuffled)* 0.8)] 
df_test  =  df_shuffled[int(len(df_shuffled)* 0.8):] 

def predict(row):
    # TODO: implement
    found_rows = df_train[(df_train.Sex == row.Sex) & (df_train.Survived == row.Survived) & (df_train.Age == row.Age) & (df_train.Pclass == row.Pclass)]
    if len(found_rows) > 0:
        randomRow = found_rows.sample(n=1)
        return randomRow.iloc[0].Survived
    else:
        return random.randint(0, 1)


def getAcc():
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index, row in df_test.iterrows():
        predicted = predict(row)
        actual = row["Survived"]

        if actual:
            if predicted:
                tp += 1
            else:
                tn += 1
        else:
            if predicted:
                fp += 1
            else:
                fn += 1

    
    return (tp + tn) / (tp + tn + fp +fn)

print(getAcc())
#predict(df_test[1])
