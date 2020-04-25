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


def normalize_colummn(feature, column):
    mean = feature.mean().loc[column]
    std = feature.std().loc[column]
    for index in range(len(feature)):
        feature.loc[index, column] = (feature.loc[index, column]- mean) / std

    
def normalize(df):
    new_dataFrame = df.copy()
    normalize_colummn(new_dataFrame, "Age")


    return None # TODO implement

df_train_norm = normalize(df_train) # TODO : implement
df_test_norm =  None # TODO : implement

df.head()
