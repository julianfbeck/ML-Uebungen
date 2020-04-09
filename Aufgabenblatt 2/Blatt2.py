import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import pandas as pd
import itertools

DATA_FILE = './Data/original_titanic.csv'
df = pd.read_csv(DATA_FILE, header=0)
from functools import reduce

product = reduce((lambda x, y: x + y), df.Survived)