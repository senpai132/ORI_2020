import pandas as pd
import numpy as np
from sklearn.preprocessing import  scale, normalize


def read_data():
    data = pd.read_csv('credit_card_data.csv')
    data.drop(data.columns[0], axis = 1, inplace = True)
    getRidOfInvalidPostions(data)

    return data

def getRawData():
    return read_data()

def getNormalisedData():
    data = pd.DataFrame(normalize(read_data()))
    return data

def getScaledData():
    data = pd.DataFrame(scale(read_data()))
    return data

def getCustomlyNormalisedData():
    pass

def getRidOfInvalidPostions(data):
    invalid_positions = np.where(np.isnan(data))

    for i in range (len(invalid_positions[0])):
        data.iloc[invalid_positions[0][i],invalid_positions[1][i]] = 0

#read_data()