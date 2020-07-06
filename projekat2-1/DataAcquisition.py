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
    data = read_data()
    maxi = data['BALANCE'].max()
    data['BALANCE'] = data['BALANCE'].div(maxi)

    maxi = data['PURCHASES'].max()
    data['PURCHASES'] = data['PURCHASES'].div(maxi)

    maxi = data['ONEOFF_PURCHASES'].max()
    data['ONEOFF_PURCHASES'] = data['ONEOFF_PURCHASES'].div(maxi)

    maxi = data['INSTALLMENTS_PURCHASES'].max()
    data['INSTALLMENTS_PURCHASES'] = data['INSTALLMENTS_PURCHASES'].div(maxi)

    maxi = data['CASH_ADVANCE'].max()
    data['CASH_ADVANCE'] = data['CASH_ADVANCE'].div(maxi)

    maxi = data['CASH_ADVANCE_TRX'].max()
    data['CASH_ADVANCE_TRX'] = data['CASH_ADVANCE_TRX'].div(maxi)

    maxi = data['PURCHASES_TRX'].max()
    data['PURCHASES_TRX'] = data['PURCHASES_TRX'].div(maxi)

    maxi = data['CREDIT_LIMIT'].max()
    data['CREDIT_LIMIT'] = data['CREDIT_LIMIT'].div(maxi)

    maxi = data['PAYMENTS'].max()
    data['PAYMENTS'] = data['PAYMENTS'].div(maxi)

    maxi = data['MINIMUM_PAYMENTS'].max()
    data['MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].div(maxi)

    maxi = data['TENURE'].max()
    data['TENURE'] = data['TENURE'].div(maxi)

    return data

def getRidOfInvalidPostions(data):
    invalid_positions = np.where(np.isnan(data))

    for i in range (len(invalid_positions[0])):
        data.iloc[invalid_positions[0][i],invalid_positions[1][i]] = 0

#read_data()