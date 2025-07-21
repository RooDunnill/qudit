from sklearn.preprocessing import MinMaxScaler
import numpy as np


def encode_ds1(data, remove_cols):                              #encodes any strings in the set into 0s or 1s
    data = data[1:, :]                                                        #strips the titles
    for i in range(len(data)):                                            #iterates through each data point
        if data[i,0] == 'F':                                              #sorts the genders into binary data
            data[i,0] = 1
        else:
            data[i,0] = 0
        if data[i,-1] == 'Y':                                             #encdoes the results into 0s and 1s
            data[i,-1] = 1
        else:
            data[i,-1] = 0
    scaler = MinMaxScaler() 
    stripped_data = np.delete(data, remove_cols, axis=1)                     #strips them from the array
    data = scaler.fit_transform(stripped_data)            #normalises the data between 0 and 1
    return data

def encode_ds2(data, remove_cols):                              #encodes any strings in the set into 0s or 1s
    data = data[1:, :]                                                        #strips the titles
    scaler = MinMaxScaler() 
    stripped_data = np.delete(data, remove_cols, axis=1)                     #strips them from the array
    data = scaler.fit_transform(stripped_data)            #normalises the data between 0 and 1
    return data

