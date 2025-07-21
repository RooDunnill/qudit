import numpy as np
from sklearn.preprocessing import MinMaxScaler


data = []
data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)
data = data[1:, :]
remove_cols = [2,3,5,6,7,8,9]
stripped_data = np.delete(data, remove_cols, axis=1)

data_size = len(data)

testing_percentage = 0.20

tdp_range = int(np.floor(testing_percentage * data_size / 2))
testing_data_points = tdp_range * 2

scaler = MinMaxScaler()

def encode(data):
    for i in range(len(data)):
        if data[i,0] == 'F':
            data[i,0] = 1
        else:
            data[i,0] = 0
        if data[i,-1] == 'Y':
            data[i,-1] = 1
        else:
            data[i,-1] = 0
    return data

encoded_stripped_data = encode(stripped_data)
normalized_data = scaler.fit_transform(encoded_stripped_data)

pos_testing_data = normalized_data[-tdp_range:,:4]
neg_testing_data = normalized_data[:tdp_range,:4]
normalized_data = normalized_data[tdp_range:-tdp_range, :]

np.random.shuffle(normalized_data)

results = normalized_data[:,4:]
normalized_data = np.delete(normalized_data, 4, axis=1)


def results_encode(data):
    for i in range(len(data)):
        if data[i,0] == 'Y':
            data[i,0] = 1
        else:
            data[i,0] = 0
    return data




print(normalized_data)
print(results)
print(pos_testing_data)
print(neg_testing_data)
print(testing_data_points)