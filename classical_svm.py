import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

data = []
data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)
data = data[1:, :]
remove_cols = [2,3,5,6,7,8,9]
stripped_data = np.delete(data, remove_cols, axis=1)
features = 11 - len(remove_cols)

data_size = len(data)

testing_percentage = 0.8
training_percentage = 0.2
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

results = normalized_data[:,features:]

data = normalized_data[:,:features]

training_data, testing_data, training_results, testing_results = train_test_split(data, results, shuffle=True, test_size=testing_percentage, train_size=training_percentage)

training_results = training_results.T[0]
testing_results = testing_results.T[0]

print(f"Machine Learning Results:")
print(f"Training Data size: {len(training_data)}")
clf = svm.SVC(kernel = "linear")
print(training_data)
print(testing_data)
print(training_results)
print(testing_results)
clf.fit(training_data, training_results)
print(clf.predict(testing_data))
print(clf.score(training_data, training_results))
print(accuracy_score(testing_results, clf.predict(testing_data)))