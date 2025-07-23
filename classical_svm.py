import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import csv


def main():
    data = []
    data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)
    data = data[1:, :]
    remove_cols = [2,3,4,6,7,8,9]
    stripped_data = np.delete(data, remove_cols, axis=1)
    features = 11 - len(remove_cols)

    data_size = len(data)

    testing_percentage = 160
    training_percentage = 40
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
    clf = svm.SVC(kernel = "linear")
    print(f"Training Size: {len(training_data)}")
    print(f"Testing Size: {len(testing_data)}")
    clf.fit(training_data, training_results)


    classical_ml_train_accuracy = clf.score(training_data, training_results)                 #computes the accuracy of the algorithm in training
    classical_ml_test_accuracy = accuracy_score(testing_results, clf.predict(testing_data))



    with open('accuracy_data_classical.csv', 'a', newline='') as f:    #saves the relevent daccuracy data
        writer = csv.writer(f)
        writer.writerow([classical_ml_train_accuracy, classical_ml_test_accuracy])

while True:
    main()