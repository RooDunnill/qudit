import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from qiskit.circuit.library import ZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

data = []                                                                 #creates an array for the dataset from the csv
data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)     #imports the data from the csv
data = data[1:, :]                                                        #strips the titles
remove_cols = [1,2,3,4,5,6,7,8,9]                                         #removes the features that aren't needed
stripped_data = np.delete(data, remove_cols, axis=1)                      #strips them from the array
features = 11 - len(remove_cols)                                          #finds the amount of features used
data_size = len(data)                                                     #computes the size of the data set
shots = 1000
testing_percentage = 2                                                    #the size of the testing set
training_percentage = 4                                                   #the size of the training set
                         
scaler = MinMaxScaler()                               

def encode(data):                                                         #encodes any strings in the set into 0s or 1s
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


feature_map = ZFeatureMap(feature_dimension=features, reps=1).decompose()
feature_map_inv = ZFeatureMap(feature_dimension=features, reps=1, parameter_prefix="y").decompose().inverse()

circ = QuantumCircuit(features)
circ.compose(feature_map, inplace=True)
circ.compose(feature_map_inv, inplace=True)
circ.measure_all()
bound_circ = circ.assign_parameters(np.append(training_data[0], training_data[1]))
print(circ.draw())
sim = AerSimulator()

result = sim.run(bound_circ).result()
counts = result.get_counts(bound_circ)

def square_kernel_mat(training_data):
    mat_dim = len(training_data)
    kernel_mat = np.zeros((mat_dim, mat_dim))
    sim = AerSimulator()
    for i in range(mat_dim):
        for j in range(i, mat_dim):
            if j == i:
                kernel_mat[i,i] += 1
            else:
                bound_circ = circ.assign_parameters(np.append(training_data[i], training_data[j]))
                result = sim.run(bound_circ, shots=shots).result()
                counts = result.get_counts(bound_circ)
                kernel_mat[i,j] = counts['0' * features] / shots
                kernel_mat[j,i] = counts['0' * features] / shots
    return kernel_mat

def testing_kernel_mat(training_data, testing_data):
    y_mat_dim = len(training_data)
    x_mat_dim = len(testing_data)
    kernel_mat = np.zeros((x_mat_dim, y_mat_dim))
    sim = AerSimulator()
    for i in range(x_mat_dim):
        for j in range(y_mat_dim):
                bound_circ = circ.assign_parameters(np.append(testing_data[i], training_data[j]))
                result = sim.run(bound_circ, shots=shots).result()
                counts = result.get_counts(bound_circ)
                kernel_mat[i,j] = counts['0' * features] / shots
    return kernel_mat













print(f"Machine Learning Results:")

print(training_data)
print(testing_data)
print(training_results)
print(testing_results)

print(feature_map.draw())
print(feature_map_inv.draw())
print(circ.draw())
print(counts)
print(square_kernel_mat(training_data))
print(testing_kernel_mat(training_data, testing_data))