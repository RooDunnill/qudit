import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import qiskit_superstaq as qss
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from qiskit.circuit import ParameterVector

data = []                                                                 #creates an array for the dataset from the csv
data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)     #imports the data from the csv
data = data[1:, :]                                                        #strips the titles
remove_cols = [2,3,4,9]                                         #removes the features that aren't needed
stripped_data = np.delete(data, remove_cols, axis=1)                      #strips them from the array
features = 11 - len(remove_cols)                                          #finds the amount of features used
data_size = len(data)                                                     #computes the size of the data set
shots = 1000
testing_percentage = 4                                                    #the size of the testing set
training_percentage = 4                                                   #the size of the training set
                         
scaler = MinMaxScaler()                               

def encode(data):                                                         #encodes any strings in the set into 0s or 1s
    for i in range(len(data)):                                            #iterates through each data point
        if data[i,0] == 'F':                                              #sorts the genders into binary data
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
#normalized_data[:,1:-1] = normalized_data[:, 1:-1] * 2 * np.pi


results = normalized_data[:,features:]

data = normalized_data[:,:features]

training_data, testing_data, training_results, testing_results = train_test_split(data, results, shuffle=True, test_size=testing_percentage, train_size=training_percentage)

training_results = training_results.T[0]
testing_results = testing_results.T[0]

theta = np.pi

def custom_feature_map(features, reps=1):
    x= ParameterVector("x", length=features)

    fqc = QuantumCircuit(features)

    for _ in range(reps):
        for i in range(features):
            fqc.h(i)
            fqc.rz(x[i], i)

        for i in range(features - 1):
            fqc.crx(theta, i, i + 1)
        fqc.crx(theta, features - 1, 0)

    return fqc

feature_map = custom_feature_map(features=3, reps=1)
feature_map_inv = custom_feature_map(features=3, reps=1).inverse()



feature_map = ZZFeatureMap(feature_dimension=features, reps=1, entanglement="linear").decompose()
feature_map_inv = ZZFeatureMap(feature_dimension=features, reps=1, parameter_prefix="y", entanglement="linear").decompose().inverse()

circ = QuantumCircuit(features)
circ.compose(feature_map, inplace=True)
circ.compose(feature_map_inv, inplace=True)
circ.measure_all()
bound_circ = circ.assign_parameters(np.append(training_data[0], training_data[1]))
sim = AerSimulator()

result = sim.run(bound_circ).result()
counts = result.get_counts(bound_circ)

def square_kernel_mat(training_data):
    mat_dim = len(training_data)
    kernel_mat = np.zeros((mat_dim, mat_dim))
    for i in range(mat_dim):
        for j in range(i, mat_dim):
            if j == i:
                kernel_mat[i,i] += 1
            else:
                bound_circ = circ.assign_parameters(np.append(training_data[i], training_data[j]))
                result = sim.run(bound_circ, shots=shots).result()
                counts = result.get_counts(bound_circ)
                kernel_mat[i,j] = counts.get('0' * features, 0) / shots
                kernel_mat[j,i] = counts.get('0' * features, 0) / shots
    return kernel_mat

def testing_kernel_mat(training_data, testing_data):
    y_mat_dim = len(training_data)
    x_mat_dim = len(testing_data)
    kernel_mat = np.zeros((x_mat_dim, y_mat_dim))
    for i in range(x_mat_dim):
        for j in range(y_mat_dim):
                bound_circ = circ.assign_parameters(np.append(testing_data[i], training_data[j]))
                result = sim.run(bound_circ, shots=shots).result()
                counts = result.get_counts(bound_circ)
                kernel_mat[i,j] = counts.get('0' * features, 0) / shots
    return kernel_mat

training_kernel = square_kernel_mat(training_data)
testing_kernel = testing_kernel_mat(training_data, testing_data)

svc = svm.SVC(kernel="precomputed")
svc.fit(training_kernel, training_results)
y_predict = svc.predict(testing_kernel)


print(f"Machine Learning Results:")
print(y_predict)
print(testing_results)
print(f"Balanced Accuracy Score: {balanced_accuracy_score(testing_results, y_predict)}")
print(f"f1 Score: {f1_score(testing_results, y_predict, average="weighted")}")
# plt.figure()
plt.imshow(training_kernel, cmap="hot")
plt.show()
