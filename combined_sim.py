import os
import sys
currentdir = os.path.abspath("")
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import qiskit_superstaq as qss
from sklearn.metrics import balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from config import API_KEY
from encoding_funcs import encode_ds1, encode_ds2
import csv


data_set = 0
data = [] 


if data_set == 0:
    data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)     #imports the data from the csv
    remove_cols = [2,3,4,5,6,7,8,9]                                         #removes the features that aren't needed
    features = 11 - len(remove_cols)                                          #finds the amount of features used
    normalized_data = encode_ds1(data, remove_cols)
else:
    data = np.loadtxt("diabetes_dataset_2.csv", delimiter=",", dtype=str)
    remove_cols = [1,2,3,4]
    features = 8 - len(remove_cols)
    normalized_data = encode_ds2(data, remove_cols)



data_size = len(normalized_data)                                                     #computes the size of the data set
shots = 1000
testing_size = 20                                                    #the size of the testing set
training_size = 80                                                   #the size of the training set        
run_superstaq = 0


                              
results = normalized_data[:,features:]
data = normalized_data[:,:features]

training_data, testing_data, training_results, testing_results = train_test_split(data, results, shuffle=True, test_size=testing_size, train_size=training_size)

training_results = training_results.T[0]
testing_results = testing_results.T[0]



feature_map = ZZFeatureMap(feature_dimension=features, reps=1, entanglement="linear").decompose()
feature_map_inv = ZZFeatureMap(feature_dimension=features, reps=1, parameter_prefix="y", entanglement="linear").decompose().inverse()

circ = QuantumCircuit(features)
circ.compose(feature_map, inplace=True)
circ.compose(feature_map_inv, inplace=True)
circ.measure_all()
bound_circ = circ.assign_parameters(np.append(training_data[0], training_data[1]))

provider = qss.SuperstaqProvider(api_key=API_KEY)
superstaq_sim = provider.get_backend("cq_sqale_qpu")
sim = AerSimulator()


def train_k_mat_aer(training_data):
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

def test_k_mat_aer(training_data, testing_data):
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


def train_k_mat_superstaq(training_data):
    mat_dim = len(training_data)
    kernel_mat = np.zeros((mat_dim, mat_dim))
    for i in range(mat_dim):
        for j in range(i, mat_dim):
            if j == i:
                kernel_mat[i,i] += 1
            else:
                bound_circ = circ.assign_parameters(np.append(training_data[i], training_data[j]))
                result = superstaq_sim.run(bound_circ, shots=shots, method="sim").result()
                counts = result.get_counts()
                kernel_mat[i,j] = counts.get('0' * features, 0) / shots
                kernel_mat[j,i] = counts.get('0' * features, 0) / shots
    return kernel_mat

def test_k_mat_superstaq(training_data, testing_data):
    y_mat_dim = len(training_data)
    x_mat_dim = len(testing_data)
    kernel_mat = np.zeros((x_mat_dim, y_mat_dim))
    provider = qss.SuperstaqProvider(api_key=API_KEY)
    sim = provider.get_backend("cq_sqale_qpu")
    for i in range(x_mat_dim):
        for j in range(y_mat_dim):
                bound_circ = circ.assign_parameters(np.append(testing_data[i], training_data[j]))
                result = sim.run(bound_circ, shots=shots, method="sim").result()
                counts = result.get_counts()
                kernel_mat[i,j] = counts.get('0' * features, 0) / shots
    return kernel_mat



aer_training_kernel = train_k_mat_aer(training_data)
aer_testing_kernel = test_k_mat_aer(training_data, testing_data)

svc_aer = svm.SVC(kernel="precomputed")
svc_aer.fit(aer_training_kernel, training_results)
aer_y_predict = svc_aer.predict(aer_testing_kernel)


clf = svm.SVC(kernel = "linear")                  #classical simulator kernel
clf.fit(training_data, training_results)          #trains the kernel with the training data

classical_ml_train_accuracy = clf.score(training_data, training_results)
classical_ml_test_accuracy = accuracy_score(testing_results, clf.predict(testing_data))
aer_balanced_accuracy = balanced_accuracy_score(testing_results, aer_y_predict)
aer_f1_score = f1_score(testing_results, aer_y_predict, average="weighted")


print(f"Machine Learning Results of data set {data_set+1}".center(100, "="))
print(f"Classical Machine Learning Training Accuracy: {classical_ml_train_accuracy}")
print(f"Classical Machine Learning Testing Accuracy: {classical_ml_test_accuracy}")

print(f"Aer Simulator Balanced Accuracy Score: {aer_balanced_accuracy}")
print(f"Aer Simulator F1 Score: {aer_f1_score}")

if run_superstaq == 1:
    superstaq_training_kernel = train_k_mat_superstaq(training_data)
    superstaq_testing_kernel = test_k_mat_superstaq(training_data, testing_data)
    svc_superstaq = svm.SVC(kernel="precomputed")
    svc_superstaq.fit(superstaq_training_kernel, training_results)
    superstaq_y_predict = svc_superstaq(superstaq_testing_kernel)
    print(f"Superstaq Simulator Balanced Accuracy Score: {balanced_accuracy_score(testing_results, superstaq_y_predict)}")
    print(f"Superstaq Simulator F1 Score: {f1_score(testing_results, superstaq_y_predict, average="weighted")}")


plt.figure()
plt.imshow(aer_training_kernel, cmap="hot")
plt.show()


np.savetxt("./training_kernels/aer_kernel_matrix.csv", aer_training_kernel, delimiter=",")

with open('accuracy_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([classical_ml_train_accuracy, classical_ml_test_accuracy, aer_balanced_accuracy, aer_f1_score])