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
import time

def main():
    start_time = time.time()
    data_set = 0
    data = [] 
    shots = 1000
    testing_size = 2                                                    #the size of the testing set
    training_size = 2                                                   #the size of the training set        
    run_superstaq = 0

    if data_set == 0:
        data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)     #imports the data from the csv
        remove_cols = [2,3,4,7,8,9]                                           #removes the features that aren't needed
        features = 11 - len(remove_cols)                                          #finds the amount of features used
        normalized_data = encode_ds1(data, remove_cols)
    else:
        data = np.loadtxt("diabetes_dataset_2.csv", delimiter=",", dtype=str)
        remove_cols = [1,2,3,4]
        features = 8 - len(remove_cols)
        normalized_data = encode_ds2(data, remove_cols)



    


                                
    results = normalized_data[:,features:]                              #removes all columns of data bar results
    data = normalized_data[:,:features]                                 #removes the results column off of the end
                                                                        #splits both the results and the data up into testing and training
    training_data, testing_data, training_results, testing_results = train_test_split(data, results, shuffle=True, test_size=testing_size, train_size=training_size)

    training_results = training_results.T[0]                          #reshapes the arrays
    testing_results = testing_results.T[0]



    feature_map = ZZFeatureMap(feature_dimension=features, reps=1, entanglement="linear").decompose()   #entangled feature map for double z rotations
    feature_map_inv = ZZFeatureMap(feature_dimension=features, reps=1, parameter_prefix="y", entanglement="linear").decompose().inverse()

    circ = QuantumCircuit(features)                           #initiates a circuit with the num of qubits equal to the num of features
    circ.compose(feature_map, inplace=True)                   #adds the feature map to the circuit
    circ.compose(feature_map_inv, inplace=True)               #adds the inverted feature map to the circuit
    circ.measure_all()                                        #measures every qubit

    provider = qss.SuperstaqProvider(api_key=API_KEY)         #initiates and verifies the superstaq provider
    superstaq_sim = provider.get_backend("cq_sqale_qpu")      #uses the given gpu as the backend
    sim = AerSimulator()                                      #uses qiskits aer sim


    def train_k_mat_aer(training_data):                     
        """This function creates the training data kernel for Aer"""
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
        """This function creates the testing data kernel for Aer"""
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
        """This function creates the training data kernel for Superstaq"""
        mat_dim = len(training_data)
        kernel_mat = np.zeros((mat_dim, mat_dim))          #creates an empty array
        for i in range(mat_dim):                           #loops over the size of the square matrix
            for j in range(i, mat_dim):                    #loops over the offdiagonal components of that row
                if j == i:                                 #makes diagonal terms equal 1
                    kernel_mat[i,i] += 1
                else:
                    bound_circ = circ.assign_parameters(np.append(training_data[i], training_data[j]))  #takes the two data points and adds them to the feature map
                    result = superstaq_sim.run(bound_circ, shots=shots, method="noise-sim").result()          #runs the superstaq simulator
                    counts = result.get_counts()                                                        #returns the counts for that given result
                    kernel_mat[i,j] = counts.get('0' * features, 0) / shots                             #takes the counts for |0^n> and normalises
                    kernel_mat[j,i] = counts.get('0' * features, 0) / shots                             #copies to the other side of the matrix
        return kernel_mat

    def test_k_mat_superstaq(training_data, testing_data):
        """This function creates the testing data kernel for Superstaq"""
        y_mat_dim = len(training_data)
        x_mat_dim = len(testing_data)
        kernel_mat = np.zeros((x_mat_dim, y_mat_dim))
        provider = qss.SuperstaqProvider(api_key=API_KEY)
        sim = provider.get_backend("cq_sqale_qpu")
        for i in range(x_mat_dim):
            for j in range(y_mat_dim):
                    bound_circ = circ.assign_parameters(np.append(testing_data[i], training_data[j]))
                    result = sim.run(bound_circ, shots=shots, method="noise-sim").result()
                    counts = result.get_counts()
                    kernel_mat[i,j] = counts.get('0' * features, 0) / shots
        return kernel_mat



    aer_training_kernel = train_k_mat_aer(training_data)                          #runs the training data through the kernel matrix function
    aer_testing_kernel = test_k_mat_aer(training_data, testing_data)              #creates the testing kernel

    svc_aer = svm.SVC(kernel="precomputed")                                       #takes a precomputed kernel and runs
    svc_aer.fit(aer_training_kernel, training_results)                            #trains with the given kernel matrix
    aer_y_predict = svc_aer.predict(aer_testing_kernel)                           #predicts outcome from the testing data


    clf = svm.SVC(kernel = "linear")                  #classical simulator kernel
    clf.fit(training_data, training_results)          #trains the kernel with the training data

    classical_ml_train_accuracy = clf.score(training_data, training_results)                 #computes the accuracy of the algorithm in training
    classical_ml_test_accuracy = accuracy_score(testing_results, clf.predict(testing_data))  #computes the accuracy of the algorithm in testing
    aer_balanced_accuracy = balanced_accuracy_score(testing_results, aer_y_predict)          #computes the balanced accuracy of the aer simulator
    aer_f1_score = f1_score(testing_results, aer_y_predict, average="weighted")             #computes the f1 accuracy of the aer simulator

    
    print(f"Machine Learning Results of data set {data_set+1}".center(100, "="))
    print(f"Classical Machine Learning Training Accuracy: {classical_ml_train_accuracy:.3f}")
    print(f"Classical Machine Learning Testing Accuracy: {classical_ml_test_accuracy:.3f}")

    print(f"Aer Simulator Balanced Accuracy Score: {aer_balanced_accuracy:.3f}")
    print(f"Aer Simulator F1 Score: {aer_f1_score:.3f}")

    if run_superstaq == 1:           #only runs superstaq if flag is toggled
        superstaq_training_kernel = train_k_mat_superstaq(training_data)
        superstaq_testing_kernel = test_k_mat_superstaq(training_data, testing_data)
        svc_superstaq = svm.SVC(kernel="precomputed")
        svc_superstaq.fit(superstaq_training_kernel, training_results)
        superstaq_y_predict = svc_superstaq.predict(superstaq_testing_kernel)
        print(f"Superstaq Simulator Balanced Accuracy Score: {balanced_accuracy_score(testing_results, superstaq_y_predict)}")
        print(f"Superstaq Simulator F1 Score: {f1_score(testing_results, superstaq_y_predict, average="weighted")}")



    np.savetxt("./training_kernels/aer_kernel_matrix.csv", aer_training_kernel, delimiter=",")    #saves the training kernel to a file

    with open('accuracy_data.csv', 'a', newline='') as f:    #saves the relevent daccuracy data
        writer = csv.writer(f)
        writer.writerow([data_set, features, training_size, testing_size, shots, classical_ml_train_accuracy, classical_ml_test_accuracy, aer_balanced_accuracy, aer_f1_score])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Machine Learning Algorithms took {elapsed_time:.3f} seconds to run")

while True:
    main()


plt.figure()
plt.imshow(aer_training_kernel, cmap="hot")          #plots and displays a heatmap of the aer simulator
plt.show()