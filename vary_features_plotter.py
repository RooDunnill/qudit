import matplotlib.pyplot as plt
import numpy as np
import csv

classical_y_data = []
aer_y_data = []

classical_avg = np.zeros(8)
aer_avg = np.zeros(8)
classical_std = np.zeros(8)
aer_std = np.zeros(8)

with open("accuracy_data_feature_variation.csv", newline="") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        aer_y_data.append(float(row[-1]))
        classical_y_data.append(float(row[-2]))


for i in range(8):
    for j in range(10):
        classical_avg[i] += classical_y_data[j+10*i]
        aer_avg[i] += aer_y_data[j+10*i]
    classical_std[i] = np.std(classical_y_data[10*i:10*(i+1)], ddof=1) / np.sqrt(10)
    aer_std[i] = np.std(aer_y_data[10*i:10*(i+1)], ddof=1) / np.sqrt(10)
classical_avg /= 10
aer_avg /= 10

x_points = np.arange(10, 2, -1)

plt.errorbar(x_points, classical_avg, yerr=classical_std, label="Classical SVM", color="black", marker=".", linestyle="--", capsize=5)
plt.errorbar(x_points, aer_avg, yerr=aer_std, label="Qiskit Noiseless QSVM", color="red", marker=".", linestyle="--", capsize=5)

plt.ylim(0,1)
plt.xlim(10,3)
plt.legend()
plt.title("Probabilities of different PCA models against feature size")
plt.ylabel("Accuracy")
plt.xlabel("Feature Size")
plt.figtext(.16,.75,"Train Size: 80\n Test Size: 20\n Shots=1000")
plt.grid()
plt.show()