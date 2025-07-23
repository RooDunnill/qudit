import matplotlib.pyplot as plt
import numpy as np
import csv

classical_y_data = []
aer_y_data = []
superstaq_y_data = []

x_points = [0,1,2]


with open("accuracy_data_compare_3.csv", newline="") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:      
        aer_y_data.append(float(row[-2]))
        classical_y_data.append(float(row[-3]))
        superstaq_y_data.append(float(row[-1]))

classical_avg = np.sum(classical_y_data) / 5
aer_avg = np.sum(aer_y_data) / 5
superstaq_avg = np.sum(superstaq_y_data) / 5

classical_std = np.std(classical_y_data) / np.sqrt(5)
aer_std = np.std(aer_y_data) / np.sqrt(5)
superstaq_std = np.std(superstaq_y_data) / np.sqrt(5)


y_points = [classical_avg, aer_avg, superstaq_avg]
y_std = [classical_std, aer_std, superstaq_std]

plt.bar(x_points, y_points, color="orange")
plt.errorbar(x_points, y_points, yerr=y_std, linestyle="None", capsize=5, color="black")

plt.title("Model Comparisons")
plt.ylabel("Accuracy")

plt.grid()
plt.figtext(.16,.75,"Train Size: 12\n Test Size: 10\n Shots=1000\n Features: 4")

plt.text(0,0.2, "Classical")
plt.text(1,0.2, "Noiseless")
plt.text(2,0.2, "Noise")
plt.ylim(0,1)
plt.show()