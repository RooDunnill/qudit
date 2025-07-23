import csv
import numpy as np


testing_data = []
training_data = []

with open("accuracy_data_classical.csv", newline="") as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        testing_data.append(float(row[1]))
        training_data.append(float(row[0]))



test_avg = np.sum(testing_data) / len(testing_data)
train_avg = np.sum(training_data) / len(training_data)

test_std = np.std(testing_data) / np.sqrt(len(testing_data))
train_std = np.std(training_data) / np.sqrt(len(training_data))
print(f"Training Accuracy: {train_avg:.3f}+{train_std:.3f}")
print(f"Testing Accuracy: {test_avg:.3f}+{test_std:.3f}")