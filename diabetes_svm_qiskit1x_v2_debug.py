# diabetes_svm_qiskit1x_v2_debug.py
# Debug: Small dataset, DensityMatrixSimulator to avoid WSL memory crash

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

from qiskit_aer import Aer
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import matplotlib.pyplot as plt

# 📂 Load dataset
data = np.loadtxt("diabetes_hackathon.csv", delimiter=",", dtype=str)[1:]

# 🗑️ Remove unnecessary columns (keep Gender, Age, BMI, HbA1c, Class)
remove_cols = [2, 3, 5, 6, 7, 8, 9]
data = np.delete(data, remove_cols, axis=1)

# 🔥 Encode categorical data
def encode(data):
    data[data[:, 0] == 'F', 0] = 1
    data[data[:, 0] == 'M', 0] = 0
    data[data[:, -1] == 'Y', -1] = 1
    data[data[:, -1] != 'Y', -1] = 0
    return data.astype(float)

data = encode(data)

# 📊 Normalize
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X = data[:, :-1]
y = data[:, -1]

# 🔀 Small subset for debug
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = X_train[:10]
y_train = y_train[:10]
X_test = X_test[:5]
y_test = y_test[:5]

print(f"Debug: Training set size = {len(X_train)}, Test set size = {len(X_test)}")

# 🎯 Classical SVM
print("\n=== Classical SVM ===")
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred_classical = svc.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_classical))
print(classification_report(y_test, y_pred_classical))

# ⚛️ Quantum Kernel SVM
print("\n=== Quantum Kernel SVM (Debug Mode) ===")
feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=1, entanglement='full')

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

train_kernel = quantum_kernel.evaluate(x_vec=X_train)
test_kernel = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

qsvc = SVC(kernel="precomputed")
qsvc.fit(train_kernel, y_train)
y_pred_qsvc = qsvc.predict(test_kernel)

print("Accuracy:", accuracy_score(y_test, y_pred_qsvc))
print(classification_report(y_test, y_pred_qsvc))

# 📊 Compare
plt.bar(['Classical', 'Quantum'], [accuracy_score(y_test, y_pred_classical), accuracy_score(y_test, y_pred_qsvc)], color=['blue', 'purple'])
plt.ylabel('Accuracy')
plt.title('Debug: Classical vs Quantum')
plt.show()
