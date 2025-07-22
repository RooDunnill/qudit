from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import UnitaryGate


feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement="linear").decompose()   #entangled feature map for double z rotations
feature_map_inv = ZZFeatureMap(feature_dimension=3, reps=1, parameter_prefix="y", entanglement="linear").decompose().inverse()

circ = QuantumCircuit(3)                           #initiates a circuit with the num of qubits equal to the num of features
circ.compose(feature_map, inplace=True)                   #adds the feature map to the circuit
circ.compose(feature_map_inv, inplace=True)               #adds the inverted feature map to the circuit
circ.measure_all()     

circ2 = QuantumCircuit()
circ2 = UnitaryGate([[0,0],[1,1]])


print(circ.draw(filename="./circuit_diagrams/feature_map.png", output="mpl", fold=-1))