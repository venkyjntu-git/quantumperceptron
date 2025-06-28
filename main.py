from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer.primitives import Estimator
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
#qc.draw(output="mpl")
ZZ = Pauli("ZZ")
ZI = Pauli("ZI")
XX = Pauli("XX")
IZ = Pauli("IZ")
XI = Pauli("XI")
IX = Pauli("IX")
observables = [ZZ, ZI, IZ, XX, XI, IX]

estimator = Estimator()

job = estimator.run([qc]*len(observables), observables)

result = job.result()

print(result)

data = result.values
labels = [str(o) for o in observables]

plt.plot(labels, data, "o-")
plt.xlabel("Observable")
plt.ylabel("Expectation Value")
plt.show()