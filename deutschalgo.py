import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# --- 1. Define the Oracle Functions (U_f) ---
# Each oracle is a QuantumCircuit acting on 2 qubits:
# q[0] is the input qubit (x)
# q[1] is the auxiliary/output qubit (y)
# The oracle implements |x>|y> -> |x>|y XOR f(x)>

def get_oracle(function_type):
    """
    Returns a QuantumCircuit representing the oracle U_f for a given function type.
    function_type: 'f0', 'f1', 'fx', 'f1_minus_x'
    """
    qc_oracle = QuantumCircuit(2, name=f'Uf_{function_type}')

    if function_type == 'f0':  # f(x) = 0 (Constant)
        # U_f |x>|y> = |x>|y XOR 0> = |x>|y>
        # This is the identity operation, so no gates are needed.
        pass
    elif function_type == 'f1':  # f(x) = 1 (Constant)
        # U_f |x>|y> = |x>|y XOR 1> = |x>|NOT y>
        # This means q[1] is always flipped, regardless of q[0].
        qc_oracle.x(1)
    elif function_type == 'fx':  # f(x) = x (Balanced)
        # U_f |x>|y> = |x>|y XOR x>
        # This is a CNOT gate where q[0] controls q[1].
        qc_oracle.cx(0, 1)
    elif function_type == 'f1_minus_x':  # f(x) = 1 - x (Balanced)
        # U_f |x>|y> = |x>|y XOR (1-x)>
        # This means q[1] is flipped if x=0, and not flipped if x=1.
        # This is an X gate on q[0], then CNOT, then X gate on q[0] again.
        qc_oracle.x(0)   # Flip q[0] so that if q[0] was |0>, it becomes |1> (control active)
        qc_oracle.cx(0, 1) # CNOT controlled on the flipped q[0]
        qc_oracle.x(0)   # Flip q[0] back
    else:
        raise ValueError("Invalid function_type. Choose from 'f0', 'f1', 'fx', 'f1_minus_x'.")
    
    return qc_oracle.to_gate() # Convert to a reusable gate

# --- 2. Construct the Deutsch Circuit ---

def build_deutsch_circuit(oracle_gate):
    """
    Builds the full Deutsch circuit given an oracle gate.
    """
    qc = QuantumCircuit(2, 1) # 2 qubits, 1 classical bit for measurement

    # Step 1: Initialize q[1] to |1> (auxiliary qubit)
    qc.x(1)

    # Step 2: Apply Hadamard gates to both qubits
    qc.h(0)
    qc.h(1)

    qc.barrier() # Optional: For visual separation in drawings

    # Step 3: Apply the oracle U_f
    qc.append(oracle_gate, [0, 1])

    qc.barrier() # Optional: For visual separation

    # Step 4: Apply Hadamard gate to the first qubit (q[0])
    qc.h(0)

    # Step 5: Measure the first qubit (q[0])
    qc.measure(0, 0) # Measure q[0] into classical bit 0

    return qc

# --- 3. Simulate and Interpret Results ---

def run_deutsch_algorithm(function_type):
    """
    Sets up, runs, and interprets Deutsch's algorithm for a given function type.
    """
    print(f"\n--- Testing function: {function_type} ---")

    # Get the oracle for the chosen function
    oracle = get_oracle(function_type)

    # Build the full Deutsch circuit
    deutsch_circuit = build_deutsch_circuit(oracle)
    print("Circuit Diagram:")
    print(deutsch_circuit.draw(output='text'))

    # Use AerSimulator for local simulation
    simulator = AerSimulator()

    # Transpile the circuit for the simulator
    compiled_circuit = transpile(deutsch_circuit, simulator)

    # Run the simulation
    job = simulator.run(compiled_circuit, shots=1024) # Run 1024 times for statistics
    result = job.result()
    counts = result.get_counts(compiled_circuit)

    # Interpret the result
    print(f"Measurement Counts: {counts}")

    # Deutsch's algorithm output:
    # If q[0] measures '0', the function is constant.
    # If q[0] measures '1', the function is balanced.

    if '0' in counts and counts['0'] > counts.get('1', 0):
        print(f"Result: The function {function_type} is **CONSTANT**.")
        expected_type = "Constant"
    elif '1' in counts and counts['1'] > counts.get('0', 0):
        print(f"Result: The function {function_type} is **BALANCED**.")
        expected_type = "Balanced"
    else:
        print("Unexpected measurement outcome distribution.")
        expected_type = "Unknown"
    
    # Verify correctness based on our definition
    if function_type in ['f0', 'f1']:
        print(f"Expected: Constant. Match: {expected_type == 'Constant'}")
    else: # 'fx', 'f1_minus_x'
        print(f"Expected: Balanced. Match: {expected_type == 'Balanced'}")

# --- Run the algorithm for each function type ---
run_deutsch_algorithm('f0')
run_deutsch_algorithm('f1')
run_deutsch_algorithm('fx')
run_deutsch_algorithm('f1_minus_x')