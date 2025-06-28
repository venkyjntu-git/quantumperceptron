import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA # Or COBYLA, depending on what you're using
from qiskit.circuit.library import RealAmplitudes
from functools import partial

print("--- Program Start ---")

# --- 1. Data Encoding (Feature Map) ---
def feature_map_and_or(qc: QuantumCircuit, x: np.ndarray):
    if x[0] == 1:
        qc.x(0)
    if x[1] == 1:
        qc.x(1)
    qc.barrier()

# --- 2. Parameterized Quantum Circuit (Ansatz) ---
NUM_QUBITS = 3
REPS = 3

def vqc_and_or_prediction(
    input_data: np.ndarray,
    params: np.ndarray,
    simulator: AerSimulator,
    ansatz_circuit: QuantumCircuit,
    shots: int = 1024
):
    #print(f"  [Prediction] Starting for input: {input_data}") # DEBUG
    qc = QuantumCircuit(NUM_QUBITS, 1)
    feature_map_and_or(qc, input_data)

    # Check if ansatz_circuit is valid
    if not isinstance(ansatz_circuit, QuantumCircuit):
        print(f"  [Prediction ERROR] ansatz_circuit is not a QuantumCircuit! Type: {type(ansatz_circuit)}") # DEBUG
        raise TypeError("ansatz_circuit must be a QuantumCircuit object.")

    try:
        # Ensure params has the correct shape for assign_parameters
        # RealAmplitudes expects a 1D array
        if params.ndim > 1:
            params = params.flatten()
        qc.compose(ansatz_circuit.assign_parameters(params), inplace=True)
    except Exception as e:
        print(f"  [Prediction ERROR] Error assigning parameters or composing: {e}") # DEBUG
        print(f"  [Prediction ERROR] Params shape: {params.shape}, Num expected by ansatz: {ansatz_circuit.num_parameters}") # DEBUG
        raise # Re-raise to see full traceback

    qc.measure(NUM_QUBITS - 1, 0)

    try:
        compiled_circuit = transpile(qc, simulator)
        job = simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(compiled_circuit)
        prob_one = counts.get('1', 0) / shots
        #print(f"  [Prediction] Finished for input: {input_data}, Prob: {prob_one:.4f}") # DEBUG
        return prob_one
    except Exception as e:
        print(f"  [Prediction ERROR] Error during simulation or measurement: {e}") # DEBUG
        raise # Re-raise to see full traceback


# --- Full VQC Training Loop ---

def cost_function(params, training_data, simulator, ansatz_circuit):
    #print(f"  [Cost] Evaluating cost for params (first few): {params[:5]}...") # DEBUG
    total_cost = 0
    train_shots = 1024
    for x, true_y in training_data:
        #print(f"    [Cost] Processing training point: {x}") # DEBUG
        predicted_prob = vqc_and_or_prediction(np.array(x), params, simulator, ansatz_circuit, shots=train_shots)
        total_cost += (predicted_prob - true_y)**2
    #print(f"  [Cost] Total cost calculated: {total_cost:.6f}") # DEBUG
    return total_cost

print("--- Initializing ---")

training_data_or = [
    ([0,0], 0),
    ([0,1], 1),
    ([1,0], 1),
    ([1,1], 1)
]

sim = AerSimulator()
ansatz_for_vqc = RealAmplitudes(NUM_QUBITS, reps=REPS, entanglement='circular')
num_params = ansatz_for_vqc.num_parameters
initial_params = np.random.rand(num_params) * 2 * np.pi

print(f"Ansatz has {num_params} parameters.")
print(f"Initial parameters (first 5): {initial_params[:5]}") # DEBUG

optimizer = SPSA(maxiter=2000)

print("--- Preparing Optimizer ---")

objective_function_for_optimizer = partial(cost_function,
                                           training_data=training_data_or,
                                           simulator=sim,
                                           ansatz_circuit=ansatz_for_vqc)

print("--- Starting Optimization ---")
try:
    result = optimizer.minimize(
        fun=objective_function_for_optimizer,
        x0=initial_params
    )
    print("--- Optimization Finished ---") # DEBUG
except Exception as e:
    print(f"--- Optimization ERROR: {e} ---") # DEBUG
    import traceback
    traceback.print_exc() # Print full traceback
    exit() # Exit if error during optimization

optimized_params = result.x
final_cost = result.fun

print(f"\nOptimization finished.")
print(f"Optimized parameters: {optimized_params}")
print(f"Final cost: {final_cost:.6f}")

print("\n--- Testing Optimized VQC for OR Function ---")

# Test the VQC with the optimized parameters
for x1 in [0, 1]:
    for x2 in [0, 1]:
        input_val = np.array([x1, x2])
        prob_output_is_one = vqc_and_or_prediction(input_val, optimized_params, sim, ansatz_for_vqc, shots=4096)

        predicted_output = 1 if prob_output_is_one > 0.5 else 0
        true_output = 1 if (x1 == 1 or x2 == 1) else 0
        status = "CORRECT" if predicted_output == true_output else "INCORRECT"

        print(f"Input: ({x1}, {x2}), True: {true_output}, P(output=1): {prob_output_is_one:.4f}, Predicted: {predicted_output} [{status}]")

print("--- Program End ---") # DEBUG














# import numpy as np
# from qiskit import QuantumCircuit, transpile
# from qiskit_aer import AerSimulator
# from qiskit_algorithms.optimizers import COBYLA, SPSA

# from functools import partial


# # from qiskit.opflow import StateFn, PauliSumOp
# # from qiskit.utils import QuantumInstance
# # from qiskit.algorithms.minimum_eigen_solvers import VQE # Uncomment for VQE related components

# # --- 1. Data Encoding (Feature Map) ---
# def feature_map_and(qc: QuantumCircuit, x: np.ndarray):
#     """
#     A simple feature map to encode two classical inputs (x1, x2) into a quantum state.
#     For AND, inputs are 0 or 1. We'll map them directly to basis states.
#     For more complex data, this would involve rotations proportional to features.

#     Args:
#         qc (QuantumCircuit): The quantum circuit to apply the feature map to.
#         x (np.ndarray): A 2-element array [x1, x2].
#     """
#     if x[0] == 1:
#         qc.x(0)
#     if x[1] == 1:
#         qc.x(1)
#     qc.barrier()

# # --- 2. Parameterized Quantum Circuit (Ansatz) ---
# def ansatz_and(qc: QuantumCircuit, params: np.ndarray):
#     """
#     A more expressive parameterized ansatz that can learn the AND function.
#     It uses controlled rotations and single qubit rotations.
#     """
#     # Parameters for rotations: RZ on q0, RZ on q1, CRZ from q0 to q2, CRZ from q1 to q2,
#     # and a final RZ on q2. Total 5 parameters.
#     # We will adjust the number of params later based on this structure.

#     # Apply parameterized single-qubit rotations on input qubits if needed,
#     # or just to the 'output' qubit to capture overall bias.
#     qc.rz(params[0], 0) # Parameterized rotation on input q0
#     qc.rz(params[1], 1) # Parameterized rotation on input q1

#     # Key part for AND: controlled rotations to influence the output qubit (q2)
#     # only when specific input conditions are met.
#     # CRZ(angle, control, target)
#     qc.crz(params[2], 0, 2) # Control RZ on q2 by q0
#     qc.crz(params[3], 1, 2) # Control RZ on q2 by q1

#     # A final rotation on the output qubit to adjust its state
#     qc.rz(params[4], 2)

#     qc.barrier()
# # --- 3. Cost Function (Classical) and Optimization Loop ---
# # In a full VQC, you'd run this iteratively, adjusting params.

# def vqc_and_conceptual_prediction(
#     input_data: np.ndarray,
#     params: np.ndarray,
#     simulator: AerSimulator,
#     shots: int = 1024
# ):
#     """
#     Conceptual function to show how a VQC would make a prediction.
#     It simulates a quantum circuit with given inputs and parameters,
#     and returns a "classification score" (e.g., probability of measuring |1>).

#     Args:
#         input_data (np.ndarray): The [x1, x2] input.
#         params (np.ndarray): The current set of learned parameters for the ansatz.
#         simulator (AerSimulator): The Qiskit simulator.
#         shots (int): Number of shots for measurement.

#     Returns:
#         float: Probability of measuring the 'output' qubit as 1.
#     """
#     # 3 qubits: q0 (x1), q1 (x2), q2 (output/classification qubit)
#     qc = QuantumCircuit(3, 1)

#     # 1. Encode data
#     feature_map_and(qc, input_data)

#     # 2. Apply parameterized ansatz
#     ansatz_and(qc, params)

#     # 3. Measure the classification qubit (q[2])
#     qc.measure(2, 0)

#     # Execute the circuit
#     compiled_circuit = transpile(qc, simulator)
#     job = simulator.run(compiled_circuit, shots=shots)
#     result = job.result()
#     counts = result.get_counts(compiled_circuit)

#     # Get probability of measuring '1' for the output qubit
#     prob_one = counts.get('1', 0) / shots
#     return prob_one

# # print("\n--- Conceptual VQC for AND Function ---")

# # # Define a set of "learned" parameters (these would be found via optimization)
# # # For AND, if params[0] is set such that it pushes the |11> case's q2 state
# # # to |1> and others to |0>, it would work.
# # # This is a hand-picked value for illustration, not from actual training.
# # # A full training would involve an optimization loop.
# # learned_params = np.array([np.pi/2]) # Just one parameter for the simple RZ gate

# # # Initialize simulator
# # sim = AerSimulator()

# # # Test the conceptual VQC for all inputs
# # for x1 in [0, 1]:
# #     for x2 in [0, 1]:
# #         input_val = np.array([x1, x2])
# #         prob_output_is_one = vqc_and_conceptual_prediction(input_val, learned_params, sim)
        
# #         # Simple threshold for classification
# #         predicted_output = 1 if prob_output_is_one > 0.5 else 0
        
# #         print(f"Input: ({x1}, {x2}), P(output=1): {prob_output_is_one:.4f}, Predicted: {predicted_output}")




# def cost_function(params, training_data, simulator): # <--- 1. 'simulator' is a parameter here
#     total_cost = 0
#     for x, true_y in training_data:
#         predicted_prob = vqc_and_conceptual_prediction(np.array(x), params, simulator) # <--- 2. 'simulator' is passed here
#         total_cost += (predicted_prob - true_y)**2
#     return total_cost

# # Initialize simulator once
# sim = AerSimulator()


# #Initialize an optimizer (e.g., COBYLA from Qiskit's optimizers)
# #optimizer = COBYLA(maxiter=1000)

# optimizer = SPSA(maxiter=2000) # Increased iterations for SPSA

# #Initial random parameters
# initial_params = np.random.rand(5) * 2 * np.pi
#  # Training data for AND
# training_data = [
#         ([0,0], 0),
#         ([0,1], 1),
#         ([1,0], 1),
#         ([1,1], 1)
#     ]
   
# objective_function_for_optimizer = partial(cost_function,
#                                            training_data=training_data,
#                                            simulator=sim)
# # Run the optimization
# result = optimizer.minimize(
#     fun=objective_function_for_optimizer, # Pass the partial function
#     x0=initial_params
# )

# optimized_params = result.x
# print(f"\nOptimized parameters: {optimized_params}")
# final_cost = result.fun
# print(f"Final cost: {final_cost:.6f}")

# print("\n--- Testing Optimized VQC for AND Function ---")

# for x1 in [0, 1]:
#     for x2 in [0, 1]:
#         input_val = np.array([x1, x2])
#         prob_output_is_one = vqc_and_conceptual_prediction(input_val, optimized_params, sim, shots=4096)

#         predicted_output = 1 if prob_output_is_one > 0.5 else 0
#         true_output = 0 if (x1 == 0 and x2 == 0) else 1
#         status = "CORRECT" if predicted_output == true_output else "INCORRECT"

#         print(f"Input: ({x1}, {x2}), True: {true_output}, P(output=1): {prob_output_is_one:.4f}, Predicted: {predicted_output} [{status}]")
