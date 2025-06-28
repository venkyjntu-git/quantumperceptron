# Import necessary libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Qiskit imports for algorithms and optimization
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit.primitives import Sampler

# Qiskit Optimization module for problem formulation
from qiskit_optimization.applications.ising import Tsp # Import the Tsp application
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# For running on a local Aer simulator
from qiskit_aer import AerSimulator

# --- Step 1: Define the Problem (TSP on 3 cities) ---

# Define cities and their distances
# We'll use 3 cities to keep the problem small enough for basic simulation.
# For N cities, TSP typically involves N*N binary variables (qubits).
# City names (arbitrary, could be A, B, C)
cities = ["A", "B", "C"]
num_cities = len(cities)

# Distance matrix (symmetric, distance from i to j)
# distances[i][j] is the distance from city i to city j
# Example: 0->1 (A->B) is 10, 0->2 (A->C) is 15
#          1->2 (B->C) is 12
# Note: For TSP, the cost matrix usually implies the existence of an edge.
# We'll represent this as a complete graph for simplicity.
distances = np.array([
    [0, 10, 15],  # A to A, B, C
    [10, 0, 12],  # B to A, B, C
    [15, 12, 0]   # C to A, B, C
])

# Create a graph to represent the cities and distances
# In TSP, we typically work with a complete graph where all cities are connected.
graph = nx.Graph()
for i in range(num_cities):
    graph.add_node(i, label=cities[i]) # Add nodes with labels
for i in range(num_cities):
    for j in range(i + 1, num_cities):
        graph.add_edge(i, j, weight=distances[i, j]) # Add edges with weights

# Visualize the graph
plt.figure(figsize=(5, 5))
pos = nx.circular_layout(graph) # positions for all nodes
nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="lightcoral")
nx.draw_networkx_labels(graph, pos, labels={i: cities[i] for i in range(num_cities)},
                        font_size=16, font_weight="bold")
nx.draw_networkx_edges(graph, pos, width=1.5)
# Draw edge labels (distances)
edge_labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='green')
plt.title("Graph for Traveling Salesperson Problem")
plt.axis('off') # Hide axes
plt.show()

# --- Step 2: Formulate the Problem as a Quadratic Program (QUBO) ---

# Use the Tsp application from Qiskit Optimization.
# It converts the TSP instance (distances and start node) into a QuadraticProgram.
# The `factor` parameter scales the penalties for constraint violations.
# A higher factor makes violations more "expensive" and thus less likely to be chosen.
tsp_problem = Tsp(graph=graph, start_node=0) # Assuming start and end at city 0 (A)
qp = tsp_problem.to_quadratic_program()

print("Quadratic Program formulation for TSP (partial view):")
# Print a snippet of the LP string as it can be very long for TSP
qp_lp_string = qp.export_as_lp_string()
print(qp_lp_string[:500] + "...\n") # Print first 500 characters + ellipsis

# Convert the QuadraticProgram to a QUBO (Quadratic Unconstrained Binary Optimization)
# This step ensures the problem is in the correct format for QAOA.
# For N cities, the QUBO formulation often involves N*N binary variables,
# representing x_{city_idx, time_step_idx}. So for 3 cities, it's 3*3 = 9 qubits.
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)

print(f"Number of qubits required: {qubo.get_num_vars()}")
# The resulting QUBO will be internally handled by QAOA.

# --- Step 3: Configure and Run QAOA ---

# Set up the classical optimizer for QAOA.
# COBYLA is a good choice for smaller, noisy optimization problems.
optimizer = COBYLA(maxiter=250) # Increased iterations, might need more for convergence

# Set up the quantum backend (simulator)
# AerSimulator is Qiskit's high-performance simulator for quantum circuits.
simulator = AerSimulator()

# The Sampler primitive executes quantum circuits and returns probabilities
# or counts of measurement outcomes. QAOA uses it to get expectation values.
sampler = Sampler()

# Initialize QAOA.
# p (reps) is the number of QAOA layers. A larger p means a deeper circuit
# and potentially better results, but also requires more quantum resources.
# For TSP, a single layer (reps=1) might not be sufficient to find the optimal solution
# reliably, but we use it for demonstration due to simulator constraints.
qaoa_mes = QAOA(sampler=sampler, optimizer=optimizer, reps=1) # reps=p

# Use the Tsp application's built-in `to_ising()` method to get the
# Ising Hamiltonian directly from the graph. QAOA needs this for its cost function.
# The `Tsp.to_ising()` method returns a tuple: (qubit_operator, offset)
operator, offset = tsp_problem.to_ising()

# Run QAOA to find the minimum eigenvalue of the Hamiltonian,
# which corresponds to the solution of the TSP.
# The result will be a MinimumEigensolverResult object.
print("\nRunning QAOA on the simulator...")
# Note: For TSP, finding the absolute global minimum can be hard even for simulators
# with low 'reps'. The result will be an *approximate* solution.
qaoa_result = qaoa_mes.compute_minimum_eigenvalue(operator)
print("QAOA run complete.")

# --- Step 4: Interpret the Results ---

# The QAOA result contains the optimal parameters found by the classical optimizer,
# the measured eigenvalues, and the optimal state (the most probable bit string).
optimal_value = qaoa_result.optimal_value # This is the minimum eigenvalue found
optimal_state_bit_string = qaoa_result.optimal_physical_point # The bit string corresponding to the solution

print(f"\nOptimal bit string found by QAOA: {optimal_state_bit_string}")
print(f"Minimum eigenvalue (energy) found by QAOA: {optimal_value}")

# Decode the bit string back into the TSP route
# The `Tsp` application has a helper function for this
decoded_route = tsp_problem.interpret(x=optimal_state_bit_string)

# The `interpret` method returns a list of city indices for the route,
# and a `feasible` boolean indicating if the route satisfies all constraints.
# For TSP, it returns a list of lists if multiple paths are found with the same minimum value.
# We'll just take the first one if it's feasible.

# tsp_problem.interpret returns a list of feasible paths, each path is a list of node indices
found_routes = []
if decoded_route is not None:
    for route_list in decoded_route:
        # Check if route_list is not empty before processing
        if route_list:
            # Convert node indices back to city names
            city_route = [cities[idx] for idx in route_list]
            found_routes.append(city_route)

if found_routes:
    print(f"Decoded feasible route(s): {found_routes}")
    # Calculate the cost of the first feasible route found
    # The route should always start and end at the `start_node` (city 0 in our case)
    # The `interpret` function often returns routes without the explicit final return to start node
    # so we'll need to add it for cost calculation if it's not implicitly closed.
    # The interpretation logic can be complex depending on how the QP was set up.
    # For `Tsp` application, `interpret` usually gives the full cycle.

    # Let's verify the cost of the first found route if it's a full cycle
    # For a 3-city TSP, the optimal route length for [A, B, C] is A-B-C-A or A-C-B-A.
    # Let's manually calculate the cost for a sample path if `interpret` gives indices
    # e.g., if decoded_route is [[0, 1, 2]] (A->B->C)
    if found_routes[0] and len(found_routes[0]) == num_cities:
        # Construct the full cycle for cost calculation: A -> B -> C -> A
        path_indices = [cities.index(c) for c in found_routes[0]]
        full_path_indices = path_indices + [path_indices[0]] # Close the cycle
        
        calculated_cost = 0
        for i in range(len(full_path_indices) - 1):
            from_city_idx = full_path_indices[i]
            to_city_idx = full_path_indices[i+1]
            calculated_cost += distances[from_city_idx, to_city_idx]
        print(f"Calculated cost for the route {found_routes[0] + [found_routes[0][0]]}: {calculated_cost}")
    else:
        print("Warning: Decoded route might not be a complete cycle or valid. Cost calculation skipped.")

else:
    print("No feasible route found by QAOA for this run.")
    print("This can happen for complex problems, especially with low 'reps' or limited optimizer iterations.")

# For a 3-city TSP with distances given:
# A->B->C->A: 10 + 12 + 15 = 37
# A->C->B->A: 15 + 12 + 10 = 37
# The optimal cost is 37. QAOA aims to find parameters that lead to this state.
