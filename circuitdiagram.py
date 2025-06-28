# from qiskit.circuit.library import RealAmplitudes
# from qiskit import QuantumCircuit

# # Define the parameters for the ansatz
# NUM_QUBITS = 3
# REPS = 2
# ENTANGLEMENT_TYPE = 'circular'

# # Create the RealAmplitudes ansatz
# ansatz = RealAmplitudes(NUM_QUBITS, reps=REPS, entanglement=ENTANGLEMENT_TYPE)

# # You can optionally add a measure gate for completeness if you like,
# # though the ansatz itself doesn't have it by default.
# # qc_with_measure = ansatz.copy()
# # qc_with_measure.measure_all() # This adds measurement gates to all qubits

# # Draw the circuit
# # To get a better output for slides, you can choose 'mpl' (Matplotlib)
# # which is generally high quality. You might need to install matplotlib:
# # pip install matplotlib
# circuit_diagram = ansatz.draw(output='mpl', style={'fontsize': 12, 'subfontsize': 10}, idle_wires=False) # 'text' is default, 'mpl' for nice image

# # Save the diagram to a file
# # Make sure you have matplotlib installed: pip install matplotlib
# try:
#     circuit_diagram.savefig('real_amplitudes_ansatz_diagram.png', bbox_inches='tight')
#     print("Circuit diagram saved as 'real_amplitudes_ansatz_diagram.png'")
# except Exception as e:
#     print(f"Error saving diagram (do you have matplotlib installed?): {e}")
#     # Fallback to text drawing if mpl fails
#     print("\n--- Text representation of the circuit (copy this if image fails) ---")
#     print(ansatz.draw(output='text', idle_wires=False))

# # If you just want to see it in a Jupyter notebook/IPython environment:
# # ansatz.draw(output='mpl')

# import matplotlib
# # Set the backend *before* importing pyplot or any other matplotlib-dependent libraries
# matplotlib.use('Agg') # 'Agg' is a non-interactive backend for PNG/raster graphics

# from qiskit.circuit.library import RealAmplitudes
# from qiskit import QuantumCircuit

# print("Generating detailed circuit diagram...")

# import matplotlib
# # Set the backend *before* importing pyplot or any other matplotlib-dependent libraries
# matplotlib.use('Agg') # 'Agg' is a non-interactive backend for PNG/raster graphics

# from qiskit.circuit.library import RealAmplitudes
# from qiskit import QuantumCircuit

# print("Generating detailed circuit diagram...")

# # Define the parameters for the ansatz
# NUM_QUBITS = 3
# REPS = 2
# ENTANGLEMENT_TYPE = 'circular'

# # Create the RealAmplitudes ansatz
# ansatz = RealAmplitudes(NUM_QUBITS, reps=REPS, entanglement=ENTANGLEMENT_TYPE)

# # *** IMPORTANT: Decompose the ansatz to show its internal gates ***
# # This method breaks down the higher-level composite gates into their fundamental components.
# decomposed_ansatz = ansatz.decompose()

# # Draw the decomposed circuit
# try:
#     # Use 'mpl' for high-quality image output
#     # 'style' can be customized further if needed. 'idle_wires=False' removes unused classical bits.
#     circuit_diagram_figure = decomposed_ansatz.draw(output='mpl', style={'fontsize': 10, 'subfontsize': 8}, idle_wires=False)

#     if circuit_diagram_figure:
#         # Save the diagram to a file with high resolution
#         circuit_diagram_figure.savefig('real_amplitudes_ansatz_detailed_diagram.png', bbox_inches='tight', dpi=300)
#         print("Detailed circuit diagram saved successfully as 'real_amplitudes_ansatz_detailed_diagram.png'")
#     else:
#         print("Matplotlib figure object was not returned or is empty after decomposition.")

# except ImportError:
#     print("\nMatplotlib is not installed. Please install it using: pip install matplotlib")
#     print("Cannot generate image diagram. Falling back to text representation.")
#     print("--- Text representation of the circuit (decomposed) ---")
#     print(decomposed_ansatz.draw(output='text', idle_wires=False))
# except Exception as e:
#     print(f"\nAn error occurred during detailed diagram generation/saving: {e}")
#     print("This might be due to a missing or misconfigured Matplotlib dependency or other rendering issues.")
#     print("Falling back to text representation.")
#     print("--- Text representation of the circuit (decomposed) ---")
#     print(decomposed_ansatz.draw(output='text', idle_wires=False))

# print("\nProcessing complete.")


import matplotlib
# Set the backend *before* importing pyplot or any other matplotlib-dependent libraries
matplotlib.use('Agg') # 'Agg' is a non-interactive backend for PNG/raster graphics

from qiskit.circuit.library import RealAmplitudes
from qiskit import QuantumCircuit

print("Generating detailed circuit diagram with custom colors...")

# Define the parameters for the ansatz
NUM_QUBITS = 3
REPS = 2
ENTANGLEMENT_TYPE = 'circular'

# Create the RealAmplitudes ansatz
ansatz = RealAmplitudes(NUM_QUBITS, reps=REPS, entanglement=ENTANGLEMENT_TYPE)

# Decompose the ansatz to show its internal gates
decomposed_ansatz = ansatz.decompose()

# Define custom style for the diagram
custom_style = {
    'fontsize': 10,
    'subfontsize': 8,
    'gate_fill_color': '#ADD8E6',  # Light Blue color for gate blocks (hex code)
    'backgroundcolor': '#FFFFFF',  # White background for the entire figure
    'line_color': '#000000',      # Black lines for wires and connections
    'displaytext': {'id': False} # Optionally hide gate IDs if they clutter
}

# Draw the decomposed circuit with the custom style
try:
    circuit_diagram_figure = decomposed_ansatz.draw(
        output='mpl',
        style=custom_style, # Apply the custom style here
        idle_wires=False
    )

    if circuit_diagram_figure:
        # Save the diagram to a file with high resolution
        circuit_diagram_figure.savefig('real_amplitudes_ansatz_detailed_colored_diagram.png', bbox_inches='tight', dpi=300)
        print("Detailed circuit diagram with custom colors saved successfully as 'real_amplitudes_ansatz_detailed_colored_diagram.png'")
    else:
        print("Matplotlib figure object was not returned or is empty after decomposition.")

except ImportError:
    print("\nMatplotlib is not installed. Please install it using: pip install matplotlib")
    print("Cannot generate image diagram. Falling back to text representation.")
    print("--- Text representation of the circuit (decomposed) ---")
    print(decomposed_ansatz.draw(output='text', idle_wires=False))
except Exception as e:
    print(f"\nAn error occurred during detailed diagram generation/saving: {e}")
    print("This might be due to a missing or misconfigured Matplotlib dependency or other rendering issues.")
    print("Falling back to text representation.")
    print("--- Text representation of the circuit (decomposed) ---")
    print(decomposed_ansatz.draw(output='text', idle_wires=False))

print("\nProcessing complete.")