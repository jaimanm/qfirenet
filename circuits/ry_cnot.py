import pennylane as qml


def create_ry_cnot_circuit(config):
    """RY encoding + ring-topology CNOT entanglement circuit.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get('n_qubits', 8)
    n_layers = config.get('n_layers', 2)
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
        # Encode inputs via RY rotations
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers with ring-topology CNOT entanglement
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights": (n_layers, n_qubits)
    }
    return circuit, weight_shapes
