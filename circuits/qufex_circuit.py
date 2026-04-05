import pennylane as qml


def create_qufex_circuit(config):
    """QuFeX angle-embedding circuit with U1/U2 encoding blocks.

    Architecture (Jain & Kalev, 2025):
      • Split n_qubits evenly into two halves.
      • U1 block: angle-embed the first  n/2 input features → first  n/2 qubits.
      • U2 block: angle-embed the second n/2 input features → second n/2 qubits.
      • Each block is followed by trainable RY/RZ rotations (θ parameters).
      • A ring of CNOT gates entangles all qubits after both blocks.
      • Measure ⟨Z⟩ on every qubit → n_qubits scalar outputs.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get("n_qubits", 8)
    n_layers = config.get("n_layers", 2)

    assert n_qubits % 2 == 0, "n_qubits must be even for the U1/U2 split."
    half = n_qubits // 2

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights_u1, weights_u2):
        # ---- U1 block: first half of feature channels ----
        for i in range(half):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)

        # Trainable rotations on U1 qubits
        for layer in range(n_layers):
            for i in range(half):
                qml.RY(weights_u1[layer, i, 0], wires=i)
                qml.RZ(weights_u1[layer, i, 1], wires=i)

        # ---- U2 block: second half of feature channels ----
        for i in range(half):
            qml.RY(inputs[half + i], wires=half + i)
            qml.RZ(inputs[half + i], wires=half + i)

        # Trainable rotations on U2 qubits
        for layer in range(n_layers):
            for i in range(half):
                qml.RY(weights_u2[layer, i, 0], wires=half + i)
                qml.RZ(weights_u2[layer, i, 1], wires=half + i)

        # ---- Entangling layer across all qubits (ring topology) ----
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])  # close the ring

        # ---- Pauli-Z measurements ----
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] 

    weight_shapes = {
        "weights_u1": (n_layers, half, 2),  # θ for U1 qubits (RY + RZ per layer)
        "weights_u2": (n_layers, half, 2),  # θ for U2 qubits (RY + RZ per layer)
    }
    return circuit, weight_shapes