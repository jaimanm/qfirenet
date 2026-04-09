import pennylane as qml


def create_qufex_circuit(config):
    """QuFeX circuit — Qu-Net 4(2) configuration (Jain & Kalev, 2025).

    Changes from previous implementation, cited from paper:

    1. Encoding: Hadamard + RZ (X-basis angle encoding)
       Paper: "the input data is encoded using RZ gates with angle encoding
       in the X basis"
       Previous code applied RY + RZ to inputs — RZ-in-X-basis means H then RZ.

    2. U1 trainable gates: RX + RZ
       Paper (Figure 14/16): U1 = RX(θ₁) + RZ(θ₂)
       Previous code used RY + RZ for U1.

    3. U2 trainable gates: RX + RY
       Paper (Figure 14/16): U2 = RX(θ₃) + RY(θ₄)
       Previous code used RY + RZ for U2.

    4. Entanglement: CZ gates
       Paper: "The pooling gates (Vⱼ's) are control-Z gates in all implementations."
       Previous code used CNOT.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get("n_qubits", 4)
    n_layers = config.get("n_layers", 2)

    assert n_qubits % 2 == 0, "n_qubits must be even for the U1/U2 split."
    half = n_qubits // 2

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights_u1, weights_u2):
        # ---- Encoding: X-basis angle encoding (H + RZ) ----
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(inputs[i], wires=i)

        # ---- Variational layers ----
        for layer in range(n_layers):
            # U1 block: first half of qubits — RX + RZ
            for i in range(half):
                qml.RX(weights_u1[layer, i, 0], wires=i)
                qml.RZ(weights_u1[layer, i, 1], wires=i)

            # U2 block: second half of qubits — RX + RY
            for i in range(half):
                qml.RX(weights_u2[layer, i, 0], wires=half + i)
                qml.RY(weights_u2[layer, i, 1], wires=half + i)

            # ---- Entanglement: CZ ring ----
            for i in range(n_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[n_qubits - 1, 0])

        # ---- Pauli-Z measurements ----
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights_u1": (n_layers, half, 2),  # RX + RZ per qubit per layer
        "weights_u2": (n_layers, half, 2),  # RX + RY per qubit per layer
    }
    return circuit, weight_shapes
