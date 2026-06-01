import pennylane as qml


def create_strongly_entangling_circuit(config):
    """AngleEmbedding + StronglyEntanglingLayers circuit.

    Uses AngleEmbedding (one rotation gate per qubit) instead of
    AmplitudeEmbedding so that:
      - The circuit input size equals n_qubits (not 2^n_qubits), matching
        the SE-net channel-attention bottleneck in the model.
      - No input normalisation is required, removing the near-zero amplitude
        states that caused NaN gradients with the adjoint method.

    Uses parameter-shift differentiation instead of adjoint because
    parameter-shift is unconditionally numerically stable (finite differences
    on the quantum circuit), whereas adjoint can produce NaNs for certain
    quantum states during backpropagation.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get('n_qubits', 8)
    n_layers = config.get('n_layers', 2)
    dev = qml.device('lightning.gpu', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs, weights):
        # AngleEmbedding encodes n_qubits features as RY rotations — no
        # normalisation, no padding, no near-zero amplitude pitfalls.
        qml.AngleEmbedding(features=inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    }
    return circuit, weight_shapes
