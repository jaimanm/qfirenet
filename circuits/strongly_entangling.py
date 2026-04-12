import pennylane as qml


def create_strongly_entangling_circuit(config):
    """AmplitudeEmbedding + StronglyEntanglingLayers circuit.

    Returns (qnode, weight_shapes) for use with qml.qnn.TorchLayer.
    """
    n_qubits = config.get('n_qubits', 8)
    n_layers = config.get('n_layers', 2)
    dev = qml.device('lightning.gpu', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='parameter-shift')
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits),
                               pad_with=0.0, normalize=True)
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    }
    return circuit, weight_shapes
