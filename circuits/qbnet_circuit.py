import pennylane as qml


def create_qbnet_circuit(config):
    """QB-Net quantum bottleneck circuit — Xu, Zhao & Li (2025), HiQC paper.

    Published in: Quantum Machine Intelligence 7, 97 (2025).
    DOI: https://doi.org/10.1007/s42484-025-00321-0

    Architectural differences from QuFeX (Jain & Kalev 2025):

    1. Encoding: RY angle embedding (no Hadamard prefix)
       QuFeX used X-basis encoding (H + RZ per qubit).

    2. Trainable gates: U3(θ, φ, λ) on ALL qubits — full single-qubit freedom.
       QuFeX split qubits into U1 (RX+RZ) and U2 (RX+RY) halves.

    3. Entanglement: nearest-neighbour CNOT ladder (open chain).
       QuFeX used a CZ ring with wrap-around (qubit[n-1] → qubit[0]).

    4. Symmetric ansatz: same gate family on every qubit (no half-split).
       Weight tensor is a single (n_layers, n_qubits, 3) array.

    Parameters
    ----------
    config : dict
        n_qubits : int   Number of qubits. Default 4.
        n_layers : int   Number of variational layers. Default 2.

    Returns
    -------
    circuit : qml.QNode
        PennyLane QNode with signature (inputs, weights).
        inputs  — shape [n_qubits, B*H*W], one feature per qubit (transposed).
        weights — shape [n_layers, n_qubits, 3].
    weight_shapes : dict
        {"weights": (n_layers, n_qubits, 3)}
    """
    n_qubits = config.get("n_qubits", 4)
    n_layers = config.get("n_layers", 2)

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):
        # ------------------------------------------------------------------ #
        # Encoding: RY angle embedding                                         #
        # inputs shape delivered from _quantum_forward: [n_qubits, B*H*W]     #
        # PennyLane broadcasts over the batch dimension automatically.         #
        # ------------------------------------------------------------------ #
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

        # ------------------------------------------------------------------ #
        # Variational layers                                                   #
        # ------------------------------------------------------------------ #
        for layer in range(n_layers):
            # U3(θ, φ, λ) = RZ(φ) · RY(θ) · RZ(λ)  (PennyLane convention)
            # Gives full single-qubit rotational freedom with 3 parameters.
            for i in range(n_qubits):
                qml.U3(
                    weights[layer, i, 0],  # theta
                    weights[layer, i, 1],  # phi
                    weights[layer, i, 2],  # lambda
                    wires=i,
                )

            # Nearest-neighbour CNOT ladder (open chain).
            # Unlike QuFeX's CZ ring, there is NO wrap-around connection,
            # so entanglement is strictly local (not global).
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # ------------------------------------------------------------------ #
        # Measurements: Pauli-Z expectation value on every qubit              #
        # ------------------------------------------------------------------ #
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    weight_shapes = {
        # Single unified weight tensor: (layers, qubits, 3 params per U3)
        # Compare: QuFeX used weights_u1 (n_layers, half, 2)
        #                          + weights_u2 (n_layers, half, 2) = same total
        #          QB-Net uses weights  (n_layers, n_qubits, 3)   — 50% more params
        "weights": (n_layers, n_qubits, 3),
    }
    return circuit, weight_shapes


def get_circuit(config):
    """Router used by QBNetBottleneck — mirrors the interface of circuits.py
    in the Jain & Kalev codebase so this file is a drop-in replacement.

    Config key
    ----------
    circuit : str   Must be "qbnet_circuit" (default).
    """
    circuit_name = config.get("circuit", "qbnet_circuit")
    if circuit_name == "qbnet_circuit":
        return create_qbnet_circuit(config)
    raise ValueError(f"Unknown circuit: '{circuit_name}'. "
                     f"Supported: 'qbnet_circuit'.")