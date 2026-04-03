from circuits.strongly_entangling import create_strongly_entangling_circuit
from circuits.ry_cnot import create_ry_cnot_circuit
from circuits.qufex_circuit import create_qufex_circuit

CIRCUIT_REGISTRY = {
    'strongly_entangling': create_strongly_entangling_circuit,
    'ry_cnot': create_ry_cnot_circuit,
    'qufex_circuit' : create_qufex_circuit
}


def get_circuit(config):
    """Return (qnode, weight_shapes) for the circuit named in config['circuit']."""
    circuit_name = config.get('circuit', 'strongly_entangling')
    if circuit_name not in CIRCUIT_REGISTRY:
        raise ValueError(f"Unknown circuit '{circuit_name}'. Available: {list(CIRCUIT_REGISTRY.keys())}")
    return CIRCUIT_REGISTRY[circuit_name](config)
