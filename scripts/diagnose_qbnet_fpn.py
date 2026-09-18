"""Diagnostics for the QBNet-FPN quantum bottleneck.

Answers the question: "is the quantum circuit actually wired into the model and
learning, or is it a dead/no-op layer that the classical convs + residual are
routing around?"

Run on an environment with PennyLane installed (e.g. the HPC `quantum` env):

    python scripts/diagnose_qbnet_fpn.py
    python scripts/diagnose_qbnet_fpn.py --checkpoint experiments/<run>/best_model.pth

Checks performed
----------------
1. GRADIENT FLOW  — after one forward+backward, do the quantum weights
   (`qbnet_bottleneck.weights`) receive a non-zero gradient? If grad is None or
   ~0, the circuit is disconnected from the loss and cannot learn (BROKEN).
2. INPUT SENSITIVITY — does the circuit produce different outputs for different
   inputs? A circuit stuck at a constant output contributes no information.
3. QUANTUM vs RESIDUAL MAGNITUDE — how large is the quantum branch
   `post_conv(circuit(...))` relative to the residual `x`? If it's a tiny
   fraction, the bottleneck is effectively an identity and the circuit is
   negligible even if technically connected.
4. WEIGHT DRIFT (needs --checkpoint) — did the trained quantum weights move away
   from their `randn*0.01` initialisation? No drift => the circuit never learned.
"""
import argparse
import os
import sys

# Ensure the repo root is importable when run as `python scripts/diagnose_qbnet_fpn.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import get_model


def build_model():
    # mode 4 => 3 input channels (SWIR); matches configs/qbnet_fpn.yaml
    cfg = {'model': 'qbnet_fpn', 'mode': 4, 'n_classes': 2,
           'n_qubits': 4, 'n_layers': 2, 'fpn_out_channels': 128}
    return get_model(cfg), cfg


def check_gradient_flow(model):
    print("\n[1] GRADIENT FLOW to quantum weights")
    model.train()
    x = torch.randn(2, 3, 64, 64)                 # small input -> x5 is 4x4
    y = torch.randint(0, 2, (2, 64, 64))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()

    w = model.qbnet_bottleneck.weights
    g = w.grad
    if g is None:
        print("    grad = None  -> BROKEN: quantum weights are not in the graph.")
        return
    gnorm = g.norm().item()
    print(f"    loss={loss.item():.4f}  weight.grad L2 norm = {gnorm:.3e}")
    print("    -> " + ("OK: gradient reaches the circuit."
                       if gnorm > 1e-9 else
                       "BROKEN: gradient is ~0, circuit cannot learn."))


def check_input_sensitivity(model):
    print("\n[2] CIRCUIT INPUT SENSITIVITY")
    bn = model.qbnet_bottleneck
    a = torch.randn(1, bn.in_channels, 4, 4)
    b = torch.randn(1, bn.in_channels, 4, 4)
    with torch.no_grad():
        out_a = bn(a)
        out_b = bn(b)
    # Difference in the bottleneck output for two different inputs
    diff = (out_a - out_b).abs().mean().item()
    print(f"    mean|out(a) - out(b)| = {diff:.3e}")
    print("    -> " + ("OK: bottleneck responds to input."
                       if diff > 1e-6 else
                       "SUSPECT: output barely changes with input (near-constant)."))


def check_quantum_vs_residual(model):
    print("\n[3] QUANTUM BRANCH vs RESIDUAL MAGNITUDE")
    bn = model.qbnet_bottleneck
    x = torch.randn(2, bn.in_channels, 4, 4)
    with torch.no_grad():
        # Replicate QBNetBottleneck._quantum_forward up to the branch, minus residual
        B, C, H, W = x.shape
        xq = bn.pre_conv(x)
        xq = xq.permute(0, 2, 3, 1).reshape(B * H * W, bn.n_qubits)
        from models.qbnet_model import preprocess_quantum_input
        xq = preprocess_quantum_input(xq)
        result = bn.qnode(xq.cpu().T, bn.weights.cpu())
        xq = torch.stack(result, dim=1).float().reshape(B, H, W, bn.n_qubits).permute(0, 3, 1, 2)
        quantum_branch = bn.post_conv(xq)          # what gets added to the residual
    q = quantum_branch.norm().item()
    r = x.norm().item()
    ratio = q / (r + 1e-12)
    print(f"    ||quantum branch|| = {q:.3e}   ||residual x|| = {r:.3e}   ratio = {ratio:.3f}")
    print("    -> " + ("OK: quantum branch is a non-trivial fraction of the signal."
                       if ratio > 0.05 else
                       "SUSPECT: quantum branch << residual; bottleneck ~ identity."))


def check_weight_drift(model, ckpt_path):
    print("\n[4] WEIGHT DRIFT from initialisation")
    init_w = model.qbnet_bottleneck.weights.detach().clone()
    state = torch.load(ckpt_path, map_location='cpu')
    state = state.get('model_state_dict', state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    trained_w = model.qbnet_bottleneck.weights.detach()
    drift = (trained_w - init_w).abs().mean().item()
    print(f"    init std={init_w.std().item():.3e}  trained std={trained_w.std().item():.3e}")
    print(f"    mean|trained - init| = {drift:.3e}")
    print("    -> " + ("OK: quantum weights moved during training."
                       if drift > 1e-4 else
                       "SUSPECT: weights ~ unchanged; circuit may not have learned."))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='Path to a trained best_model.pth for the drift check.')
    args = ap.parse_args()

    model, cfg = build_model()
    print(f"Model: {cfg['model']}  n_qubits={cfg['n_qubits']} n_layers={cfg['n_layers']}")
    print(f"Quantum weight tensor shape: {tuple(model.qbnet_bottleneck.weights.shape)}")

    check_gradient_flow(model)
    check_input_sensitivity(model)
    check_quantum_vs_residual(model)
    if args.checkpoint:
        check_weight_drift(model, args.checkpoint)
    else:
        print("\n[4] WEIGHT DRIFT: skipped (pass --checkpoint experiments/<run>/best_model.pth)")
