
import numpy as np

from load_connectome import load_connectome
from neuron_dynamics import WormNervousSystem

# Anterior ("front") gentle-touch sensory neurons
ANTERIOR_TOUCH_NEURONS = ["ALML", "ALMR", "AVM"]

# Posterior ("back") gentle-touch sensory neurons
POSTERIOR_TOUCH_NEURONS = ["PLML", "PLMR", "PVM"]

# Command interneurons (included for reference / future debugging, not
# required for reading out motor output directly)
BACKWARD_COMMAND_INTERNEURONS = ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"]
FORWARD_COMMAND_INTERNEURONS = ["AVBL", "AVBR", "PVCL", "PVCR"]

# Motor neuron classes. Real names in the dataset are things like
# "VA1".."VA12", "DA1".."DA9", "VB1".."VB11", "DB1".."DB7" — we match by
# prefix rather than hardcoding every numbered instance.
BACKWARD_MOTOR_PREFIXES = ("VA", "DA")
FORWARD_MOTOR_PREFIXES = ("VB", "DB")


def classify_motor_neurons(all_neuron_names):
    forward_motor = [n for n in all_neuron_names if n.startswith(FORWARD_MOTOR_PREFIXES)]
    backward_motor = [n for n in all_neuron_names if n.startswith(BACKWARD_MOTOR_PREFIXES)]
    return forward_motor, backward_motor


def stimulate(net: WormNervousSystem, sensory_neurons: list[str], strength: float = 2.0,
              stim_steps: int = 5, settle_steps: int = 2) -> np.ndarray:
    
    external = np.zeros(net.n, dtype=np.float32)
    for name in sensory_neurons:
        if name in net.index:
            external[net.index[name]] = strength
        else:
            print(f"  [warning] sensory neuron {name} not found in graph — skipping")

    for t in range(stim_steps):
        net.step(external)
    for t in range(settle_steps):
        net.step(None)

    return net.activation.copy()


def motor_drive(net: WormNervousSystem, activation: np.ndarray, motor_neurons: list[str]) -> float:
    """Sum of |activation| across a set of motor neurons — our 'how hard is this circuit firing' signal."""
    total = 0.0
    for name in motor_neurons:
        if name in net.index:
            total += abs(activation[net.index[name]])
    return total


def run_touch_trials():
    graph = load_connectome()
    forward_motor, backward_motor = classify_motor_neurons(list(graph.nodes()))
    print(f"Forward motor neurons found:  {len(forward_motor)}")
    print(f"Backward motor neurons found: {len(backward_motor)}\n")

    # --- Trial 1: touch the front ---
    net = WormNervousSystem(graph, decay=0.7, gain=0.3)
    activation = stimulate(net, ANTERIOR_TOUCH_NEURONS)
    fwd = motor_drive(net, activation, forward_motor)
    bwd = motor_drive(net, activation, backward_motor)
    print("Anterior (front) touch:")
    print(f"  forward-motor drive  = {fwd:.3f}")
    print(f"  backward-motor drive = {bwd:.3f}")
    print(f"  -> {'BACKWARD (correct)' if bwd > fwd else 'FORWARD (unexpected)'}\n")

    # --- Trial 2: touch the back ---
    net = WormNervousSystem(graph, decay=0.7, gain=0.3)
    activation = stimulate(net, POSTERIOR_TOUCH_NEURONS)
    fwd = motor_drive(net, activation, forward_motor)
    bwd = motor_drive(net, activation, backward_motor)
    print("Posterior (back) touch:")
    print(f"  forward-motor drive  = {fwd:.3f}")
    print(f"  backward-motor drive = {bwd:.3f}")
    print(f"  -> {'FORWARD (correct)' if fwd > bwd else 'BACKWARD (unexpected)'}")


if __name__ == "__main__":
    run_touch_trials()