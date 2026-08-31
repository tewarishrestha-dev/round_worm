"""
Day 2 — Neuron dynamics: run a leaky integrate-and-fire (LIF) style update
over the real connectome graph from Day 1.

This is deliberately NOT a biophysically accurate Hodgkin-Huxley model
(that's what OpenWorm's own simulator spends years on). The goal here is a
*functional* recurrent network: each neuron has an activation level that
leaks toward zero over time and gets pushed up or down by weighted input
from the neurons that synapse onto it — the connectome literally IS the
recurrent weight matrix.

Think of this as a hand-rolled RNN cell, except the connectivity is real
biology instead of randomly initialized weights.
"""

import numpy as np
import networkx as nx

from load_connectome import load_connectome


class WormNervousSystem:
    def __init__(self, graph: nx.MultiDiGraph, decay: float = 0.8, gain: float = 1.0):
        """
        graph : the connectome graph from Day 1
        decay : how much activation "leaks away" each step (0-1). Higher = more memory.
        gain  : scales how strongly incoming signal affects a neuron. Tune to avoid
                activity exploding (runaway feedback) or dying out completely.
        """
        self.graph = graph
        self.decay = decay
        self.gain = gain

        # Fix a stable ordering of neurons so we can use plain numpy vectors/matrices.
        self.neurons = list(graph.nodes())
        self.index = {name: i for i, name in enumerate(self.neurons)}
        self.n = len(self.neurons)

        # Build a dense weighted adjacency matrix once, up front.
        # W[i, j] = total synaptic weight from neuron i -> neuron j.
        # Chemical and electrical synapses between the same pair both
        # contribute (MultiDiGraph can have multiple edges per pair).
        self.W = np.zeros((self.n, self.n), dtype=np.float32)
        for pre, post, data in graph.edges(data=True):
            i, j = self.index[pre], self.index[post]
            self.W[i, j] += data["weight"]

        # Normalize weights so a highly-connected hub neuron (like AVAL,
        # degree 170) doesn't automatically dominate the whole network just
        # because it has more synapses. This keeps the simulation stable
        # regardless of how connected a given neuron happens to be.
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero for isolated nodes
        self.W_normalized = self.W / row_sums

        self.activation = np.zeros(self.n, dtype=np.float32)

    def step(self, external_input: np.ndarray | None = None) -> np.ndarray:
        """
        Advance the network by one timestep.

        external_input : optional vector of length n, added directly to
                          specific neurons (e.g. sensory stimulus). Pass
                          None for a free-running step with no new input.
        """
        # Signal propagation: each neuron's new incoming drive is the
        # weighted sum of every neuron that synapses onto it, using their
        # CURRENT activation (this is what makes it recurrent).
        incoming = self.activation @ self.W_normalized

        if external_input is not None:
            incoming = incoming + external_input

        # Leaky integrate: old activation decays, new drive is added, then
        # squashed through tanh so activity can't blow up to infinity.
        self.activation = np.tanh(self.decay * self.activation + self.gain * incoming)
        return self.activation

    def neuron_activation(self, name: str) -> float:
        return float(self.activation[self.index[name]])


def sanity_check_run(steps: int = 30, stim_neuron: str = "ASHL", stim_strength: float = 2.0):
    """
    Poke a single sensory-ish neuron with a stimulus and confirm activity
    actually propagates through the real network and settles down rather
    than exploding or instantly dying to zero.
    """
    graph = load_connectome()
    net = WormNervousSystem(graph, decay=0.7, gain=0.3)

    external = np.zeros(net.n, dtype=np.float32)
    external[net.index[stim_neuron]] = stim_strength

    print(f"Stimulating {stim_neuron} for 5 steps, then releasing...\n")
    for t in range(steps):
        stim = external if t < 5 else None
        net.step(stim)

        total_activity = np.abs(net.activation).sum()
        active_neurons = int((np.abs(net.activation) > 0.05).sum())

        if t % 5 == 0 or t == steps - 1:
            print(f"t={t:2d}  total_activity={total_activity:7.3f}  "
                  f"active_neurons={active_neurons:3d}/{net.n}")


if __name__ == "__main__":
    sanity_check_run()