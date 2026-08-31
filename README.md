# Digital Worm — A Real Connectome, Simulated

A functional simulation of *C. elegans* — the roundworm whose entire 302-neuron
nervous system has been fully mapped — built on the actual, published connectome
data rather than a toy or synthetic network.

This is **not** an attempt at biophysical research-grade accuracy (that's what the
[OpenWorm project](http://openworm.org/) has been working on since 2011). The goal
here is a simplified but *real-data-driven* functional simulation: real neurons,
real synapses, a simple dynamics model, and a live visualization — built to be
finished in about a week, not a career.

## Data source

- **Connectome**: White et al. 1986 — the original electron-microscopy
  reconstruction of the *C. elegans* nervous system — as curated and distributed
  by the [OpenWorm ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox)
  (`aconnectome_white_1986_whole.csv`).
- 309 nodes (302 neurons plus a few lumped/motor entries like
  `LegacyBodyWallMuscles`), ~3,500 directed synaptic connections (chemical +
  electrical/gap-junction).
- **Known limitation**: the dataset gives synapse *counts*, not synapse
  *polarity* (excitatory vs. inhibitory). This matters — see below.

## What's built so far

### `load_connectome.py`
Loads the raw connectome into a `networkx` `MultiDiGraph` (a directed multigraph,
since a single neuron pair can have both a chemical *and* an electrical
connection). Validated against known biology: the top-degree hub neurons that
fall out of the real data are **AVAL/AVAR**, which in actual *C. elegans*
research are the command interneurons for backward locomotion — a strong sign
the data is loading correctly, not just producing plausible-looking numbers.

### `neuron_dynamics.py`
Each neuron is modeled as a simplified leaky-integrate unit — a hand-rolled
recurrent update rule where the real connectome *is* the recurrent weight
matrix (row-normalized so high-degree hub neurons don't automatically dominate
just by virtue of having more synapses). Not Hodgkin-Huxley biophysics — a
lightweight stand-in that's tunable via two parameters:
- `decay` — how much a neuron "remembers" its previous activation
- `gain` — how strongly incoming signal drives a neuron

Validated by stimulating a real sensory neuron (`ASHL`) and confirming activity
spreads through the network and decays back down after the stimulus is
released, rather than saturating the whole network or dying out instantly.

### `sensory_motor.py`
Maps real, documented sensory and motor neuron classes onto the network,
implementing the **touch-withdrawal reflex** (Chalfie et al. 1985) — the
best-experimentally-characterized sensorimotor circuit in *C. elegans*:
- Anterior touch (`ALML`, `ALMR`, `AVM`) → should trigger backward locomotion
- Posterior touch (`PLML`, `PLMR`, `PVM`) → should trigger forward locomotion
- Read out via motor neuron classes: `VA*`/`DA*` (backward), `VB*`/`DB*` (forward)

**Honest result**: the simulation currently predicts the *wrong* direction for
both cases. Diagnosis: since the raw connectome data has no excitatory/inhibitory
labels, and real *C. elegans* locomotor switching is known to be governed
substantially by inhibitory connections, a purely-excitatory network cannot
reproduce the correct reflex direction. Getting true polarity would require
pulling in additional neurotransmitter-identity data per neuron — out of scope
for this build.

## Running it

```bash
pip install pandas networkx numpy

python load_connectome.py    # Day 1: load + validate the connectome
python neuron_dynamics.py    # Day 2: run the recurrent dynamics sanity check
python sensory_motor.py      # Day 3: touch-withdrawal reflex trial
```

Requires `connectome.csv` (the White et al. 1986 dataset) in a `data/` subfolder
next to these scripts.

