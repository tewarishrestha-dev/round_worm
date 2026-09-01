
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from load_connectome import load_connectome
from neuron_dynamics import WormNervousSystem
from worm_environment import WormAgent, WORLD_WIDTH, WORLD_HEIGHT

app = FastAPI(title="Digital Worm API")

# --- Simulation state (single global instance -- fine for a portfolio demo,
# would need per-session state for multi-user production use) ---
_graph = load_connectome()
_net = WormNervousSystem(_graph, decay=0.7, gain=0.3)
_worm = WormAgent(_net)


class StepRequest(BaseModel):
    touch: str | None = None  # "anterior", "posterior", or None


class StateResponse(BaseModel):
    x: float
    y: float
    state: str
    world_width: float
    world_height: float
    active_neuron_count: int
    top_active_neurons: list[str]  # names of the most active neurons this step


def _current_state() -> StateResponse:
    act = np.abs(_net.activation)
    active_count = int((act > 0.05).sum())

    top_idx = np.argsort(act)[::-1][:8]
    top_names = [_worm.net.neurons[i] for i in top_idx if act[i] > 0.01]

    return StateResponse(
        x=_worm.x,
        y=_worm.y,
        state=_worm.state,
        world_width=WORLD_WIDTH,
        world_height=WORLD_HEIGHT,
        active_neuron_count=active_count,
        top_active_neurons=top_names,
    )


@app.get("/state", response_model=StateResponse)
def get_state():
    """Read the current simulation state without advancing it."""
    return _current_state()


@app.post("/step", response_model=StateResponse)
def step(req: StepRequest):
    """Advance the simulation by one tick, optionally with a touch stimulus."""
    _worm.step(req.touch)
    return _current_state()


@app.post("/reset", response_model=StateResponse)
def reset():
    """Reset the worm to its starting position and clear network activation."""
    global _worm, _net
    _net = WormNervousSystem(_graph, decay=0.7, gain=0.3)
    _worm = WormAgent(_net)
    return _current_state()


@app.get("/health")
def health():
    return {"status": "ok", "neurons": _net.n}