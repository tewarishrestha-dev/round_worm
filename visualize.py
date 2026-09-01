
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering -- no display needed to save a GIF
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from load_connectome import load_connectome
from neuron_dynamics import WormNervousSystem
from sensory_motor import ANTERIOR_TOUCH_NEURONS, POSTERIOR_TOUCH_NEURONS
from worm_environment import WormAgent, WORLD_WIDTH, WORLD_HEIGHT

N_STEPS = 90
OUTPUT_PATH = "worm_demo.gif"


def build_touch_schedule(n_steps: int):

    schedule = [None] * n_steps
    for t in range(5, 12):
        schedule[t] = "anterior"
    for t in range(40, 47):
        schedule[t] = "posterior"
    return schedule


def main():
    graph = load_connectome()
    net = WormNervousSystem(graph, decay=0.7, gain=0.3)
    worm = WormAgent(net)
    schedule = build_touch_schedule(N_STEPS)

    # Compute the connectome graph layout ONCE up front -- recomputing a
    # force-directed layout every frame would be slow and would also make
    # the graph visually "jitter" instead of staying in a stable shape.
    print("Computing graph layout (one-time cost)...")
    layout = nx.spring_layout(graph, seed=42, k=0.3)

    fig, (ax_world, ax_brain) = plt.subplots(1, 2, figsize=(14, 7))

    # Precompute the worm's full trajectory + activation history up front,
    # so the animation just replays it -- simpler and more reliable than
    # trying to step the simulation live inside the animation callback.
    positions = []
    activations = []
    states = []
    for t in range(N_STEPS):
        touch = schedule[t]
        worm.step(touch)
        positions.append((worm.x, worm.y))
        activations.append(net.activation.copy())
        states.append(worm.state)

    def draw_frame(t):
        ax_world.clear()
        ax_brain.clear()

        # --- Left: worm world ---
        xs = [p[0] for p in positions[: t + 1]]
        ys = [p[1] for p in positions[: t + 1]]
        ax_world.plot(xs, ys, "-", color="gray", alpha=0.5, linewidth=1)
        color = {"forward": "tab:green", "backward": "tab:red", "idle": "tab:blue"}[states[t]]
        ax_world.plot(positions[t][0], positions[t][1], "o", color=color, markersize=14)
        ax_world.set_xlim(0, WORLD_WIDTH)
        ax_world.set_ylim(0, WORLD_HEIGHT)
        ax_world.set_title(f"Worm position — state: {states[t]}  (t={t})")
        ax_world.set_aspect("equal")

        # --- Right: connectome activity ---
        act = activations[t]
        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            a = abs(act[net.index[node]])
            node_colors.append(a)
            node_sizes.append(15 + 200 * min(a, 1.0))
        nx.draw_networkx_edges(graph, layout, ax=ax_brain, alpha=0.05, arrows=False)
        nx.draw_networkx_nodes(
            graph, layout, ax=ax_brain,
            node_color=node_colors, node_size=node_sizes,
            cmap="hot", vmin=0, vmax=1,
        )
        ax_brain.set_title("Real connectome — live activation")
        ax_brain.axis("off")

    print(f"Rendering {N_STEPS} frames...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=N_STEPS, interval=100)
    anim.save(OUTPUT_PATH, writer=animation.PillowWriter(fps=10))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()