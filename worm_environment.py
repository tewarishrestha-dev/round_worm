
import numpy as np

from load_connectome import load_connectome
from neuron_dynamics import WormNervousSystem
from sensory_motor import (
    ANTERIOR_TOUCH_NEURONS,
    POSTERIOR_TOUCH_NEURONS,
)

# --- Environment ---
WORLD_WIDTH = 100.0
WORLD_HEIGHT = 100.0
FORWARD_SPEED = 1.5
BACKWARD_SPEED = 1.0
TOUCH_RADIUS = 3.0  # how close an obstacle needs to be to count as "touching" the worm


class WormAgent:
    def __init__(self, net: WormNervousSystem):
        self.net = net
        self.x = WORLD_WIDTH / 2
        self.y = WORLD_HEIGHT / 2
        self.heading = 0.0  # radians; 0 = facing +x
        self.state = "idle"  # "idle" | "forward" | "backward"
        self.history = [(self.x, self.y)]

    def sense_touch(self, obstacle_x: float, obstacle_y: float) -> str | None:
 
        dx = obstacle_x - self.x
        dy = obstacle_y - self.y
        dist = np.hypot(dx, dy)
        if dist > TOUCH_RADIUS:
            return None

        # Project the obstacle position onto the worm's heading direction to
        # decide front vs back.
        forward_component = dx * np.cos(self.heading) + dy * np.sin(self.heading)
        return "anterior" if forward_component > 0 else "posterior"

    def step(self, touch: str | None):

        if touch == "anterior":
            self.net.step(self._external_input(ANTERIOR_TOUCH_NEURONS))
            self.state = "backward"
        elif touch == "posterior":
            self.net.step(self._external_input(POSTERIOR_TOUCH_NEURONS))
            self.state = "forward"
        else:
            self.net.step(None)
            if self.state not in ("forward", "backward"):
                self.state = "idle"

        speed = 0.0
        if self.state == "forward":
            speed = FORWARD_SPEED
        elif self.state == "backward":
            speed = -BACKWARD_SPEED

        self.x += speed * np.cos(self.heading)
        self.y += speed * np.sin(self.heading)

        # Keep the worm inside the world bounds (simple clamp + turn, not
        # real physics -- fine for this scope).
        if not (0 <= self.x <= WORLD_WIDTH) or not (0 <= self.y <= WORLD_HEIGHT):
            self.x = np.clip(self.x, 0, WORLD_WIDTH)
            self.y = np.clip(self.y, 0, WORLD_HEIGHT)
            self.heading += np.pi / 2  # bounce-ish turn

        self.history.append((self.x, self.y))

    def _external_input(self, sensory_neurons: list[str]) -> np.ndarray:
        external = np.zeros(self.net.n, dtype=np.float32)
        for name in sensory_neurons:
            if name in self.net.index:
                external[self.net.index[name]] = 2.0
        return external


def run_demo(steps: int = 60):
    graph = load_connectome()
    net = WormNervousSystem(graph, decay=0.7, gain=0.3)
    worm = WormAgent(net)

    # A fixed "obstacle" the worm will bump into from the front around
    # step 10, demonstrating the withdrawal reflex.
    obstacle = (worm.x + 2, worm.y)  # within TOUCH_RADIUS=3.0, unlike the +5 that shipped in the first version

    for t in range(steps):
        touch = worm.sense_touch(*obstacle) if t < 15 else None
        worm.step(touch)

        if t % 10 == 0 or touch is not None:
            active = int((np.abs(net.activation) > 0.05).sum())
            print(f"t={t:2d}  pos=({worm.x:6.2f}, {worm.y:6.2f})  "
                  f"state={worm.state:8s}  touch={str(touch):10s}  active_neurons={active}")

    print(f"\nFinal position: ({worm.x:.2f}, {worm.y:.2f})")
    print(f"Path length (points recorded): {len(worm.history)}")


if __name__ == "__main__":
    run_demo()