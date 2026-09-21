"""Behavior-cloning mini lab utilities."""

import math

import matplotlib.pyplot as plt
import torch


Point = tuple[float, float]
Velocity = tuple[float, float]


def action_towards_goal(robot: Point, goal: Point) -> Velocity:
    """Return a unit action vector pointing from ``robot`` to ``goal``.

    Args:
        robot: Current position, ``(x, y)``.
        goal: Target position, ``(gx, gy)``.

    Returns:
        ``(vx, vy)`` with magnitude 1.  If the robot is already at the
        goal, returns ``(0.0, 0.0)`` because no direction is needed.
    """
    x, y = robot
    gx, gy = goal

    dx = gx - x
    dy = gy - y
    distance = math.hypot(dx, dy)

    if distance == 0:
        return 0.0, 0.0

    return dx / distance, dy / distance


def make_dataset(
    size: int = 1000,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate inputs and four-step action-chunk outputs."""
    X = torch.rand(size, 4) * 2 - 1

    action_chunks = [
        make_action_chunk((x, y), (gx, gy), dt) for x, y, gx, gy in X.tolist()
    ]
    Y = torch.tensor(
        [[value for action in chunk for value in action] for chunk in action_chunks],
        dtype=X.dtype,
    )

    return X, Y


def make_action_chunk(
    position: Point,
    goal: Point,
    dt: float,
    chunk_size: int = 4,
) -> list[Velocity]:
    """Create a chunk of actions by simulating the updated position."""
    actions: list[Velocity] = []

    for _ in range(chunk_size):
        action = action_towards_goal(position, goal)
        actions.append(action)
        position = robot_dynamics(position, action, dt)

    return actions


class BCPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def robot_dynamics(
    position: Point,
    velocity: Velocity,
    dt: float,
) -> Point:
    x, y = position
    vx, vy = velocity
    return x + vx * dt, y + vy * dt


def calc_dist(
    position1: Point,
    position2: Point,
) -> float:
    x1, y1 = position1
    x2, y2 = position2
    return math.hypot(x1 - x2, y1 - y2)


def rollout(
    policy: BCPolicy,
    start: Point,
    goal: Point,
    dt: float = 0.1,
    max_steps: int = 50,
) -> list[Point]:
    position = start
    trajectory = [position]

    with torch.no_grad():
        for step in range(max_steps):
            # 1. position + goal → observation Tensor
            X = torch.tensor(position + goal, dtype=torch.float32)
            # 2. policy(observation) → four velocities at once
            action_chunk = policy(X).reshape(4, 2).tolist()

            # 3. Apply the four predicted actions in sequence.
            for action_index, action in enumerate(action_chunk):
                position = robot_dynamics(position, action, dt)
                trajectory.append(position)
                distance_to_goal = calc_dist(position, goal)
                print(f"[{step * 4 + action_index}] ({position})")

                if distance_to_goal < 0.1:
                    print("GOAL!")
                    return trajectory

        else:
            print("Reached max steps...")

    return trajectory


if __name__ == "__main__":
    X, Y = make_dataset()
    print(f"X.shape = {tuple(X.shape)}")
    print(f"Y.shape = {tuple(Y.shape)}")

    policy = BCPolicy()
    for name, parameter in policy.named_parameters():
        print(name, parameter.shape)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        optimizer.zero_grad()

        predicted_Y = policy(X)
        loss = criterion(predicted_Y, Y)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch}: loss = {loss.item():.6f}")

    start: Point = (0.0, 0.0)
    goal: Point = (0.8, 0.8)
    trajectory = rollout(policy, start, goal)

    xs, ys = zip(*trajectory)
    plt.plot(xs, ys, "o-", label="trajectory")
    plt.scatter(*start, color="green", s=100, label="start", zorder=3)
    plt.scatter(*goal, color="red", s=100, label="goal", zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Policy rollout")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()
