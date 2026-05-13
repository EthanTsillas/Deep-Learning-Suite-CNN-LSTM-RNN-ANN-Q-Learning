from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridWorld:
    size: int = 5
    obstacles: tuple[tuple[int, int], ...] = ((1, 1), (2, 3), (3, 1))
    max_steps: int = 50

    def __post_init__(self) -> None:
        self.goal = (self.size - 1, self.size - 1)
        self.reset()

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def to_state(self, position: tuple[int, int]) -> int:
        return position[0] * self.size + position[1]

    def reset(self) -> int:
        self.position = (0, 0)
        self.steps = 0
        return self.to_state(self.position)

    def step(self, action: int) -> tuple[int, float, bool]:
        row, col = self.position
        if action == 0:
            row -= 1
        elif action == 1:
            col += 1
        elif action == 2:
            row += 1
        elif action == 3:
            col -= 1
        else:
            raise ValueError("action must be between 0 and 3")

        row = int(np.clip(row, 0, self.size - 1))
        col = int(np.clip(col, 0, self.size - 1))
        self.position = (row, col)
        self.steps += 1

        if self.position == self.goal:
            return self.to_state(self.position), 10.0, True
        if self.position in self.obstacles:
            return self.to_state(self.position), -10.0, True
        if self.steps >= self.max_steps:
            return self.to_state(self.position), -2.0, True
        return self.to_state(self.position), -0.1, False


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 1.0,
    ) -> None:
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def act(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


def train_q_learning(episodes: int = 400, epsilon_decay: float = 0.99, min_epsilon: float = 0.05) -> dict:
    env = GridWorld()
    agent = QLearningAgent(env.n_states, env.n_actions)
    rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

    policy = np.argmax(agent.q_table, axis=1).reshape(env.size, env.size)
    return {
        "q_table": agent.q_table,
        "policy": policy,
        "rewards": rewards,
        "average_last_50_reward": float(np.mean(rewards[-50:])),
    }
