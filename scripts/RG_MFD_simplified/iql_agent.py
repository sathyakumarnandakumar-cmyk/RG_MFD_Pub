import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from rg_mfd_simplified.utils import get_field_or_attribute


def build_q_network(observation_size, number_of_regions, hidden_size=64):
    return nn.Sequential(
        nn.Linear(observation_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, number_of_regions),
    )


def masked_q_values(q_values, action_mask):
    return q_values.masked_fill(action_mask <= 0, -1e9)


class IQLAgent:
    def __init__(self, observation_size, number_of_regions, epsilon=1.0, learning_rate=1e-3,
                 discount=0.99, soft_update_rate=0.005, buffer_size=100_000, batch_size=64,
                 decisions_per_update=4, reward_scale=1e-6):
        self.number_of_regions = number_of_regions
        self.observation_size = observation_size
        self.reward_scale = reward_scale
        self.q_network = build_q_network(observation_size, number_of_regions)
        self.target_network = build_q_network(observation_size, number_of_regions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.discount = discount
        self.soft_update_rate = soft_update_rate
        self.batch_size = batch_size
        self.decisions_per_update = decisions_per_update
        self.replay_buffer = deque(maxlen=buffer_size)
        self.open_transitions = {}
        self.decision_count = 0
        self.last_loss = float("nan")

    def select_action(self, vehicle, state, feasible, action_mask):
        observation = np.asarray(state["observation"], dtype=np.float32)
        vehicle_id = int(get_field_or_attribute(vehicle, "uid", id(vehicle)))
        self._close_open_transition(vehicle_id, observation, done=False)
        if random.random() < self.epsilon:
            region = random.choice(feasible)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.from_numpy(observation))
                q_values = masked_q_values(q_values, torch.from_numpy(observation[-self.number_of_regions:]))
            region = int(q_values.argmax().item()) + 1
        self.open_transitions[vehicle_id] = [observation, region - 1, 0.0]
        self.decision_count += 1
        if self.decision_count % self.decisions_per_update == 0:
            self._train_step()
        return region

    def observe_event_reward(self, reward):
        for transition in self.open_transitions.values():
            transition[2] += reward * self.reward_scale

    def observe_arrival(self, vehicle):
        vehicle_id = int(get_field_or_attribute(vehicle, "uid", id(vehicle)))
        self._close_open_transition(vehicle_id, np.zeros(self.observation_size, dtype=np.float32), done=True)
        self.open_transitions.pop(vehicle_id, None)

    def end_episode(self):
        self.open_transitions.clear()

    def _close_open_transition(self, vehicle_id, next_observation, done):
        transition = self.open_transitions.get(vehicle_id)
        if transition is not None:
            observation, action, reward = transition
            self.replay_buffer.append((observation, action, reward, next_observation, done))

    def _train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        observations = torch.tensor(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_observations = torch.tensor(np.stack([b[3] for b in batch]))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        predicted = self.q_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = masked_q_values(self.target_network(next_observations),
                                     next_observations[:, -self.number_of_regions:])
            target = rewards + self.discount * (1.0 - dones) * next_q.max(dim=1).values
        loss = nn.functional.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.item())

        with torch.no_grad():
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.mul_(1.0 - self.soft_update_rate).add_(param, alpha=self.soft_update_rate)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
