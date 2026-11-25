import numpy as np
import torch
import torch.nn as nn

from rg_mfd_simplified.utils import get_field_or_attribute


class ActorCritic(nn.Module):
    def __init__(self, observation_size, number_of_regions, hidden_size=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.policy_head = nn.Linear(hidden_size, number_of_regions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations, action_mask):
        features = self.shared(observations)
        logits = self.policy_head(features).masked_fill(action_mask <= 0, -1e9)
        return torch.distributions.Categorical(logits=logits), self.value_head(features).squeeze(-1)


class PPOAgent:
    def __init__(self, observation_size, number_of_regions, learning_rate=3e-4, discount=0.99,
                 gae_lambda=0.95, clip_ratio=0.2, entropy_coefficient=0.01, value_coefficient=0.5,
                 epochs_per_update=4, minibatch_size=256, reward_scale=1e-6, deterministic=False):
        self.number_of_regions = number_of_regions
        self.network = ActorCritic(observation_size, number_of_regions)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.epochs_per_update = epochs_per_update
        self.minibatch_size = minibatch_size
        self.reward_scale = reward_scale
        self.deterministic = deterministic
        self.trajectories = {}
        self.finished_vehicle_ids = set()
        self.last_loss = float("nan")

    def select_action(self, vehicle, state, feasible, action_mask):
        observation = torch.from_numpy(np.asarray(state["observation"], dtype=np.float32))
        mask = observation[-self.number_of_regions:]
        with torch.no_grad():
            distribution, value = self.network(observation, mask)
            action = distribution.probs.argmax() if self.deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        vehicle_id = int(get_field_or_attribute(vehicle, "uid", id(vehicle)))
        self.trajectories.setdefault(vehicle_id, []).append(
            {"observation": observation, "action": int(action), "log_probability": float(log_probability),
             "value": float(value), "reward": 0.0, "done": False})
        return int(action) + 1

    def observe_event_reward(self, reward):
        for steps in self.trajectories.values():
            if steps and not steps[-1]["done"]:
                steps[-1]["reward"] += reward * self.reward_scale

    def observe_arrival(self, vehicle):
        vehicle_id = int(get_field_or_attribute(vehicle, "uid", id(vehicle)))
        steps = self.trajectories.get(vehicle_id)
        if steps:
            steps[-1]["done"] = True
        self.finished_vehicle_ids.add(vehicle_id)

    def end_episode(self):
        batch = self._build_batch()
        self.trajectories = {}
        self.finished_vehicle_ids = set()
        if batch is not None and not self.deterministic:
            self._update(*batch)

    def _build_batch(self):
        observations, actions, log_probabilities, advantages, returns = [], [], [], [], []
        for vehicle_id, steps in self.trajectories.items():
            if not steps:
                continue
            finished = vehicle_id in self.finished_vehicle_ids
            usable = steps if finished else steps[:-1]
            if not usable:
                continue
            bootstrap_value = 0.0 if finished else steps[-1]["value"]
            running_advantage = 0.0
            trajectory_advantages = np.zeros(len(usable))
            for t in reversed(range(len(usable))):
                next_value = usable[t + 1]["value"] if t + 1 < len(usable) else bootstrap_value
                not_done = 0.0 if usable[t]["done"] else 1.0
                delta = usable[t]["reward"] + self.discount * next_value * not_done - usable[t]["value"]
                running_advantage = delta + self.discount * self.gae_lambda * not_done * running_advantage
                trajectory_advantages[t] = running_advantage
            for t, step in enumerate(usable):
                observations.append(step["observation"])
                actions.append(step["action"])
                log_probabilities.append(step["log_probability"])
                advantages.append(trajectory_advantages[t])
                returns.append(trajectory_advantages[t] + step["value"])
        if not observations:
            return None
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return (torch.stack(observations), torch.tensor(actions), torch.tensor(log_probabilities, dtype=torch.float32),
                advantages, torch.tensor(returns, dtype=torch.float32))

    def _update(self, observations, actions, old_log_probabilities, advantages, returns):
        count = observations.shape[0]
        for _ in range(self.epochs_per_update):
            order = torch.randperm(count)
            for start in range(0, count, self.minibatch_size):
                index = order[start:start + self.minibatch_size]
                batch_observations = observations[index]
                mask = batch_observations[:, -self.number_of_regions:]
                distribution, values = self.network(batch_observations, mask)
                ratio = torch.exp(distribution.log_prob(actions[index]) - old_log_probabilities[index])
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                policy_loss = -torch.min(ratio * advantages[index], clipped_ratio * advantages[index]).mean()
                value_loss = nn.functional.mse_loss(values, returns[index])
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                self.last_loss = float(loss.item())

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
