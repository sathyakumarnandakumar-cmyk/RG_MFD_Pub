# Reinforcement Learning in RG_MFD: Complete Architecture Guide

This document is the definitive technical guide for understanding and implementing Reinforcement Learning (RL) algorithms in the `RG_MFD` framework. It consolidates the physics of the simulation, the Markov Decision Process (MDP) formulation, the event-driven Semi-Markov (SMDP) callback architecture, and the highly-parallelised asynchronous `TorchRL` integration.

---

## Part 1: The Physical World & MDP Formulation

### 1. The Macroscopic Fundamental Diagram (MFD)
RG_MFD models a 7-region macroscopic urban road network. Instead of simulating individual road links, traffic dynamics are governed by the **Macroscopic Fundamental Diagram (MFD)**. A region's traffic speed is a quadratic function of its total vehicle accumulation `n`:

`speed(n) = aMFD * n² + bMFD * n + cMFD`

If `n` hits the jam density (`njamMFD = 15660`), speed drops to 0 mph (gridlock). Connected Autonomous Vehicles (CAVs) travel more efficiently than Human-Driven Vehicles (HDVs), shifting the MFD production curve upwards.

### 2. Dual Fleet Architecture and Behavior Profiles
The CAV population (default 60% of all traffic) is split into two independent fleets based on the `fleet1_ratio` parameter. Each fleet uses its own independent Q-network and can be assigned a distinct **behavior profile** (e.g., Selfish, Altruistic, Malicious).

### 3. State Space (Observations)
Every time a CAV makes a routing decision, it receives a **16-dimensional observation vector**:
- `obs[0]`: Current Region (integer 1-7)
- `obs[1]`: Destination Region (integer 1-7)
- `obs[2:9]`: Macroscopic accumulation `n_r(t)` for all 7 regions (normalized)
- `obs[9:16]`: Boolean Action Mask (1 if a region is physically adjacent and logically reachable)

### 4. Action Space & Feasibility
The action space is **Discrete(7)**. However, the `feasible_next_regions` utility masks out regions that are not physically adjacent or don't lead to the destination. The TorchRL neural network outputs Q-values, and the `action_mask` forces the argmax to only select from valid topological neighbors.

### 5. Reward Signal & Exact Credit Assignment
Unlike older implementations where all CAVs shared a generic global penalty, the modern `iql_torchrl.py` tracks proportional congestion. The reward is calculated precisely for each vehicle using its assigned behavior profile coefficients:

`Reward = (a * own_tt) + (b * my_fleet_congestion) + (c * other_fleet_and_hdv_congestion) + (d * global_congestion)`

| Profile | a | b | c | d | Description |
|---|---|---|---|---|---|
| **Selfish** | `-1.0` | `0.0` | `0.0` | `0.0` | Minimizes only its own travel time. |
| **Altruistic** | `0.0` | `0.0` | `0.0` | `-1.0` | Minimizes global system congestion. |
| **Cooperative** | `0.0` | `-1.0` | `0.0` | `0.0` | Minimizes the delay of its *own* fleet exclusively. |
| **Malicious** | `0.0` | `0.0` | `1.0` | `0.0` | Sole purpose is to maximize traffic jams for the competing fleet. |

---

## Part 2: The Event-Driven Callback Architecture (SMDP)

RG_MFD is an event-driven simulation (a Semi-Markov Decision Process). Decisions happen at variable time intervals (whenever a vehicle crosses a region boundary), not fixed timesteps.

### 1. Inversion of Control
Instead of a standard `env.step(action)` Gymnasium loop where the agent is in control, **the simulation engine owns the time loop**. The RL agent is passed in as a passive plugin.

### 2. Reward Propagation (Riemann Sum)
Between two CAV decisions, dozens of other simulation events (departures, arrivals) occur, constantly changing the network congestion. 
To accurately capture the time-integral of congestion, the simulation calls `agent.observe_event_reward(reward, state)` at **every single event**, allowing the agent to accumulate the fractional penalties globally without `N²` array updates.

### 3. Closing the Transition (`active_trajectories`)
*Note: This mechanism resolves the "Trajectory Mixing" bug present in older versions of the repository.*

Because multiple vehicles share the same RL agent, we cannot use a single global transition buffer. The `IQLAgent` maintains a dictionary `self.active_trajectories` keyed by `veh_id`. 
When a CAV makes a decision:
1. **Close Previous**: The agent looks up the exact global time and global congestion accumulators from when this specific vehicle started its leg. It subtracts them from the current accumulators to calculate the exact `(S, A, R, S')` transition tailored perfectly to this vehicle's specific holding time and behavior profile.
2. **Push to Replay Buffer**: The finalized transition is pushed to the TorchRL buffer.
3. **Open New**: The agent runs a forward pass to pick the next action, and records the current state and accumulators under `active_trajectories[veh_id]`.

---

## Part 3: Asynchronous Background Training (TorchRL)

Training PyTorch graphs synchronously during the simulation loop is too slow. `iql_torchrl.py` solves this via genuine multi-threading.

1. **Background Daemon Thread**: The PyTorch optimizer runs in a separate `threading.Thread`. Every `update_interval=4` decisions, it wakes up, samples a batch of 64 transitions, computes the `DQNLoss`, and calls `optimizer.step()`.
2. **Shared Memory Weights**: The training thread and the simulation thread share the exact same PyTorch `nn.Module` weight tensors in RAM. When `optimizer.step()` mutates the weights, the simulation thread's next `select_action()` forward-pass instantly sees the updated intelligence.
3. **Thread Safety**: Because PyTorch releases the Python GIL during heavy tensor math (C++/BLAS), both threads run truly parallel on the CPU without locking overhead.

---

## Part 4: The Gridlock Problem (The "Silent Freeze")

A major caveat when training macroscopic RL is the "Silent Freeze". 
If an exploring RL policy routes too many cars into a single region, it hits `njamMFD` (100% full). The MFD physics dictate that traffic speed drops to exactly `0 mph` and outflow drops to `0`. 
Because cars are moving at 0 mph, they never reach a boundary to trigger another RL decision. The simulation clock spins forward endlessly calculating a 0-mph traffic jam.

**The Solution:** The engine enforces a hard event limit via `--max-events 300000`. A normal episode takes ~165,000 events. If the limit is hit, the simulation forces a break and ends the episode, saving the experiment from an infinite loop.

---

## Part 5: RL Algorithms Roadmap & Architectures

The current `iql_torchrl.py` is a **Fully Decentralised (DIAL)** architecture using Independent Q-Learning. Every CAV observes its local state and learns independently using shared parameters.

### Next Steps for Implementation:
1. **MAPPO / QMIX (CTDE)**: Centralised Training, Decentralised Execution. During training, a centralized critic evaluates the full global state (all 7 region accumulations + all agent intents). During execution, agents only use their 16-dim local observation.
2. **Graph Neural Networks (GNN)**: Replace the flat 16-dim MLP input with a GNN that encodes the physical topology of the road network, allowing the trained policy to transfer to larger, unseen cities (e.g., a 19-region map) without retraining.
3. **SMDP Discounting**: Update the TD-target calculation to use `discount = γ^τ` (where `τ` is the actual holding time) instead of a fixed `0.99` per step, to properly account for variable travel times between macro-regions.

### Key Hyperparameters Recap
- `annealing_num_steps = 1,500,000`: Epsilon exhausts over ~100 episodes, ensuring long-term exploration. (Resolves the early-exhaustion bug from previous versions).
- `lr = 3e-4`, `gamma = 0.99`, `tau = 0.005` (soft target update).
- `batch_size = 64`, `update_interval = 4`.
