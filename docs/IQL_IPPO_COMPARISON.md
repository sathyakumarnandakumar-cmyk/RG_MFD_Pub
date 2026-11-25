# IQL vs. IPPO: Algorithm Comparison and Technical Specs

This guide documents the two Reinforcement Learning architectures currently integrated into the RG_MFD (Macroscopic Fundamental Diagram) traffic routing simulation. 

Because a single 2-hour traffic episode generates upwards of **34,000 decision events**, these algorithms required custom adaptations to ensure they could train efficiently on CPUs without causing catastrophic memory spikes.

---

## 1. IQL (Independent Q-Learning)
**File:** `scripts/iql_torchrl.py`

IQL is an **Off-Policy** value-based algorithm. Because it is off-policy, it can learn from old historical data. 

### Algorithmic Explanation (For Beginners)
Imagine teaching a dog a trick. In Q-Learning, the agent builds a "cheat sheet" (the Q-table, or in our case, a Neural Network). For every situation (State) it finds itself in, the cheat sheet outputs a score (Q-Value) estimating how much total reward it will get for taking each possible action. 

The agent looks at the scores, picks the action with the highest score, sees what actually happens, and then updates its cheat sheet based on whether the result was better or worse than expected. It is "Off-Policy" because it can learn by reviewing its old cheat sheets and past mistakes (historical data) at any time.

### How it works natively:
It utilizes a historical Replay Buffer. At every step in the simulation, the agent throws its new experience `(State, Action, Reward, Next State)` into the buffer. During the simulation, the agent constantly pauses to grab a tiny random handful (minibatch) of 64 past experiences and trains its network on them. 

Because the CPU only ever has to calculate gradients for a matrix of 64 rows, IQL trains extremely fast out of the box.

### Technical Specs (IQL)
* **Architecture:** Deep Q-Network (Single MLP predicting Q-Values for all 7 actions).
* **Buffer:** `TensorDictReplayBuffer` with `LazyTensorStorage`.
* **Update Frequency:** Every 4 simulation steps.
* **Batch Size:** 64 transitions.
* **Loss Function:** `DQNLoss` (Temporal Difference Error).
* **Learning Rate:** `3e-4`
* **Discount Factor ($\gamma$):** `0.99`

---

## 2. IPPO (Independent Proximal Policy Optimization)
**File:** `scripts/ippo_torchrl.py`

IPPO is an **On-Policy** Actor-Critic algorithm. It cannot use a historical Replay Buffer because the math requires that it calculates gradients using the *exact probabilities* of its current "brain state". 

### Algorithmic Explanation (For Beginners)
Unlike Q-Learning (which learns the "value" of actions), PPO has two separate brains:
1. **The Actor:** Its job is to directly choose what action to take (the Policy). 
2. **The Critic:** Its job is to judge how good the current situation is (the Value).

When the Actor makes a move, the Critic looks at the result and calculates an **"Advantage"** (Did that action go better or worse than the Critic expected?). If it went better, the Actor's brain is tweaked to make that action more likely in the future. If it went worse, it is tweaked to make it less likely. 

It is "On-Policy" because the Critic can only judge the Actor's *current* behavior. It cannot learn from old historical data because the Actor's brain has already changed since then.

### The Scale Problem
Standard PPO waits until the episode finishes to calculate its gradients. In an environment with 34,000 steps per episode, attempting to pass a `34000 x 16` matrix through PyTorch forward and backward passes 10 times in a row will completely lock up a CPU.

### The Solution: Intra-Episode Bootstrapping
We engineered `ippo_torchrl.py` to use continuous **Intra-Episode Updates** and **Minibatching** to bring its speed up to par with IQL:
1. **Rolling Buffer:** The agent acts until it collects `2048` new steps.
2. **Bootstrapping:** It pauses the simulation. It asks the Critic network to guess the value of the 2049th step, allowing it to calculate accurate Temporal Difference targets without waiting for the terminal state.
3. **Minibatching:** It slices the 2048 steps into tiny chunks of `256` rows.
4. **Optimization:** It passes these tiny chunks through the PPO loss function for 10 epochs.
5. It deletes the data and resumes the simulation.

### Technical Specs (IPPO)
* **Architecture:** Actor-Critic
  * **Actor:** `ProbabilisticActor` using a `Categorical` distribution to output action probabilities.
  * **Critic:** `ValueOperator` predicting the scalar state-value $V(s)$.
* **Advantage Estimation:** Generalized Advantage Estimation (`GAE`) with $\lambda = 0.95$.
* **Update Frequency:** Every `2048` simulation steps.
* **Minibatch Size:** `256` transitions.
* **PPO Epochs:** 10 epochs per update.
* **Loss Function:** `ClipPPOLoss` with a clip epsilon of `0.2` and an Entropy Coefficient of `0.01` to encourage exploration.
* **Learning Rate:** `3e-4`

---

## Shared Environment Specs
Regardless of the algorithm, the underlying environment interface remains identical:
* **Observation Space:** 16 dimensions `[Current Region (1), Destination (1), Macroscopic Regional Occupancies (7), Valid Action Mask (7)]`.
* **Action Space:** 7 Discrete Actions (representing the adjacent regions a Connected Autonomous Vehicle can be routed to).
* **Checkpointing:** The training scripts automatically save Neural Network weights (`.pt` files) and generate a `losses.png` plot every 10 episodes into the `results/` folder to prevent data loss during multi-day runs.
