# Technical README

How to install, run and modify the simplified RG-MFD package in `scripts/RG_MFD_simplified/`.

## Requirements

| Item | Needed |
|---|---|
| Python | 3.10 or newer (tested on 3.12) |
| Packages | `numpy`, `scipy`, `numba`, `torch` (see `requirements.txt`) |
| Optional | `h5py` (only for MATLAB v7.3 `.mat` files, not used by the shipped inputs), `matplotlib` (only for an optional demand plot) |
| Hardware | CPU is enough. A full episode takes minutes to hours depending on `--max-events`. About 4 GB of free RAM is comfortable, and keep to at most 2 simultaneous runs on an 8 GB machine. |
| OS | Linux or macOS. Windows should work but is untested. |

## Install

```bash
git clone https://github.com/sathyakumarnandakumar-cmyk/RG_MFD_Pub.git
cd RG_MFD_Pub/scripts/RG_MFD_simplified

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` is the large download. A CPU-only build is fine.

Run every script from inside `scripts/RG_MFD_simplified/`. The network data is in the same folder, so
nothing else needs configuring.

## Run

```bash
python run_greedy.py --episodes 5                     # greedy CAV routing, HDVs on Dijkstra
python train_iql.py  --id my_iql --episodes 50        # train an IQL (DQN) agent
python eval_iql.py   --id my_iql                      # evaluate it
python train_ppo.py  --id my_ppo --episodes 50        # train an IPPO agent (shared policy)
python eval_ppo.py   --id my_ppo
```

A quick check that everything is wired up (a few seconds per script):

```bash
python run_greedy.py --episodes 1 --max-events 2000
python train_iql.py --id smoke --episodes 2 --max-events 2000
```

With so few events the printed travel times only cover the first minutes of simulated time, so do not
read anything into the ratios. The IQL loss prints `nan` until it has collected 64 transitions.

For long runs, start them in `tmux` or `screen` so they survive a closed terminal.

## Command-line flags

| Script | Flag | Default | Meaning |
|---|---|---|---|
| all | `--id` | per script | Run name. Output goes to `results/<id>/`. |
| all | `--demand-scale` | 0.7 | Multiplier on future demand. Higher means more congestion. |
| all | `--max-events` | none (train), 80000 (eval) | Cap on simulation events per episode. Cut-off trips are not trained on. |
| `run_greedy.py`, `train_*` | `--episodes` | 5 (greedy), 50 (train) | Number of episodes. |
| `run_greedy.py`, `train_*` | `--seed` | 42 | Base random seed. Episode `k` uses `seed + k`. |
| `train_iql.py` | `--epsilon-start` / `--epsilon-end` | 1.0 / 0.05 | Exploration schedule. |
| `train_iql.py` | `--epsilon-decay-fraction` | 0.6 | Fraction of episodes over which epsilon falls. |
| `train_iql.py` | `--learning-rate` | 1e-3 | Adam learning rate. |
| `train_ppo.py` | `--learning-rate` | 3e-4 | Adam learning rate. |
| `train_ppo.py` | `--entropy-coefficient` | 0.01 | Entropy bonus weight. |
| `eval_*` | `--first-seed` | 5000 | First evaluation seed. |
| `eval_*` | `--number-of-seeds` | 5 | Seeds 5000 to 5004 by default. |

## Outputs

| File | Written by | Contents |
|---|---|---|
| `results/<id>/episodes.csv` | `run_greedy.py`, `train_*` | Per-episode CAV and HDV mean travel time, vehicle counts, CAV/HDV ratio (plus epsilon and loss for training). |
| `results/<id>/q_network.pt` | `train_iql.py` | IQL Q-network weights. |
| `results/<id>/policy_network.pt` | `train_ppo.py` | IPPO actor-critic weights. |
| `results/<id>/eval_summary.json` | `eval_*` | Per-seed CAV/HDV, all-vehicles over all-Dijkstra, completed and total vehicle counts, and their means. |

The two numbers to look at in an evaluation are:

- `cav_over_hdv`: CAV mean travel time divided by HDV mean travel time in the same run. Below 1 means CAVs are faster than the HDVs around them.
- `all_vehicles_over_dijkstra`: mean travel time of all vehicles divided by the same measure when every vehicle uses Dijkstra on the same seed. Below 1 means the network as a whole improved.

Always check `completed_policy` against `total_vehicles`. A run that strands vehicles can look fast
because only the quick trips finished.

## Layout

```
scripts/RG_MFD_simplified/
  run_greedy.py            greedy lookup agent and runner
  train_iql.py  eval_iql.py  iql_agent.py     DQN-style agent, soft target update
  train_ppo.py  eval_ppo.py  ppo_agent.py     PPO agent with a shared policy
  config/
    config.py              simulation parameters (SimulationConfig, build_simulation_config)
    mfd_profiles.py       reads mfd.json and topology.json from a network folder
  networks/7_region_new/
    mfd.json               MFD calibration for all regions
    qmax_het_7.mat         max-flow and demand parameters
    xInit_ss_7.mat         initial accumulation
  rg_mfd_simplified/
    simulation.py          event loop, vehicle store, observation builder
    routing.py             Dijkstra region routing
    path_choice.py         feasible next regions, agent and HDV choice
    mfd.py                 MFD speed and production
    demand.py  travelers.py  matlab_io.py  utils.py
```

## How a run works

1. `build_simulation_config` reads the network folder: 7 regions, ring of 6 plus a central hub, with the MFD parameters from `mfd.json`.
2. `run_episode` builds the OD demand and the initial vehicles, then steps through events (departures and region-boundary crossings).
3. At each event every region's speed comes from its accumulation through the MFD. Dijkstra prices each region on all vehicles (CAVs and HDVs).
4. When a vehicle reaches a region boundary it picks the next region. HDVs follow Dijkstra. CAVs ask the agent, which sees a 28-number observation: current region (7), destination (7), accumulation over jam accumulation per region (7), and the action mask (7).
5. After every event the agent is told the reward, which is minus the time elapsed times the number of active vehicles. It is also told when a CAV arrives and when the episode ends.

## Changing things

| Goal | Where |
|---|---|
| Lighter or heavier traffic | `--demand-scale`. A value near 0.8 or above is where the network starts to congest. |
| CAV share (default 60%) | `CAV_penetration` in `config/config.py`. |
| MFD shape or region capacity | `networks/7_region_new/mfd.json`. |
| Start with an emptier or fuller network | `initial_accumulation_scale` in `config/config.py`. |
| Allow revisiting regions | `ban_revisit` argument of `run_episode` in `rg_mfd_simplified/simulation.py` (on by default). |
| A different agent | Any object with `select_action(vehicle, state, feasible, action_mask)` works as `rl_agent`. Optional hooks: `observe_event_reward(reward)`, `observe_arrival(vehicle)`, `end_episode()`. `state["observation"]` holds the 28-number vector. |

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: config` or `rg_mfd_simplified` | You are not inside `scripts/RG_MFD_simplified/`. `cd` there first. |
| `FileNotFoundError` for a `.mat` file | The `networks/7_region_new/` folder was moved or is incomplete. |
| First run is slow to start | `numba` compiles the MFD kernel on first use. It is cached afterwards. |
| Killed or out of memory | Run fewer simultaneous jobs, or lower `--max-events`. |
| Run never finishes | Gridlock. Set `--max-events`, or lower `--demand-scale`. |
| `loss nan` for IQL | Fewer than 64 transitions collected yet. Use more events. |
| Evaluation says few vehicles completed | The episode was cut by `--max-events`. Raise it before comparing numbers. |

## Not included

The full research code (IQL and PPO training with multiple fleets and behaviour profiles, curriculum
and oracle experiments, the 19-region network, MPC) is not in this repository. The files in `docs/` describe it.
