# RG_MFD: Event-Triggered RL Route Guidance for CAVs on Regional MFD Networks

A macroscopic traffic simulation of a 7-region urban network (based on Ingolstadt, Germany) and a
reinforcement-learning study of routing Connected and Autonomous Vehicles (CAVs) in mixed traffic.
Each region is an MFD "tank" (accumulation, speed, production), and routing decisions are made at
region boundaries rather than on individual road links.

The full write-up is in [`docs/RG_MFD_Report_2026-11-24.pdf`](docs/RG_MFD_Report_2026-11-24.pdf)
(Markdown source: [`docs/RG_MFD_Report_2026-11-24.md`](docs/RG_MFD_Report_2026-11-24.md)).

## What the study asks

Once the number of vehicles in a region passes a critical point, the region completes fewer trips per
unit time. So how vehicles are spread across regions decides how well road capacity is used. Only CAVs
can be guided centrally, while human-driven vehicles (HDVs) choose routes on their own. The question is
whether a learned routing policy for the CAVs can beat plain shortest-path (Dijkstra) routing.

## How it works

- **Network:** 7 regions (ring of 6 plus a central hub), each region modelled by a macroscopic fundamental diagram. 60% of vehicles are CAVs.
- **Event-driven simulation:** vehicles are followed individually through the regions. A CAV consults the policy only when it reaches a region boundary, and picks the next adjacent region.
- **Semi-Markov decisions:** the time between boundary crossings varies, so a transition spans many simulation events and its reward is accumulated over the whole leg.
- **Agents:** one shared policy for all CAVs, trained with DQN (IQL) or PPO over masked legal moves. HDVs recompute a deterministic shortest path at every boundary.
- **Observation:** current region, destination, region densities and the action mask. Later variants add congestion trends and the demand phase.
- **Reward:** negative total time spent, with behaviour profiles (selfish, collaborative, social, altruistic) that mix in own travel time and network-wide terms.

## Headline findings

- Adding regional congestion trends and the demand phase to the state stabilises training above light load. An ablation shows these features, not the change in normalisation, are responsible.
- Reward design strongly affects learnability. Profiles that keep a weight on the vehicle's own travel time can be learned. A purely network-wide travel-time reward fails, because one vehicle's action has a negligible effect on the total across thousands of vehicles.
- PPO copes with noisy rewards and higher demand better than DQN. PPO stays stable up to demand 1.30, while DQN diverges above 0.90.
- No configuration beats the Dijkstra-only baseline by more than its seed-to-seed variation, for CAVs against nearby HDVs or for the network as a whole.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/RG_MFD_simplified/` | Minimal, self-contained 7-region package: Dijkstra routing, a greedy agent, IQL training and evaluation. Start here to run something. |
| `docs/` | The report (PDF and Markdown) and design notes on the RL framework, IQL vs IPPO, MFD physics, demand scaling and the UE-SO gap. |
| `results/` | Metrics, plots and configs for every run in the report's experimental grid, plus greedy-evaluation outputs and the Dijkstra baseline sweep. |
| `networks/` | Input data for the 7-region and 19-region networks. |
| `reference_matlab_scripts/` | Original MATLAB reference implementation. |
| `URB_reference/` | The external URB benchmark, kept for cross-checking. |

## Quick start (simplified package)

Full install and usage notes are in [`TECHNICAL_README.md`](TECHNICAL_README.md).

```bash
cd scripts/RG_MFD_simplified
pip install -r requirements.txt
python run_greedy.py --episodes 5                    # greedy CAV routing vs Dijkstra HDVs
python train_iql.py --id my_run --episodes 50        # train an IQL agent
python eval_iql.py --id my_run                       # epsilon 0, seeds 5000-5004, vs all-Dijkstra
```

Add `--max-events 1000` to any script for a quick smoke test. See
[`scripts/RG_MFD_simplified/README.md`](scripts/RG_MFD_simplified/README.md) for details.

## Documentation

- [`docs/RG_MFD_Report_2026-11-24.pdf`](docs/RG_MFD_Report_2026-11-24.pdf): the full report.
- [`docs/RL_FRAMEWORK.md`](docs/RL_FRAMEWORK.md): MDP and semi-MDP formulation, hyperparameters, algorithm roadmap.
- [`docs/IQL_IPPO_COMPARISON.md`](docs/IQL_IPPO_COMPARISON.md): value-based vs policy-gradient training.
- [`docs/mfd_physics_and_logging.md`](docs/mfd_physics_and_logging.md): MFD physics and what the simulation logs.
- [`docs/DEMAND_SCALING_AND_CRITICAL_ACCUMULATION.md`](docs/DEMAND_SCALING_AND_CRITICAL_ACCUMULATION.md): how demand scaling relates to critical accumulation.
- [`docs/HOW_TO_TRAIN.md`](docs/HOW_TO_TRAIN.md): training flags, behaviour profiles and mechanics.
- [`docs/UE-SO_GAP_Report.md`](docs/UE-SO_GAP_Report.md): the user-equilibrium vs system-optimal gap.

The full research code (training scripts for IQL and PPO, curriculum and oracle experiments) is not
included in this repository.
