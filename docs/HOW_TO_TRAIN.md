# How to train

Training notes copied from the research code documentation. The commands and flags describe the full research script `scripts/iql_torchrl.py`, which is not part of this repository. For something runnable here, see `scripts/RG_MFD_simplified/` and the commands at the end of this file.

## Behaviour profiles and reward

Each fleet's per-vehicle **behavior reward** is a weighted sum of the vehicle's own travel time and three congestion/travel-time integrals, scaled and divided by 1000:
```python
veh_reward = (a * own_tt
              + b * scaled_my      # own-fleet congestion/TTS accumulated over the leg
              + c * scaled_other    # other-fleet + HDV congestion/TTS
              + d * scaled_all)     # whole-network congestion/TTS
             / 1000.0
```
- `own_tt` is the wall-clock time the vehicle spent on its leg (`global_time - start_time`).
- `scaled_*` are running congestion integrals multiplied by `congestion_scale` (default `0.01`).

The coefficients `(a, b, c, d)` are set by the `--behavior1` / `--behavior2` profiles (`get_behavior_coefs` in `scripts/iql_torchrl.py`). The profiles **actually wired up** are:

| Profile | a (own tt) | b (own fleet) | c (other/HDV) | d (global) | Intuition |
|---------|-----------|---------------|---------------|-----------|-----------|
| `selfish` | −1.0 | 0 | 0 | 0 | Minimize only my own travel time |
| `competitive` | −2.0 | 0 | +1.0 | 0 | Minimize mine **and** worsen others |
| `collaborative` | −0.5 | −0.5 | 0 | 0 | Balance self and own fleet |
| `martyr` | 0 | −1.0 | 0 | 0 | Only help own fleet, ignore self |
| `social` | −0.5 | 0 | 0 | −0.5 | Balance self and whole network |
| `altruistic` | 0 | 0 | 0 | −1.0 | Minimize total network delay |

> [!NOTE]
> Three further profiles (`malicious`, `collectivist`, `militant`) exist in `get_behavior_coefs` but are **commented out** — passing them will raise "unknown behavior". Re-enable them in the source if needed.

> [!IMPORTANT]
> **This behavior reward is NOT the same signal as the system reward.** The engine's per-event *system* reward (in `simulation.py`) is currently **pure negative Total-Time-Spent**: `reward = -tts_penalty` where `tts_penalty = dt * active_vehicles`. The quadratic congestion-penalty splits were commented out to isolate the TTS signal (see `reward_issues.md`). The congestion terms in the fleet formula above are therefore currently fed by per-fleet **TTS** integrals (`tts_fleet1/2/hdv`), not a separate congestion penalty. **Fleet reward (per-vehicle) and system reward (per-event) are on different scales and are not directly comparable** — always caveat which one a plot shows.


## Training mechanics

### 4. Asynchronous Training Thread

`IQLAgent` runs the DQN gradient updates on a **daemon background thread**, signalled every `--update-interval` decision steps (default 4). The simulation thread never blocks on backprop. Crucially, the inference network (`qnet_explore`) and the training/loss module **share the same underlying weight tensors** (no copying): `optimizer.step()` on the training thread mutates tensors the simulation thread's next forward pass immediately sees. A separate, slowly soft-updated **target network** provides stable TD targets. This is safe because PyTorch releases the GIL during tensor ops and per-step weight deltas are tiny.

### 5. Memory-Safe Checkpoint Chunking

Long runs are split into ~10-episode blocks via `--start-episode` / `--checkpoint-dir` (see `scripts/run_experiments.sh`). Each block loads the previous checkpoint (`results/<id>/checkpoints/agent1.pt`), trains its episodes, and saves — so a multi-hour run never holds all episodes' tensors in memory at once, avoiding PyTorch OOM.

### 6. Typical Invocation

```bash
python scripts/iql_torchrl.py --id my_exp --net Traveler_Info_7subreg_demand.mat \
    --episodes 50 --behavior1 selfish --behavior2 selfish --fleet1-ratio 1.0
```
`--fleet1-ratio 1.0` puts every CAV in Fleet 1 (single-fleet training); values in `(0, 1)` split CAVs across two independently-learning fleets. Add `--max-events 1000` for a fast smoke test (seconds instead of hours).

## Part III: Hyperparameter & Algorithm Reference

### Action Space

The action is **which adjacent region to enter next** — a discrete choice over `nb_region` (7) targets. Before every decision, infeasible regions (non-adjacent, or already-visited dead-ends) are masked out via `action_mask_from_feasible`, so the effective action set is only the vehicle's `feasible` neighbours. If exactly one region is feasible it is taken immediately without querying the network.

### Full CLI / Hyperparameter Reference

All flags on `scripts/iql_torchrl.py` (defaults shown):

| Flag | Default | Meaning |
|------|---------|---------|
| `--id` | *(required)* | Experiment name → `results/<id>/` |
| `--net` | *(required)* | Network `.mat` file (e.g. `Traveler_Info_7subreg_demand.mat`) |
| `--episodes` | 10 | Episodes to train this run |
| `--env-seed` / `--torch-seed` | 42 / 42 | RNG seeds for the simulation and PyTorch |
| `--max-events` | 175000 | Hard cap on events/episode (aborts runaway gridlock; use ~1000 for smoke tests) |
| `--behavior1` / `--behavior2` | selfish | Reward profile per fleet (see Part I table) |
| `--fleet1-ratio` | 1.0 | Fraction of CAVs in Fleet 1 (1.0 = single fleet) |
| `--cav-penetration` | None | Override CAV fraction (else uses config's 0.6) |
| `--lr` | 1e-3 | Adam learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--tau` | 0.005 | Target-network soft-update (Polyak) rate |
| `--buffer-size` | 10000 | Replay buffer capacity (transitions) |
| `--batch-size` | 64 | SGD minibatch size |
| `--update-interval` | 4 | Train once every N decision steps |
| `--annealing-steps` | 100000 | ε linear-anneal length |
| `--eps-end` | 0.1 | ε floor |
| `--eps-reset-interval` / `--eps-reset-value` | 100 / 0.5 | Sawtooth ε resets every N episodes up to this value (0 disables) |
| `--congestion-scale` | 0.01 | Scales the congestion/TTS terms in the fleet reward |
| `--double-dqn` / `--no-double-dqn` | **on** | Double-DQN targets (default ON) |
| `--mc-target` / `--no-mc-target` | off | Monte-Carlo return targets instead of bootstrapping |
| `--start-episode` / `--checkpoint-dir` | 1 / None | Chunked-run resume (see Part I §5) |
| `--save-every` | 10 | Write per-vehicle episode CSVs every N episodes |
| `--future-demand-scale` | 1.0 | Scale future demand |
| `--debug-sample-rate` | 0.0 | Fraction of vehicles to trace into debug logs |

### Learning Target: DQN → Double-DQN → Monte-Carlo

The agent stores transitions `(obs, action, reward, next_obs, done)` in a replay buffer and, every `--update-interval` steps, samples a minibatch and regresses `Q(s,a)` toward a target. Three target modes exist:

1. **Vanilla DQN** (`--no-double-dqn`): `target = r + γ · maxₐ' Q_target(s', a')`.
2. **Double-DQN** (default): action is chosen by the **online** net but evaluated by the **target** net — `target = r + γ · Q_target(s', argmaxₐ' Q_online(s', a'))`. This removes the max-operator overestimation bias of TD(0).
3. **Monte-Carlo return** (`--mc-target`): drops bootstrapping entirely and regresses `Q(s,a)` directly onto the realized γ-discounted return-to-go. Added because the reward investigation (`reward_issues.md` Test #6) found bootstrapping was converging to the wrong fixed point on this problem, while Test #5 proved the return target is learnable to R²=1.0.

The **target network** is a slowly Polyak-averaged copy of the online weights (`θ_target ← τ·θ_online + (1−τ)·θ_target` each update, `τ=0.005`), giving stable TD targets. As noted in Part I §4, the online *inference* net and the *loss* module share weight tensors directly; only the target net is a separate copy.

### Outputs

Each run writes under `results/<id>/`:
- `checkpoints/agent1.pt` — the trained Fleet-1 Q-network (loaded on chunked resume).
- `metrics/episode_metrics.csv` — per-episode travel time / reward / throughput.
- `losses/loss_values.csv` — TD-loss curve.
- `plots/` — `loss.png`, `avg_reward.png`, `avg_travel_times.png` (via `scripts/plot_experiment.py`).
- `exp_config.json` — the exact args the run was launched with.

### Known Issue (read before trusting results)

The selfish agent has historically converged to a suboptimal **hub-routing habit** (over-using region 7) even after reward and exploration fixes — see `reward_issues.md` / `reward_issues_simplified.md` at the repo root, and Appendix A of `RUN_FULL_SIMULATION_routing.md` for why the hub is a genuinely hard local optimum on this topology. Treat the RL pipeline as an active investigation, not a finished result.

## Simplified package

```bash
cd scripts/RG_MFD_simplified
pip install -r requirements.txt
python run_greedy.py --episodes 5
python train_iql.py --id my_run --episodes 50
python eval_iql.py --id my_run
```
