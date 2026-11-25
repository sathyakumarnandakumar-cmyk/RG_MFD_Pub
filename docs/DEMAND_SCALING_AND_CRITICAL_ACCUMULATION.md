# Demand scaling and critical accumulation

**How to choose `--future-demand-scale` / `--init-acc-scale` to land peak region
accumulation at a target fraction of `n_critical` — and why the obvious calculation
is wrong by a factor of ~3.5.**

Companion to `scripts/load_calc.py`, which implements everything below. This document
is the derivation and the evidence; the script is the tool.

Answers open question #1 of `docs/UE_SO_GAP_FINDINGS.md` §7 ("sweep demand on
`7_region_new` to land peak accumulation in the 80–110% of critical band")
analytically rather than by probing: **demand ≈ 0.79 at `--init-acc-scale 0.3`.**

All numbers below are for `networks/7_region_new` at CAV penetration 0.6 with
`ban_revisit=True` and a `selfish` policy, measured 2026-08-18.

---

## 0. The question, and the one-paragraph answer

`n_critical = 2812` veh/region is known from the calibration. You want the demand and
initial-accumulation scales that put the network at some target load relative to it.
The naive approach — count the vehicles entering the simulation, compare to
`7 × n_critical` — **fails**, because total vehicles is a cumulative quantity while
`n_critical` is instantaneous, and because the network never loads uniformly. The
working calculation needs three layers: an exact vehicle budget, a Little's-Law
bridge from cumulative to instantaneous (valid only in free flow), and an empirically
measured **concentration factor** capturing how badly routing funnels load onto the
hub. The third layer is not derivable from the MFD and dominates the answer.

---

## 1. The vehicle budget — exact, and insufficient

Two sources put vehicles into a run.

**Initial accumulation.** At `t=0` the network is pre-loaded from the 7×7 OD matrix
`xInit_ss_7.mat`, scaled by `--init-acc-scale` (`simulation.py:1025`):

```python
initial_accumulation = cfg.initial_accumulation_scale * xInit_ss_7
```

`sum(xInit_ss_7) = 12,742.9` veh at scale 1.0.

**Demand inflow.** Vehicles injected over the loading window `tfinq = 3600 s`, scaled
by `--future-demand-scale`. Total at scale 1.0 = **26,518** veh.

```
total_vehicles = 12,742.9 × init_acc_scale  +  26,518 × demand_scale
```

Verified against six launched runs; agreement <0.5% (residual is demand
discretisation into integer vehicles):

| run | init_acc | demand | predicted | simulator reported |
|---|---|---|---|---|
| `crit_fd070` | 0.3 | 0.7 | 22,386 | 22,480 |
| `crit_fd080` | 0.3 | 0.8 | 25,037 | 25,127 |
| `crit_fd090` | 0.3 | 0.9 | 27,689 | 27,748 |
| `crit_fd100` | 0.3 | 1.0 | 30,341 | 30,340 |
| `fd_new_acc0.3` | 0.3 | 1.0 | 30,341 | 30,340 |
| `fd_new_acc1.0` | 1.0 | 1.0 | 39,261 | 39,261 |

### Why this is not the answer

Network capacity at critical is `7 × 2812 = 19,684` veh. The budget at demand 0.7 is
22,386 — **above** it. Naive conservation therefore predicts demand 0.7 is already
over critical. Measured: R1 peaked at **73%** of critical. The naive calculation is
wrong, and §2–§5 are why.

---

## 2. Total vehicles ≠ simultaneous accumulation

`n_critical` is **instantaneous** (vehicles in a region at one moment); the budget is
**cumulative** (vehicles over the whole episode). A vehicle entering at t=100 and
leaving at t=800 counts once toward the budget but occupies the network for only 700 s.

The bridge is **Little's Law**, `L = λ·W`:

- `L` — average vehicles in system (accumulation)
- `λ` — arrival rate, veh/s
- `W` — average time in system (travel time)

Worked at demand 0.7: `λ = 18,563/3600 = 5.16` veh/s, `W = 1,867 s` (measured
`avg_tt_all`), so `L = 9,629` veh network-wide. Measured mean network accumulation was
7,782 and peak 11,482 — so `peak ≈ 1.19 × L`.

| demand | λ (veh/s) | W (s) | L = λ·W | measured peak | peak / L |
|---|---|---|---|---|---|
| 0.5 | 3.68 | 1,739 | 6,405 | 7,947 | **1.24×** |
| 0.7 | 5.16 | 1,867 | 9,629 | 11,482 | **1.19×** |
| 0.9 | 6.63 | 3,066 | 20,329 | 15,358 | 0.76× |
| 1.1 | 8.10 | 5,254 | 42,573 | 20,357 | 0.48× |

Stable at ~1.2× in free flow; collapses above critical.

---

## 3. Why Little's Law breaks above critical

At demand 1.1 it predicts `L = 42,573` veh — but the **entire vehicle budget** at that
demand is 32,993. The prediction exceeds the number of vehicles that exist. The model
has failed, for two compounding reasons.

**(a) `W` is endogenous.** Little's Law treats `W` as an independent input, but travel
time depends on accumulation: more vehicles → lower speed → larger `W` → more
accumulation. Below critical the MFD is near-linear and this feedback is weak. Above
it the feedback is strong and `L = λ·W` becomes circular rather than predictive. The
honest form is the fixed point `L = λ·W(L)` — which is what the simulator solves
numerically, and why the simulator exists.

**(b) The system is transient, not steady-state.** Little's Law is exact in
equilibrium. Here demand loads for 3,600 s and then **stops**: the network fills, then
drains. Episode-average `W` includes the long congested drain phase where stuck
vehicles inflate the mean. At demand 1.1, `W = 5,254 s` mostly reflects gridlocked
vehicles, not sustained throughput.

➡️ Little's Law is a good **free-flow** estimator and a useless **congested** one.
`load_calc.py` reports it with its validity range rather than extrapolating.

---

## 4. Network throughput capacity — real physics, but not the constraint

A second, independent bound: can the network *clear* vehicles as fast as they arrive?

From the MFD speed polynomial (`simulation.py:1042/1095`):

```
speed(n) = ave_trip_length × (a·n² + b·n + c) × pen_factor × speed_unit_scale   [m/s]
```

A vehicle crossing a region covers `ave_trip_length` metres, so completes at rate
`speed / ave_trip_length` per second. With `n` present, the region's **trip completion
rate** is:

```
G(n) = n × speed(n) / ave_trip_length
     = n × (a·n² + b·n + c) × pen_factor × speed_unit_scale                     [veh/s]
```

`ave_trip_length` cancels. `G(n)` is maximised exactly at `n_critical` — that is the
*definition* of critical accumulation (see `UE_SO_GAP_FINDINGS.md` §4a).

At ρ = 0.6, `pen_factor = 1.4869`:

- `G(2812) = 2.869` veh/s per region
- network: `7 × 2.869 = **20.08 veh/s**`

| demand | λ (veh/s) | % of network capacity |
|---|---|---|
| 0.7 | 5.16 | 26% |
| 0.9 | 6.63 | 33% |
| 1.0 | 7.37 | 37% |
| 1.1 | 8.10 | 40% |
| **2.73** | **20.08** | **100%** |

**The network would need demand 2.73 to saturate on aggregate throughput. We observe
catastrophic congestion at 0.9.** Aggregate physics claims 3.5× more headroom than
exists. This is the entire discrepancy — and it means **no purely physics-based
calculation from `mfd.json` can answer the question.**

---

## 5. The binding constraint: concentration

The resolution: **the network is never uniformly loaded.** Aggregate capacity sums
over regions, but a region jams on *its own* `n`. Routing funnels traffic into R1 and
R7, so those saturate while R3–R6 sit at ~60% and the aggregate still looks healthy.

```
concentration = max_region_n / mean_region_n      (evaluated at the network-wide peak)
```

Measured from the probe runs' `episodes/*_accumulation.npz`:

| demand | 0.5 | 0.7 | 0.9 | 1.1 |
|---|---|---|---|---|
| concentration | 1.18× | 1.24× | 1.34× | **1.94×** |
| worst region (veh) | 1,343 | 2,039 | 2,940 | 5,644 |
| worst region (% critical) | 48% | 73% | 105% | 201% |

**It exceeds 1.0 everywhere** — load is never uniform even in free flow. Structural:
R7 is the hub (degree 6) and appears in every outer region's action set, so shortest
paths run through it disproportionately.

**It grows with demand, and accelerates** (1.18 → 1.24 → 1.34 is gradual; 1.34 → 1.94
is a jump). This is a feedback loop: congestion in R1/R7 degrades their speeds →
routing costs shift → but the deterministic all-or-nothing router (`route_choice`,
`k_shortest_paths=False` — see §7) has no mechanism to *spread* load, so it keeps
committing traffic to the same corridors, worsening the congestion. There is no
pressure-relief valve.

> ⚠️ **Concentration is a property of the ROUTING POLICY, not of the MFD.** It cannot
> be derived from `mfd.json`. Re-measure it whenever the network, CAV penetration,
> `ban_revisit` setting, or trained policy changes:
> ```bash
> python scripts/load_calc.py --network networks/7_region_new --calibrate results/<id>
> ```
> Measured on `crit_fd070` it held at 1.18–1.31× across 150 episodes, so it is stable
> *within* a configuration.

This is the irreducible empirical core: probing cannot be eliminated, only reduced
from "sweep blindly" to "measure once, then compute".

---

## 6. The inverse calculation

Chain the layers, worst-region-first:

```
1. target worst-region load       = n_critical × target_pct/100
2. required network accumulation  = target × nb_region / concentration
3. required demand scale          ← invert the measured peak-vs-demand fit
4. concentration depends on demand → iterate to a fixed point
```

Step 3 uses a linear fit to the measured network peaks (7,947 / 11,482 / 15,358 at
demand 0.5 / 0.7 / 0.9; linear with R² ≈ 1.0 below critical) rather than `λ·W`,
precisely because `λ·W` fails in this range (§3).

```bash
$ python scripts/load_calc.py --network networks/7_region_new --target-pct 100

TARGET 100% of critical  (worst region = 2812 veh)
  concentration at that demand   1.29x
  required network accumulation  15304 veh
  => --future-demand-scale       0.79   (at --init-acc-scale 0.3)
```

### Independent validation

The model was fitted **only** to the four probe accumulations — not to any training
outcome. It places 100% of critical at demand **0.79**. Independently:

| run | demand | outcome |
|---|---|---|
| `crit_fd070` | 0.7 | converges cleanly, **1.06× HDV**, flat from episode 40 |
| `crit_fd080` | 0.8 | fails to converge, **2.28×**, oscillating |

The empirical regime change sits between 0.7 and 0.8; the model puts critical at 0.79.
It reproduces where learning breaks down without having been shown that data.

---

## 7. Corollary: why `--init-acc-scale` is the wrong lever

At demand 0.7 the two sources contribute:

- initial: `12,742.9 × 0.3 = 3,823` veh — present at `t=0`, **draining from the first
  second**, ~17% of the budget
- injected: `18,563` veh — arriving continuously over 3,600 s, ~83%, **sustained**

The initial condition washes out within roughly one loading window; peak accumulation
is set almost entirely by demand.

Measured, this is stark:

| lever | range swept | effect on R7 peak |
|---|---|---|
| `--init-acc-scale` | 0.3 → 2.0 (6.7×) | 346% → 389% of critical (**~nothing**) |
| `--future-demand-scale` | 0.5 → 1.1 (2.2×) | 48% → 201% of critical (**4.2×**) |

`UE_SO_GAP_FINDINGS.md` §2c recorded the same structural point for the shipped
calibration ("demand ×4 moved congestion from 24% → 25% of critical"). Both
observations are the same phenomenon seen from opposite ends: on a calibration with
no jam point demand cannot bite, and on one with a jam point the initial condition
cannot persist.

---

## 8. Limits — what this cannot do

1. **Valid only below ~critical.** The peak-vs-demand fit is linear in free flow and
   stops being linear exactly where regions congest. Targets above ~110% are
   unreliable; `load_calc.py` prints this warning rather than silently extrapolating.
2. **Concentration is configuration-specific.** Measured with `ban_revisit=True`,
   selfish, ρ=0.6, on `7_region_new`. Re-measure with `--calibrate` for anything else.
3. **Predicts the worst region, not the distribution.** It will not tell you R3–R6 sit
   at 60% while R1/R7 saturate. The per-episode `[accum]` line in `iql_torchrl.py`
   still does that.
4. **Cannot predict gridlock onset.** It tells you how to *stay below* critical, not
   what happens above it — that regime is where every closed-form relation here breaks.

---

## 9. Reproducing

```bash
# sweep table (demand vs predicted worst-region load)
python scripts/load_calc.py --network networks/7_region_new

# forward: what does this configuration produce?
python scripts/load_calc.py --network networks/7_region_new --init-acc 0.3 --demand 0.7

# inverse: what demand hits a target?
python scripts/load_calc.py --network networks/7_region_new --target-pct 100

# re-measure concentration from a run's saved accumulation npz files
python scripts/load_calc.py --network networks/7_region_new --calibrate results/crit_fd070
```

The `episodes/ep<N>_accumulation.npz` files that `--calibrate` consumes are written
every `--save-every` episodes by `iql_torchrl.py` and contain `accumulation`,
`production`, `speed`, `time`, `n_critical`, `n_jam` as `(7 × n_events)` arrays.

## Related

- `docs/MAP_AND_LOAD.md` — the **mechanism** companion to this document: what `x_init` and
  `κ` are, how demand is created and routed, with pseudocode. Read it first if the question
  is "what do these two knobs actually do?" rather than "what should I set them to?"
- `scripts/load_calc.py` — the implementation
- `docs/UE_SO_GAP_FINDINGS.md` §4a/§4b — `n_critical`/`n_jam` derivation, where
  `njamMFD` is (and is not) enforced in code
- `networks/7_region/README.md` — why the shipped calibration has no jam point
- `scripts/mfd_params.py` — derive `n_critical`/`n_jam` for any calibration
