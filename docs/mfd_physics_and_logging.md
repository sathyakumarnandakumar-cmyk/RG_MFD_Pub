# Macroscopic Fundamental Diagram (MFD) Physics and Logging

This document explains the core physics calculations used to evaluate Macroscopic Fundamental Diagrams (MFD) within the codebase, specifically clarifying the differences between Accumulation and Production, and the mathematical derivation of the dynamic speed calculation formula. It also details the logging optimizations used to handle large-scale simulation outputs.

## 1. Accumulation vs. Production

These are the two fundamental axes of a Macroscopic Fundamental Diagram (MFD):

*   **Accumulation ($n$)**: Simply the raw number of vehicles actively driving inside the region right now (e.g., 1,500 vehicles). Think of it as a headcount.
*   **Production ($P$)**: The total travel distance completed by *all* vehicles in the network per unit of time (e.g., veh-meters/second). It is the true measure of network throughput (how much "work" the road network is getting done).
    *   *Formula:* `Production = Accumulation * Average Speed` ($P = n \times v$).

## 2. The Speed Formula Derivation

At first glance, the formula used for speed calculation in the codebase looks incorrect:

```python
speed = rd.ave_trip_length * (rd.aMFD * n**2 + rd.bMFD * n + rd.cMFD) * pen_factor
```

It looks like the quadratic component ($a n^2 + b n + c$) represents the speed, but typically it represents the MFD curve itself. However, mathematically, it perfectly resolves to speed based on how the MFD parameters are calibrated in this specific MATLAB-ported physics environment.

In this codebase, the `aMFD`, `bMFD`, and `cMFD` parameters are specifically calibrated to define the **Trip Completion Rate** (also known as Outflow, $G$) as a **cubic (third-degree) polynomial** based on accumulation ($n$):

$$G(n) = a \cdot n^3 + b \cdot n^2 + c \cdot n$$

*(Note: There is no intercept $d$ because if there are 0 vehicles, 0 trips are completed).*

Here is the step-by-step derivation connecting Trip Completion Rate to Speed using standard laws of traffic physics:

1.  **Trip Completion Rate ($G$)** is equal to the total network Production ($P$) divided by the Average Trip Length ($L$):
    $$G(n) = \frac{P}{L}$$

2.  We know that Production ($P$) is Accumulation ($n$) multiplied by Speed ($v$). Substitute that in:
    $$G(n) = \frac{n \cdot v}{L}$$

3.  Now, solve for Speed ($v$) by isolating it:
    $$v = \frac{G(n) \cdot L}{n}$$

4.  Finally, substitute the codebase's cubic polynomial $G(n)$ into the equation:
    $$v = \frac{(a \cdot n^3 + b \cdot n^2 + c \cdot n) \cdot L}{n}$$

**If you divide the polynomial by $n$, the $n$ cancels out from each term, leaving a quadratic equation:**

$$v = L \cdot (a \cdot n^2 + b \cdot n + c)$$

Which perfectly matches the codebase's `speed` equation. 

### Summary of Terms
*   $a n^3 + b n^2 + c n$ = **Total Outflow** (Vehicles finishing trips per second)
*   $a n^2 + b n + c$ = **Outflow per vehicle** (The rate at which a *single* vehicle finishes a trip)
*   Multiply the "outflow per vehicle" by the "distance of the trip" ($L$), and you get the **Speed** (Distance per second).

---

## 3. Efficient MFD Logging (NPZ vs CSV)

During simulation, writing the state of 7 regions for up to 175,000 events into a standard plain text `.csv` file consumes roughly ~100MB of disk space per episode. Across large reinforcement learning training runs, this generates unacceptable disk I/O overhead and consumes massive storage.

To mitigate this (achieving a ~90% reduction to ~10MB per file), the core simulation avoids returning bloated tracking arrays. Instead, it only returns the raw `accumulation_save` and `time_plot` arrays. At the end of an episode, the Python script intercepts these raw arrays and dynamically re-calculates the `speed` and `production` metrics using the derived physics formulas from above. 

The resulting variables are then packaged and compressed into a binary NumPy archive (`.npz`) using `np.savez_compressed()`.

### What's built every episode vs. what's persisted to disk

These are two different cadences, easy to conflate:

*   **`accumulation_save` (per-region $n_r(t)$) is built every single episode, unconditionally.**
    Inside `run_main()` (`rg_mfd/simulation.py`), it is appended to on *every simulation event*
    (`accumulation_save.append(current_reg_n)`), regardless of `--save-every` -- the training
    loop's console printout (`[accum] peak per-region: R1=...(...%cr/...%jam)...`) reads this array
    every episode to report peak accumulation. It is a plain Python list local to that one
    `run_main()` call: at up to 175,000 events x 7 regions, it costs a few MB while that episode
    runs, then is freed once the episode's training-loop iteration ends -- it is **not** retained
    in memory across episodes, so it does not contribute to any cross-episode memory growth (see
    the per-process RAM discussion in the training scripts' `peak_ram_mb` column, which climbs and
    plateaus for unrelated reasons -- allocator warm-up and the fact that `ru_maxrss` is a
    high-water mark, not a live-usage gauge).

*   **Production, speed, the `.npz` snapshot, and the per-vehicle records CSV are only computed
    and written on `save_every` episodes** (`if episode % args.save_every == 0:`). Production is
    not tracked continuously -- it only ever gets computed as a byproduct of writing a snapshot, from
    that episode's `accumulation_save` array via the speed formula above. Two files land together
    on the same cadence, both under `results/<id>/episodes/`:
    *   `ep{episode}.csv` -- one row per vehicle that arrived that episode (`travel_time`, `is_cav`,
        `vehicle_type`, `origin_reg`, `destination_reg`). Plain CSV, not compressed -- this is the
        small, per-vehicle-outcome file (e.g. `ep70.csv`), distinct from the per-region MFD trace.
    *   `ep{episode}_accumulation.npz` -- the per-region `accumulation`/`production`/`speed`/`time`
        trace described below, compressed.

    With the default `--save-every 10` on a 1000-episode run, that's 100 of each file, not 1000 --
    the other 900 episodes' accumulation/production traces exist only transiently in memory for
    that one episode's console printout, then are discarded with no disk trace at all. This is also
    why `/plot-intra-episode` can only plot episodes landing on a save-every boundary: it reads the
    `.npz`, and non-boundary episodes never wrote one.

*   **Identical convention in both training scripts.** `scripts/iql_torchrl.py` and
    `scripts/ippo_torchrl.py` implement this save-cadence logic independently but identically --
    same `if episode % args.save_every == 0:` gate, same two files, same physics re-derivation.
    IPPO is not a special case here despite having a very different training loop (on-policy,
    synchronous PPO update vs. IQL's async replay-buffer thread) -- the accumulation/production
    logging path is shared logic, copied verbatim between the two scripts, not routed through a
    common module.

### Implementation Snippet (`scripts/iql_torchrl.py`; `scripts/ippo_torchrl.py` is the same code)

```python
# ---- Save per-region accumulation + production trajectory every save_every episodes ----
if episode % args.save_every == 0:
    acc = np.asarray(result_data.get("accumulation_save"), dtype=float)
    tvec = np.asarray(result_data.get("time_plot"), dtype=float)
    
    if acc.size:
        # 1. Compute penetration factor for Connected Autonomous Vehicles (CAVs)
        pen = compute_pen_factor(
            sim_cfg.region_data[0].dMFD, sim_cfg.region_data[0].eMFD,
            sim_cfg.region_data[0].fMFD,
            args.cav_penetration if args.cav_penetration is not None else sim_cfg.CAV_penetration,
        )
        scale_u = getattr(sim_cfg, "speed_unit_scale", 1.0)
        speed = np.empty_like(acc)
        
        # 2. Dynamically re-calculate Speed using the derived quadratic formula 
        #    v = L * (a*n^2 + b*n + c)
        for i in range(sim_cfg.nb_region):
            rd = sim_cfg.region_data[i]
            n_i = acc[i]
            speed[i] = rd.ave_trip_length * (rd.aMFD * n_i**2 + rd.bMFD * n_i + rd.cMFD) * pen * scale_u
            
        speed = np.maximum(speed, MIN_SPEED)   # same floor the event loop applies
        
        # 3. Calculate Production dynamically (P = n * v)
        production = acc * speed
        
        npz_path = episodes_dir / f"ep{episode}_accumulation.npz"
        
        # 4. Save as a highly compressed binary NumPy archive instead of a heavy text CSV
        np.savez_compressed(
            npz_path, accumulation=acc, production=production, speed=speed, time=tvec,
            n_critical=np.array([rd.ncriticalMFD for rd in sim_cfg.region_data], dtype=float),
            n_jam=np.array([rd.njamMFD for rd in sim_cfg.region_data], dtype=float),
        )
        print(f"  Saved accumulation/production ({acc.shape[1]} events x {acc.shape[0]} regions) to {npz_path.name}")
```
