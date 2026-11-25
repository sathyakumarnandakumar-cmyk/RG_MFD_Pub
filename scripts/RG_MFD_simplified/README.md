# RG_MFD_simplified

Minimal version of the RG-MFD regional macroscopic traffic simulation.

- Network: 7 regions, `networks/7_region_new` only (MFD calibration in its `mfd.json`).
- Routing: Dijkstra only, all-or-nothing on the cheapest region path. No k-shortest paths, no MPC.
- Dijkstra prices every region on 100% of the vehicles (CAVs and HDVs), not the 40% HDV share.
- CAVs: greedy lookup (`run_greedy.py`) or an IQL agent (`train_iql.py`, `eval_iql.py`).
- IQL: 28-dim observation (current region, destination, region density, action mask), reward = negative total time spent, soft target-network update.
- No plotting or diagnostics.

```bash
pip install -r requirements.txt
python run_greedy.py --episodes 5
python run_greedy.py --episodes 1 --max-events 1000   # quick check
python train_iql.py --id my_run --episodes 50
python eval_iql.py --id my_run                        # epsilon 0, seeds 5000-5004, vs all-Dijkstra
```

Output: one line per episode (CAV and HDV mean travel time, CAV/HDV ratio) and
`results/<id>/episodes.csv`.
