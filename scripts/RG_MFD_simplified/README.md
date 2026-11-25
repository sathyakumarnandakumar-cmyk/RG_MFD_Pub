# RG_MFD_simplified

Minimal version of the RG-MFD regional macroscopic traffic simulation.

- Network: 7 regions, `networks/7_region_new` only (MFD calibration in its `mfd.json`).
- Routing: Dijkstra only, all-or-nothing on the cheapest region path. No k-shortest paths, no MPC.
- Dijkstra prices every region on 100% of the vehicles (CAVs and HDVs), not the 40% HDV share.
- CAVs: greedy lookup of the best recorded trip time per (origin, destination, next region).
- No RL training, plotting or diagnostics.

```bash
pip install -r requirements.txt
python run_greedy.py --episodes 5
python run_greedy.py --episodes 1 --max-events 1000   # quick check
```

Output: one line per episode (CAV and HDV mean travel time, CAV/HDV ratio) and
`results/<id>/episodes.csv`.
