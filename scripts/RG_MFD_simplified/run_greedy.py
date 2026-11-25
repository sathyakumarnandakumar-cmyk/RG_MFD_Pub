"""Greedy CAV routing on the 7-region network (networks/7_region_new)."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))

from rg_mfd.simulation import run_episode
from rg_mfd.utils import get_field_or_attribute

NETWORK = HERE / "networks" / "7_region_new"


class GreedyAgent:
    def __init__(self, q_table):
        self.q = q_table

    def select_action(self, vehicle, state, feasible, action_mask):
        curr = int(get_field_or_attribute(vehicle, "current_location_reg", get_field_or_attribute(vehicle, "current_location_sub", 1)))
        dest = int(get_field_or_attribute(vehicle, "Destination_reg", 1))
        best, best_t = None, float("inf")
        for nxt in feasible:
            t = self.q.get((curr, dest, nxt), float("inf"))
            if t < best_t:
                best, best_t = nxt, t
        if best is None and feasible:
            best = random.choice(feasible)
        return best


def update_best_trip_times(vehicles, q):
    for v in vehicles:
        if not get_field_or_attribute(v, "is_cav", False):
            continue
        path = get_field_or_attribute(v, "actual_region_recorded", [])
        if len(path) < 2:
            continue
        duration = float(get_field_or_attribute(v, "t_arr", 0.0)) - float(get_field_or_attribute(v, "t_dep", 0.0))
        key = (int(path[0]), int(get_field_or_attribute(v, "Destination_reg", 1)), int(path[1]))
        if key not in q or duration < q[key]:
            q[key] = duration


def mean_travel_time(vehicles, cav):
    t = [float(get_field_or_attribute(v, "t_arr", 0.0)) - float(get_field_or_attribute(v, "t_dep", 0.0))
         for v in vehicles if bool(get_field_or_attribute(v, "is_cav", False)) == cav]
    return (float(np.mean(t)) if t else float("nan")), len(t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="greedy_7region_new")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demand-scale", type=float, default=0.7, help="future demand scale")
    ap.add_argument("--max-events", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    out = HERE / "results" / args.id
    out.mkdir(parents=True, exist_ok=True)
    q = {}
    rows = []
    for ep in range(1, args.episodes + 1):
        res = run_episode(NETWORK, rl_agent=GreedyAgent(q), rng_seed=args.seed + ep,
                       future_demand_scale=args.demand_scale, max_events=args.max_events)
        veh = res["vehicles"]
        update_best_trip_times(veh, q)
        cav, n_cav = mean_travel_time(veh, True)
        hdv, n_hdv = mean_travel_time(veh, False)
        ratio = cav / hdv if hdv == hdv and hdv > 0 else float("nan")
        print(f"episode {ep}/{args.episodes} | CAV {cav:.1f}s (n={n_cav}) | HDV {hdv:.1f}s (n={n_hdv}) | CAV/HDV {ratio:.3f}")
        rows.append([ep, cav, n_cav, hdv, n_hdv, ratio])
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "cav_tt", "n_cav", "hdv_tt", "n_hdv", "cav_over_hdv"])
        w.writerows(rows)
