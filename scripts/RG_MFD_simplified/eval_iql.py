import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.config import build_simulation_config
from rg_mfd_simplified.simulation import observation_dimension, run_episode
from rg_mfd_simplified.utils import get_field_or_attribute
from rl_agent import IQLAgent

NETWORK = HERE / "networks" / "7_region_new"


def travel_times(vehicles, is_cav=None):
    return [float(get_field_or_attribute(v, "t_arr", 0.0)) - float(get_field_or_attribute(v, "t_dep", 0.0))
            for v in vehicles
            if is_cav is None or bool(get_field_or_attribute(v, "is_cav", False)) == is_cav]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="iql_7region_new")
    parser.add_argument("--first-seed", type=int, default=5000)
    parser.add_argument("--number-of-seeds", type=int, default=5)
    parser.add_argument("--demand-scale", type=float, default=0.7)
    parser.add_argument("--max-events", type=int, default=80000)
    args = parser.parse_args()

    config = build_simulation_config(str(NETWORK))
    agent = IQLAgent(observation_dimension(config), config.nb_region, epsilon=0.0)
    agent.load(HERE / "results" / args.id / "q_network.pt")

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.number_of_seeds):
        policy = run_episode(NETWORK, rl_agent=agent, rng_seed=seed, future_demand_scale=args.demand_scale,
                             max_events=args.max_events, show_progress=False)
        dijkstra = run_episode(NETWORK, rl_agent=None, rng_seed=seed, future_demand_scale=args.demand_scale,
                               max_events=args.max_events, show_progress=False)
        cav = np.mean(travel_times(policy["vehicles"], True))
        hdv = np.mean(travel_times(policy["vehicles"], False))
        all_policy = np.mean(travel_times(policy["vehicles"]))
        all_dijkstra = np.mean(travel_times(dijkstra["vehicles"]))
        row = {"seed": seed, "cav_travel_time": float(cav), "hdv_travel_time": float(hdv),
               "cav_over_hdv": float(cav / hdv), "all_vehicles_over_dijkstra": float(all_policy / all_dijkstra),
               "completed_policy": len(policy["vehicles"]), "completed_dijkstra": len(dijkstra["vehicles"]),
               "total_vehicles": policy["total_vehicles"]}
        rows.append(row)
        print(row)

    summary = {"mean_cav_over_hdv": float(np.mean([r["cav_over_hdv"] for r in rows])),
               "mean_all_vehicles_over_dijkstra": float(np.mean([r["all_vehicles_over_dijkstra"] for r in rows])),
               "seeds": rows}
    output_dir = HERE / "results" / args.id
    (output_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"mean CAV/HDV {summary['mean_cav_over_hdv']:.3f} | mean all/Dijkstra {summary['mean_all_vehicles_over_dijkstra']:.3f}")
