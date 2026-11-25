import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from config.config import build_simulation_config
from rg_mfd_simplified.simulation import observation_dimension, run_episode
from rg_mfd_simplified.utils import get_field_or_attribute
from rl_agent import IQLAgent

NETWORK = HERE / "networks" / "7_region_new"


def mean_travel_time(vehicles, is_cav):
    times = [float(get_field_or_attribute(v, "t_arr", 0.0)) - float(get_field_or_attribute(v, "t_dep", 0.0))
             for v in vehicles if bool(get_field_or_attribute(v, "is_cav", False)) == is_cav]
    return (float(np.mean(times)) if times else float("nan")), len(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="iql_7region_new")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demand-scale", type=float, default=0.7)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = build_simulation_config(str(NETWORK))
    agent = IQLAgent(observation_dimension(config), config.nb_region, epsilon=args.epsilon_start,
                     learning_rate=args.learning_rate)
    output_dir = HERE / "results" / args.id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    decay_episodes = max(1.0, args.epsilon_decay_fraction * args.episodes)
    for episode in range(1, args.episodes + 1):
        fraction = min(1.0, (episode - 1) / decay_episodes)
        agent.epsilon = args.epsilon_start + fraction * (args.epsilon_end - args.epsilon_start)
        result = run_episode(NETWORK, rl_agent=agent, rng_seed=args.seed + episode,
                             future_demand_scale=args.demand_scale, max_events=args.max_events)
        cav_time, cav_count = mean_travel_time(result["vehicles"], True)
        hdv_time, hdv_count = mean_travel_time(result["vehicles"], False)
        ratio = cav_time / hdv_time if hdv_time > 0 else float("nan")
        print(f"episode {episode}/{args.episodes} | epsilon {agent.epsilon:.2f} | CAV {cav_time:.1f}s (n={cav_count}) "
              f"| HDV {hdv_time:.1f}s (n={hdv_count}) | CAV/HDV {ratio:.3f} | loss {agent.last_loss:.4f}")
        rows.append([episode, agent.epsilon, cav_time, cav_count, hdv_time, hdv_count, ratio, agent.last_loss])
        agent.save(output_dir / "q_network.pt")

    with open(output_dir / "episodes.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "epsilon", "cav_travel_time", "cav_count", "hdv_travel_time",
                         "hdv_count", "cav_over_hdv", "loss"])
        writer.writerows(rows)
