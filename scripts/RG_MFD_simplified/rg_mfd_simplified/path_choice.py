from __future__ import annotations

from collections.abc import Callable

import numpy as np

from config.config import SimulationConfig, build_simulation_config
from .routing import shortest_path_and_cost


def build_action_mask(feasible_next_regions, nb_region: int) -> np.ndarray:
    mask = np.zeros(int(nb_region), dtype=bool)
    for region in feasible_next_regions:
        region = int(region)
        if 1 <= region <= nb_region:
            mask[region - 1] = True
    return mask


def _can_reach_destination(start: int, dest: int, graph, blocked: set[int] | None = None) -> bool:
    if start == dest:
        return True
    blocked = set() if blocked is None else set(blocked)
    adj = np.asarray(graph, dtype=float)
    queue = [int(start)]
    visited = {int(start)}
    while queue:
        node = queue.pop(0)
        for nb in np.where(adj[node - 1, :] > 0)[0] + 1:
            nb = int(nb)
            if nb == dest:
                return True
            if nb in visited or nb in blocked:
                continue
            visited.add(nb)
            queue.append(nb)
    return False


def get_feasible_next_regions_from_region(current_region: int, destination_region: int, cfg: SimulationConfig | None = None, avoid_regions=None) -> list[int]:
    cfg = cfg or build_simulation_config()
    current_region = int(current_region)
    destination_region = int(destination_region)
    if current_region == destination_region:
        return []

    graph = np.asarray(cfg.reg_dgraph, dtype=float)
    blocked = set() if avoid_regions is None else set(int(region) for region in avoid_regions)
    blocked.discard(current_region)
    blocked.discard(destination_region)

    neighbors = [int(x) for x in np.where(graph[current_region - 1, :] > 0)[0] + 1]


    reach_blocked = blocked
    if blocked and getattr(cfg, "ban_revisit_count_current_region", True):
        reach_blocked = blocked | {current_region}
    feasible = [
        nb
        for nb in neighbors
        if nb not in blocked and _can_reach_destination(nb, destination_region, graph, reach_blocked)
    ]
    if not feasible and blocked:
        feasible = [nb for nb in neighbors if _can_reach_destination(nb, destination_region, graph, set())]
    return feasible


def get_feasible_next_regions_for_vehicle(vehicle, cfg: SimulationConfig | None = None, avoid_visited: bool | None = None) -> list[int]:
    cfg = cfg or build_simulation_config()
    curr = int(vehicle.get("current_location_reg", 1))
    dest = int(vehicle.get("Destination_reg", curr))


    if avoid_visited is None:
        avoid_visited = bool(getattr(cfg, "ban_revisit", False))


    if avoid_visited:
        history = set(int(x) for x in vehicle.get("actual_region_recorded", []))
        avoid_regions = history - {curr, dest}
    else:
        avoid_regions = set()
    return get_feasible_next_regions_from_region(curr, dest, cfg, avoid_regions)


def _best_feasible_by_travel_time(vehicle, feasible: list[int], state: dict) -> int:
    curr = int(vehicle.get("current_location_reg", 1))
    dest = int(vehicle.get("Destination_reg", curr))
    net_cost = state.get("region_net_cost")
    if net_cost is None:
        return int(feasible[0])

    costs = []
    for region in feasible:
        edge_cost = float(net_cost[curr - 1, region - 1])
        if region == dest:
            tail_cost = 0.0
        else:
            _, tail_cost = shortest_path_and_cost(net_cost, region, dest)
        total = edge_cost + tail_cost
        costs.append(total if np.isfinite(total) else np.inf)
    best_idx = int(np.argmin(np.asarray(costs, dtype=float)))
    return int(feasible[best_idx])


def _ask_agent_for_action(rl_agent, vehicle, state: dict, feasible: list[int], action_mask: np.ndarray):
    if rl_agent is None:
        return None


    return rl_agent.select_action(vehicle, state, feasible, action_mask)


def _convert_action_to_region(action, feasible: list[int], action_mask: np.ndarray) -> int | None:
    if action is None:
        return None


    val = int(action)
    if val in feasible:
        return val
    return None


def choose_cav_next_region_with_agent(vehicle, state, feasible_next_regions, rl_agent):
    feasible = [int(region) for region in feasible_next_regions]
    if not feasible:
        return None
    if len(feasible) == 1:
        return feasible[0]

    cfg = state.get("cfg") if isinstance(state, dict) else None
    cfg = cfg or build_simulation_config()
    action_mask = build_action_mask(feasible, cfg.nb_region)
    agent_state = dict(state)
    agent_state["action_mask"] = action_mask
    agent_state["feasible_next_regions"] = np.asarray(feasible, dtype=int)

    action = _ask_agent_for_action(rl_agent, vehicle, agent_state, feasible, action_mask)
    selected = _convert_action_to_region(action, feasible, action_mask)
    if selected in feasible:
        return selected
    return _best_feasible_by_travel_time(vehicle, feasible, agent_state)


def _normalize_probabilities(feasible: list[int], probabilities) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones(len(feasible), dtype=float) / len(feasible)
    return probabilities / total


def _sample_from_probabilities(feasible: list[int], probabilities: np.ndarray, rng) -> int:
    probabilities = _normalize_probabilities(feasible, probabilities)
    idx = int(np.searchsorted(np.cumsum(probabilities), rng.random(), side="left"))
    if idx >= len(feasible):
        idx = len(feasible) - 1
    return int(feasible[idx])


def choose_hdv_next_region(vehicle, feasible_next_regions, network_state, record: dict | None = None):
    """Sample the HDV's next region from its logit route-choice distribution."""
    def _store_hdv_probabilities_in_record(probabilities, source: str):
        if record is not None:
            record["hdv_probs"] = _normalize_probabilities(feasible, probabilities)
            record["hdv_prob_source"] = source

    feasible = [int(region) for region in feasible_next_regions]
    if not feasible:
        return None
    if len(feasible) == 1:
        _store_hdv_probabilities_in_record([1.0], "forced")
        return feasible[0]

    curr = int(vehicle.get("current_location_reg", 1))
    dest = int(vehicle.get("Destination_reg", curr))
    rng = network_state.get("rng")
    if rng is None:
        rng = np.random.RandomState()

    path_probs = network_state.get("prob_path_ij")
    shortest_paths = network_state.get("shortest_paths_ij")
    probabilities = np.zeros(len(feasible), dtype=float)
    if path_probs is not None and shortest_paths is not None:
        paths = shortest_paths[curr - 1][dest - 1]
        probs = np.asarray(path_probs[curr - 1][dest - 1], dtype=float).reshape(-1)
        for path, prob in zip(paths, probs):
            if len(path) >= 2 and int(path[1]) in feasible:
                probabilities[feasible.index(int(path[1]))] += float(prob)
    if probabilities.sum() > 0:
        _store_hdv_probabilities_in_record(probabilities, "path_set")
        return _sample_from_probabilities(feasible, probabilities, rng)

    costs = []
    net_cost = network_state.get("region_net_cost")
    for region in feasible:
        if net_cost is None:
            costs.append(0.0)
            continue
        edge_cost = float(net_cost[curr - 1, region - 1])
        if region == dest:
            tail_cost = 0.0
        else:
            _, tail_cost = shortest_path_and_cost(net_cost, region, dest)
        costs.append(edge_cost + tail_cost)
    costs = np.asarray(costs, dtype=float)
    finite = np.isfinite(costs)
    probabilities = np.zeros(len(feasible), dtype=float)
    if np.any(finite):
        scale = float(network_state.get("logit_scale", 0.5))
        stable_costs = costs[finite] - np.min(costs[finite])
        probabilities[finite] = np.exp(-scale * stable_costs)
    _store_hdv_probabilities_in_record(probabilities, "logit_fallback")
    return _sample_from_probabilities(feasible, probabilities, rng)
