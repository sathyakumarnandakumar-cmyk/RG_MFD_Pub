"""Regional route choice: cost matrices, shortest paths, HDV path probabilities."""

from __future__ import annotations

import numpy as np

from config.config import SimulationConfig, build_simulation_config
from .mfd import compute_region_production_and_travel_times, compute_cav_penetration_factor
from .utils import make_random_state, get_field_or_attribute

LOGIT_SCALE = 0.5
_CACHE_LIMIT = 2000


def _region_parameters_at_index(subregion_data, i: int):
    """Region i's own MFD record from a per-region list, so compute_region_production_and_travel_times (which reads"""
    if isinstance(subregion_data, (list, tuple)) and len(subregion_data) > i:
        return subregion_data[i]
    return subregion_data


def _cache_by_rounded_accumulation(fn):
    """Memoise on the accumulation vector, rounded to the nearest 10 vehicles."""
    cache = {}

    def cached_call(*args, **kwargs):
        key = tuple(np.round(np.asarray(args[1], dtype=float).reshape(-1), decimals=-1).astype(int))
        if key not in cache:
            if len(cache) > _CACHE_LIMIT:
                cache.clear()
            cache[key] = fn(*args, **kwargs)
        return cache[key]

    cached_call.cache = cache
    return cached_call


def compute_region_travel_times(n_total, region_data, subregion_data=None, cfg: SimulationConfig | None = None) -> np.ndarray:
    """Per-region macroscopic travel time at the given accumulation, via the MFD."""
    cfg = cfg or build_simulation_config()
    n_total = np.asarray(n_total, dtype=float).reshape(-1)
    scale = float(getattr(cfg, "speed_unit_scale", 1.0))
    return np.array([
        max(compute_region_production_and_travel_times(n_total[i], region_data[i], _region_parameters_at_index(subregion_data, i), cfg.CAV_penetration,
                        speed_unit_scale=scale)[1], 0.0)
        for i in range(cfg.nb_region)
    ])


_CLAMP_N_CACHE: dict = {}


@_cache_by_rounded_accumulation
def build_region_cost_matrix(directed_graph, n_total, region_data, subregion_data=None, cfg: SimulationConfig | None = None) -> np.ndarray:
    """Cost of crossing i->j = travel time in i plus travel time in j; np.inf where there is no edge."""
    cfg = cfg or build_simulation_config()
    tt = compute_region_travel_times(n_total, region_data, subregion_data, cfg)
    edge = np.asarray(directed_graph, dtype=float) != 0
    return np.where(edge, tt[:, None] + tt[None, :], np.inf)


@_cache_by_rounded_accumulation
def compute_dijkstra_routes(directed_graph, n_total, region_data, subregion_data=None, cfg: SimulationConfig | None = None, net_cost=None):
    """All-or-nothing assignment: every OD pair sends its whole flow down the shortest path."""
    cfg = cfg or build_simulation_config()
    nb = cfg.nb_region
    if net_cost is None:
        net_cost = build_region_cost_matrix(directed_graph, n_total, region_data, subregion_data, cfg)
    net_cost = np.asarray(net_cost, dtype=float)

    theta, paths_ij, probs_ij, costs_ij = _empty_route_tables(nb)
    for i in range(nb):
        dist, parent = _single_source_dijkstra(net_cost, i)
        for j in range(nb):
            path = _reconstruct_path(parent, i, j) if i != j and np.isfinite(dist[j]) else []
            if len(path) < 2:
                continue
            paths_ij[i][j] = [path]
            costs_ij[i][j] = np.array([float(dist[j])])
            probs_ij[i][j] = np.array([1.0])
            theta[i, j, path[1] - 1] = 1.0
    return theta, paths_ij, probs_ij, costs_ij


def choose_region_path(loc: int, dest: int, is_cav: bool, t_CAV, t_HDV, sp_ij, rng=None):
    """Samples one of the precomputed loc->dest paths according to its probability weights."""
    rng = make_random_state(rng)
    paths = sp_ij[loc - 1][dest - 1]
    probs = np.asarray((t_CAV if is_cav else t_HDV)[loc - 1][dest - 1], dtype=float).reshape(-1)
    if probs.size == 0:
        return paths[0] if paths else [loc, dest]

    probs = np.maximum(probs, 0.0)
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(probs.size, 1.0 / probs.size)
    idx = int(np.searchsorted(np.cumsum(probs), rng.random(), side="left"))
    return paths[idx if idx < len(paths) else 0]


def _single_source_dijkstra(net, source_idx: int):
    """Single-source Dijkstra over a dense cost matrix; non-edges are np.inf."""
    net = np.asarray(net, dtype=float)
    n = net.shape[0]
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    unvisited = np.ones(n, dtype=bool)
    dist[source_idx] = 0.0

    for _ in range(n):
        reachable = np.where(unvisited, dist, np.inf)
        u = int(np.argmin(reachable))
        if not np.isfinite(reachable[u]):
            break
        unvisited[u] = False
        for v in np.where(np.isfinite(net[u]))[0]:
            if dist[u] + net[u, v] < dist[v]:
                dist[v] = dist[u] + net[u, v]
                parent[v] = u
    return dist, parent


def _reconstruct_path(parent, source_idx: int, dest_idx: int) -> list[int]:
    """Walks a parent array back to the source. 1-based node ids, [] if unreachable."""
    if source_idx == dest_idx:
        return [source_idx + 1]
    path = [dest_idx]
    while path[-1] != source_idx:
        prev = int(parent[path[-1]])
        if prev < 0:
            return []
        path.append(prev)
    return [node + 1 for node in reversed(path)]


def shortest_path_and_cost(net_cost_matrix, source: int, destination: int):
    """Shortest path between two 1-based region ids. Returns ``(path, cost)``; ``([], inf)`` if unreachable."""
    s, d = int(source) - 1, int(destination) - 1
    if s == d:
        return [source], 0.0
    dist, parent = _single_source_dijkstra(net_cost_matrix, s)
    return _reconstruct_path(parent, s, d), float(dist[d])


def _empty_route_tables(nb):
    """The four parallel per-OD tables returned by compute_dijkstra_routes."""
    return (
        np.zeros((nb, nb, nb)),
        [[[] for _ in range(nb)] for _ in range(nb)],
        [[np.array([]) for _ in range(nb)] for _ in range(nb)],
        [[np.array([]) for _ in range(nb)] for _ in range(nb)],
    )


_region_cost_matrix_cache = build_region_cost_matrix.cache
_dijkstra_routes_cache = compute_dijkstra_routes.cache
