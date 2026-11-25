"""Event-driven RG-MFD simulation engine."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
import time as _time

import numpy as np

from config.config import SimulationConfig, build_simulation_config
from .matlab_io import find_data_file_path, load_mat_file, save_mat_file
from .demand import create_od_demand_matrix
from .mfd import compute_cav_penetration_factor
from .path_choice import build_action_mask, get_feasible_next_regions_for_vehicle, choose_cav_next_region_with_agent, choose_hdv_next_region
from .routing import build_region_cost_matrix, compute_dijkstra_routes, choose_region_path
from .travelers import COHORT_CODE, build_vehicle_lists
from .utils import get_field_or_attribute


MIN_SPEED = 0.5


BOUNDARY_EDGE_DISCOUNT = 0.01


HDV_LOGIT_SCALE = 0.5


DEPARTURE_TOL = 1e-5
EVENT_TIME_EPS = 1e-6
ARRIVAL_TOL = 1e-4


def _format_elapsed_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes:d}m {sec:02d}s"
    return f"{sec:d}s"


def _print_simulation_progress(count: int, current_time: float, completed: int, total: int, active: int, waiting: int, start_time: float) -> None:
    pct = 100.0 * completed / total if total else 100.0
    elapsed = _format_elapsed_time(_time.time() - start_time)
    print(
        "[progress] "
        f"events={count:,} | "
        f"sim_time={current_time:,.1f}s | "
        f"completed={completed:,}/{total:,} ({pct:5.1f}%) | "
        f"active={active:,} | "
        f"waiting={waiting:,} | "
        f"elapsed={elapsed}",
        flush=True,
    )


def _subregion_chain_by_bfs(start_node: int, end_node: int, adj_mat) -> list[int]:
    if start_node == end_node:
        return []
    adj = np.asarray(adj_mat)
    parent = {start_node: 0}
    queue = deque([start_node])
    while queue:
        u = queue.popleft()
        if u == end_node:
            break
        for nb in np.where(adj[u - 1, :] > 0)[0] + 1:
            nb = int(nb)
            if nb not in parent:
                parent[nb] = u
                queue.append(nb)
    if end_node not in parent:
        return [end_node]
    chain = []
    cur = end_node
    while cur != start_node:
        chain.insert(0, cur)
        cur = parent[cur]
    return chain


TREND_WINDOW = 120.0
TREND_HORIZON = 500.0


TREND_SAMPLE_INTERVAL = 5.0

TREND_CLIP = 2.0


_MASK64 = (1 << 64) - 1


def _subregion_chain_to_region_boundary(curr_sub: int, dest_sub: int, next_region: int, current_speed, cfg: SimulationConfig, history_nodes: set[int]) -> list[int]:
    """Subregion-level Dijkstra-like search from ``curr_sub`` to the nearest subregion"""
    target_subs = [int(idx + 1) for idx, reg in enumerate(cfg.sub2reg) if int(reg) == int(next_region)]
    if not target_subs:
        return _subregion_chain_by_bfs(curr_sub, dest_sub, cfg.sub_dgraph)
    if curr_sub in target_subs:
        return []

    sub_pass_time = np.array([sd.ave_trip_length / max(MIN_SPEED, current_speed[r]) for r, sd in enumerate(cfg.subregion_data)])
    cost = np.full((cfg.nb_subregion, cfg.nb_subregion), np.inf)
    target_set = set(target_subs)
    for r in range(1, cfg.nb_subregion + 1):
        if r in history_nodes and r != curr_sub:
            continue
        for e in np.where(cfg.sub_dgraph[r - 1, :] > 0)[0] + 1:
            e = int(e)
            if e not in history_nodes or e in target_set or e == dest_sub:
                base = sub_pass_time[e - 1]
                cost[r - 1, e - 1] = base * BOUNDARY_EDGE_DISCOUNT if e in target_set else base

    best_chain: list[int] | None = None
    best_cost = np.inf
    for target_sub in target_subs:
        dist = np.full(cfg.nb_subregion, np.inf)
        prev = np.full(cfg.nb_subregion, -1, dtype=int)
        visited = np.zeros(cfg.nb_subregion, dtype=bool)
        dist[curr_sub - 1] = 0.0
        for _ in range(cfg.nb_subregion):
            temp = dist.copy()
            temp[visited] = np.inf
            u = int(np.argmin(temp))
            if not np.isfinite(temp[u]) or u == target_sub - 1:
                break
            visited[u] = True
            for v in np.where(np.isfinite(cost[u, :]))[0]:
                alt = dist[u] + cost[u, v]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        if np.isfinite(dist[target_sub - 1]) and dist[target_sub - 1] < best_cost:
            chain = []
            cur = target_sub - 1
            while cur != curr_sub - 1:
                chain.insert(0, cur + 1)
                cur = prev[cur]
                if cur < 0:
                    chain = []
                    break
            if chain:
                best_chain = chain
                best_cost = float(dist[target_sub - 1])

    if best_chain:
        return best_chain
    for target_sub in target_subs:
        chain = _subregion_chain_by_bfs(curr_sub, target_sub, cfg.sub_dgraph)
        if chain:
            return chain
    return [dest_sub]


def _subregion_chain_to_destination(curr_sub: int, dest_sub: int, target_reg: int, current_speed, cfg: SimulationConfig, history_nodes: set[int]) -> list[int]:
    sub_pass_time = np.array([sd.ave_trip_length / max(MIN_SPEED, current_speed[r]) for r, sd in enumerate(cfg.subregion_data)])
    cost = np.full((cfg.nb_subregion, cfg.nb_subregion), np.inf)
    for r in range(1, cfg.nb_subregion + 1):
        if r in history_nodes and r != curr_sub:
            continue
        for e in np.where(cfg.sub_dgraph[r - 1, :] > 0)[0] + 1:
            e = int(e)
            if e not in history_nodes or e == dest_sub:
                base = sub_pass_time[e - 1]
                cost[r - 1, e - 1] = base * BOUNDARY_EDGE_DISCOUNT if r == curr_sub and cfg.sub2reg[e - 1] == target_reg else base

    dist = np.full(cfg.nb_subregion, np.inf)
    prev = np.full(cfg.nb_subregion, -1, dtype=int)
    visited = np.zeros(cfg.nb_subregion, dtype=bool)
    dist[curr_sub - 1] = 0.0
    for _ in range(cfg.nb_subregion):
        temp = dist.copy()
        temp[visited] = np.inf
        u = int(np.argmin(temp))
        if not np.isfinite(temp[u]) or u == dest_sub - 1:
            break
        visited[u] = True
        for v in np.where(np.isfinite(cost[u, :]))[0]:
            alt = dist[u] + cost[u, v]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    if prev[dest_sub - 1] != -1:
        rem_chain = []
        cur = dest_sub - 1
        while cur != curr_sub - 1:
            rem_chain.insert(0, cur + 1)
            cur = prev[cur]
        return rem_chain
    return _subregion_chain_by_bfs(curr_sub, dest_sub, cfg.sub_dgraph)


class ActiveVehicles:
    """Struct-of-arrays view over the active vehicle records."""

    def __init__(self, vehicles, cfg: SimulationConfig, N_mat_in=None):
        self.vehicles = self._to_vehicle_object_array(vehicles)
        self.cfg = cfg
        self.rebuild_arrays_from_vehicles(N_mat_in)

    @staticmethod
    def _to_vehicle_object_array(vehicles):
        if isinstance(vehicles, np.ndarray) and vehicles.dtype == object:
            return vehicles.reshape(-1)
        return np.asarray(vehicles, dtype=object).reshape(-1)

    def __len__(self):
        return self.vehicles.size

    def __bool__(self):
        return bool(self.vehicles.size)

    def __iter__(self):
        return iter(self.vehicles)

    def add_vehicles(self, vehicles):
        incoming = self._to_vehicle_object_array(vehicles)
        if incoming.size == 0:
            return
        incoming_data = ActiveVehicles(incoming, self.cfg)
        self.vehicles = np.concatenate((self.vehicles, incoming_data.vehicles))
        self.origin_reg = np.concatenate((self.origin_reg, incoming_data.origin_reg))
        self.current_location_reg = np.concatenate((self.current_location_reg, incoming_data.current_location_reg))
        self.destination_reg = np.concatenate((self.destination_reg, incoming_data.destination_reg))
        self.current_sub = np.concatenate((self.current_sub, incoming_data.current_sub))
        self.destination_sub = np.concatenate((self.destination_sub, incoming_data.destination_sub))
        self.current_leg_id = np.concatenate((self.current_leg_id, incoming_data.current_leg_id))
        self.is_cav = np.concatenate((self.is_cav, incoming_data.is_cav))
        self.is_fleet1 = np.concatenate((self.is_fleet1, incoming_data.is_fleet1))
        self.dynamic_sub_seq = np.concatenate((self.dynamic_sub_seq, incoming_data.dynamic_sub_seq))
        self.dynamic_sub_seq_len = np.concatenate((self.dynamic_sub_seq_len, incoming_data.dynamic_sub_seq_len))
        self.trip_L_remain = np.concatenate((self.trip_L_remain, incoming_data.trip_L_remain))
        self.remain = np.concatenate((self.remain, incoming_data.remain))
        self.t_next_arr = np.concatenate((self.t_next_arr, incoming_data.t_next_arr))
        self.N_mat_init = self._build_initial_accumulation_matrix(None)

    def select_vehicles_by_mask(self, mask):
        out = object.__new__(ActiveVehicles)
        out.cfg = self.cfg
        out.vehicles = self.vehicles[mask]
        out.origin_reg = self.origin_reg[mask].copy()
        out.current_location_reg = self.current_location_reg[mask].copy()
        out.destination_reg = self.destination_reg[mask].copy()
        out.current_sub = self.current_sub[mask].copy()
        out.destination_sub = self.destination_sub[mask].copy()
        out.current_leg_id = self.current_leg_id[mask].copy()
        out.is_cav = self.is_cav[mask].copy()
        out.is_fleet1 = self.is_fleet1[mask].copy()
        out.dynamic_sub_seq = self.dynamic_sub_seq[mask].copy()
        out.dynamic_sub_seq_len = self.dynamic_sub_seq_len[mask].copy()
        out.trip_L_remain = self.trip_L_remain[mask].copy()
        out.remain = self.remain[mask].copy()
        out.t_next_arr = self.t_next_arr[mask].copy()
        out.N_mat_init = out._build_initial_accumulation_matrix(None)
        return out

    def rebuild_arrays_from_vehicles(self, N_mat_in=None):
        size = self.vehicles.size
        self.origin_reg = np.empty(size, dtype=np.int32)
        self.current_location_reg = np.empty(size, dtype=np.int32)
        self.destination_reg = np.empty(size, dtype=np.int32)
        self.current_sub = np.empty(size, dtype=np.int32)
        self.destination_sub = np.empty(size, dtype=np.int32)
        self.current_leg_id = np.empty(size, dtype=np.int32)
        self.is_cav = np.empty(size, dtype=bool)
        self.is_fleet1 = np.empty(size, dtype=bool)
        self.dynamic_sub_seq = np.empty(size, dtype=object)
        self.dynamic_sub_seq_len = np.empty(size, dtype=np.int32)
        self.trip_L_remain = np.empty(size, dtype=object)
        self.remain = np.zeros(size, dtype=float)
        self.t_next_arr = np.empty(size, dtype=float)
        for idx, veh in enumerate(self.vehicles):
            self.refresh_arrays_for_vehicle(idx, veh)
        self.N_mat_init = self._build_initial_accumulation_matrix(N_mat_in)

    def next_region_arrival_times_after(self, current_time, eps_time):
        if not self:
            return np.empty(0, dtype=float)
        return self.t_next_arr[self.t_next_arr > float(current_time) + float(eps_time)]

    def count_vehicles_per_subregion(self):


        if not self.current_sub.size:
            return np.zeros(self.cfg.nb_subregion, dtype=float)
        idx = (self.current_sub - 1).astype(np.intp)
        return np.bincount(idx, minlength=self.cfg.nb_subregion).astype(float)

    def split_off_completed_vehicles(self, current_time=None):
        if not self:
            return [], self
        completed_mask = (self.current_leg_id > self.dynamic_sub_seq_len) & (self.current_sub == self.destination_sub)
        if not np.any(completed_mask):
            return [], self
        completed_idx = np.flatnonzero(completed_mask)
        if current_time is not None:
            current_time = float(current_time)
            for idx in completed_idx:
                veh = self.vehicles[idx]
                num_seq = int(self.dynamic_sub_seq_len[idx])
                veh.t_arr_subreg = [current_time] * num_seq
                veh.t_arr = current_time
                veh.t_next_arr = float("inf")
        completed = self.vehicles[completed_mask].tolist()
        return completed, self.select_vehicles_by_mask(~completed_mask)

    def refresh_arrays_for_vehicle(self, idx, veh=None):
        veh = self.vehicles[idx] if veh is None else veh
        if "dynamic_sub_seq" not in veh:
            veh.dynamic_sub_seq = list(veh.subregions_thru_expected)
        dynamic_sub_seq = np.asarray(veh.dynamic_sub_seq, dtype=np.int32).reshape(-1)
        trip_L_remain = np.asarray(veh.trip_L_remain, dtype=float).reshape(-1)
        curr_leg_idx = int(veh.curr_leg_idx)

        self.origin_reg[idx] = int(veh.Origin_reg)
        self.current_location_reg[idx] = int(veh.current_location_reg)
        self.destination_reg[idx] = int(veh.Destination_reg)
        self.current_sub[idx] = int(veh.current_location_sub)
        self.destination_sub[idx] = int(veh.Destination_sub)
        self.current_leg_id[idx] = curr_leg_idx
        self.is_cav[idx] = bool(veh.get("is_cav", False))
        self.is_fleet1[idx] = veh.get("vehicle_type", "") == "Fleet1"
        self.dynamic_sub_seq[idx] = dynamic_sub_seq
        self.dynamic_sub_seq_len[idx] = dynamic_sub_seq.size
        self.trip_L_remain[idx] = trip_L_remain
        if 1 <= curr_leg_idx <= trip_L_remain.size:
            self.remain[idx] = trip_L_remain[curr_leg_idx - 1]
        else:
            self.remain[idx] = 0.0
        self.t_next_arr[idx] = float(veh.t_next_arr)

    def _build_initial_accumulation_matrix(self, N_mat_in):
        if N_mat_in is not None:
            return np.asarray(N_mat_in, dtype=float)


        nb_region = self.cfg.nb_region
        c_r = self.current_location_reg
        d_r = self.destination_reg
        flat_idx = (c_r - 1).astype(np.intp) * nb_region + (d_r - 1).astype(np.intp)
        N_mat_init = np.bincount(flat_idx, minlength=nb_region * nb_region).astype(float)
        return N_mat_init.reshape(nb_region, nb_region)

    def active_vehicle_mask(self):
        return self.current_leg_id <= self.dynamic_sub_seq_len

    def recalculate_remaining_distance(self, previous_speed, dt):
        speed = previous_speed[self.current_sub - 1]
        remain = np.maximum(
            0,
            np.subtract(self.remain, np.multiply(speed, dt)),
        )
        return remain

    def store_remaining_distance(self, remain, active):
        self.remain[active] = remain[active]

    def sync_remaining_distance_on_arrival(self, idx):
        self.trip_L_remain[idx][self.current_leg_id[idx] - 1] = self.remain[idx]

    def mark_arrived_at_destination(self, idx):
        veh = self.vehicles[idx]
        leg_id = int(self.current_leg_id[idx])
        dest_sub = int(self.destination_sub[idx])
        dest_reg = int(self.destination_reg[idx])

        trip_L_remain = self.trip_L_remain[idx].copy()
        trip_L_remain[leg_id - 1] = 0.0
        self.trip_L_remain[idx] = trip_L_remain
        veh.trip_L_remain = trip_L_remain.tolist()
        veh.current_location_sub = dest_sub
        veh.current_location_reg = dest_reg
        veh.curr_leg_idx = leg_id + 1
        self.refresh_arrays_for_vehicle(idx, veh)

    def replace_remaining_route(self, idx, rem_chain, new_rem_L, current_time, dt, reg_after):
        veh = self.vehicles[idx]
        leg_id = int(self.current_leg_id[idx])
        next_sub = int(rem_chain[0])
        curr_reg = int(self.current_location_reg[idx])

        dynamic_sub_seq = np.concatenate(
            (self.dynamic_sub_seq[idx][:leg_id], np.asarray(rem_chain, dtype=np.int32))
        )
        trip_L_remain = np.concatenate(
            (self.trip_L_remain[idx][:leg_id], np.asarray(new_rem_L, dtype=float))
        )

        veh.dynamic_sub_seq = dynamic_sub_seq.tolist()
        veh.trip_L_remain = trip_L_remain.tolist()
        veh.current_location_sub = next_sub
        veh.current_location_reg = int(reg_after)
        veh.actual_path_recorded = list(veh.actual_path_recorded) + [next_sub]
        veh.sub_entry_times = list(veh.sub_entry_times) + [float(current_time)]
        veh.sub_durations = list(veh.sub_durations) + [
            float(current_time - veh.sub_entry_times[-2]) if len(veh.sub_entry_times) >= 2 else dt
        ]
        if curr_reg != int(reg_after):
            veh.actual_region_recorded = list(veh.actual_region_recorded) + [int(reg_after)]
            veh.reg_boundary_times = list(veh.reg_boundary_times) + [float(current_time)]
        veh.curr_leg_idx = leg_id + 1
        self.refresh_arrays_for_vehicle(idx, veh)

    def update_next_arrival_times(self, current_speed, current_time):
        current_speed = np.asarray(current_speed, dtype=float).reshape(-1)
        active = self.active_vehicle_mask()
        self.t_next_arr.fill(float("inf"))
        if not np.any(active):
            return
        active_idx = np.flatnonzero(active)
        speed = np.maximum(MIN_SPEED, current_speed[self.current_sub[active_idx] - 1])
        self.t_next_arr[active_idx] = float(current_time) + self.remain[active_idx] / speed

def as_active_vehicles(vehicles, cfg: SimulationConfig, N_mat_in=None):
    if isinstance(vehicles, ActiveVehicles):
        if vehicles.cfg is not cfg:
            vehicles.cfg = cfg
        if N_mat_in is not None:
            vehicles.N_mat_init = np.asarray(N_mat_in, dtype=float)
        return vehicles
    return ActiveVehicles(vehicles, cfg, N_mat_in)


def build_departure_queue(vehicles):
    return deque(sorted(vehicles, key=lambda veh: veh.t_dep))


def next_departure_time(queue, current_time, eps_time):
    stale_before = float(current_time) - float(eps_time)
    while queue and queue[0].t_dep < stale_before:
        queue.popleft()
    return float(queue[0].t_dep) if queue else float("inf")


def pop_departures_due_at(queue, current_time, tol=DEPARTURE_TOL):
    entering = []
    while queue and abs(queue[0].t_dep - current_time) < tol:
        entering.append(queue.popleft())
    return entering


def build_observation(vehicle, region_accumulation, action_mask, cfg):
    nb = cfg.nb_region
    current_region = int(get_field_or_attribute(vehicle, "current_location_reg", get_field_or_attribute(vehicle, "current_location_sub", 1)))
    destination_region = int(get_field_or_attribute(vehicle, "Destination_reg", get_field_or_attribute(vehicle, "Destination_sub", 1)))
    current_region_one_hot = np.zeros(nb)
    destination_region_one_hot = np.zeros(nb)
    if 1 <= current_region <= nb:
        current_region_one_hot[current_region - 1] = 1.0
    if 1 <= destination_region <= nb:
        destination_region_one_hot[destination_region - 1] = 1.0
    density = np.array([region_accumulation[i] / cfg.region_data[i].njamMFD for i in range(nb)])
    return np.concatenate([current_region_one_hot, destination_region_one_hot, density,
                           np.asarray(action_mask, dtype=float).reshape(-1)])


def observation_dimension(cfg):
    return 4 * cfg.nb_region


def advance_vehicles_and_route(active_vehicles, previous_time, current_time, current_speed,
                              region_data, current_sub_n, cfg, rng, rl_agent=None, use_rl_cav=False):
    current_speed = np.asarray(current_speed, dtype=float).reshape(-1)
    current_sub_n = np.asarray(current_sub_n, dtype=float).reshape(-1)
    vdata = as_active_vehicles(active_vehicles, cfg)
    current_location_reg = vdata.current_location_reg
    current_sub = vdata.current_sub

    n_total_macro = np.zeros(cfg.nb_region)
    for reg in range(1, cfg.nb_region + 1):
        n_total_macro[reg - 1] = current_sub_n[cfg.sub2reg == reg].sum()


    region_net_cost = build_region_cost_matrix(cfg.reg_dgraph, n_total_macro, region_data, cfg.subregion_data, cfg)
    _theta, shortest_paths_ij, prob_path_ij, _costs = compute_dijkstra_routes(
        cfg.reg_dgraph, n_total_macro, region_data, cfg.subregion_data, cfg, net_cost=region_net_cost)

    network_state = {
        "cfg": cfg,
        "current_time": float(current_time),
        "current_reg_n": n_total_macro.copy(),
        "region_data": region_data,
        "shortest_paths_ij": shortest_paths_ij,
        "prob_path_ij": prob_path_ij,
        "region_net_cost": region_net_cost,
        "rng": rng,
        "logit_scale": HDV_LOGIT_SCALE,
    }
    if not vdata:
        return vdata

    dt = max(0.0, float(current_time - previous_time))
    active = vdata.active_vehicle_mask()
    remain = vdata.recalculate_remaining_distance(current_speed, dt)
    vdata.store_remaining_distance(remain, active)

    for idx in np.flatnonzero(active & (remain <= ARRIVAL_TOL)):
        vdata.sync_remaining_distance_on_arrival(idx)
        veh = vdata.vehicles[idx]
        curr_sub = int(current_sub[idx])
        dest_sub = int(vdata.destination_sub[idx])
        if curr_sub == dest_sub:
            vdata.mark_arrived_at_destination(idx)
            continue
        curr_reg = int(current_location_reg[idx])
        dest_reg = int(vdata.destination_reg[idx])
        is_cav = bool(vdata.is_cav[idx])
        history_nodes = set(int(x) for x in veh.get("actual_path_recorded", []))
        target_reg = dest_reg
        if int(vdata.origin_reg[idx]) != dest_reg and curr_reg != dest_reg:
            feasible = get_feasible_next_regions_for_vehicle(veh, cfg) if use_rl_cav else []
            if use_rl_cav and is_cav:
                action_mask = build_action_mask(feasible, cfg.nb_region)
                state = dict(network_state)
                state["observation"] = build_observation(veh, n_total_macro, action_mask, cfg)
                chosen_region = choose_cav_next_region_with_agent(veh, state, feasible, rl_agent)
            elif use_rl_cav:
                chosen_region = choose_hdv_next_region(veh, feasible, network_state)
            else:
                path = choose_region_path(curr_reg, dest_reg, is_cav, prob_path_ij, prob_path_ij, shortest_paths_ij, rng)
                chosen_region = int(path[1]) if len(path) >= 2 else None
            if chosen_region is not None:
                target_reg = int(chosen_region)

        if use_rl_cav:
            rem_chain = _subregion_chain_to_region_boundary(curr_sub, dest_sub, target_reg, current_speed, cfg, history_nodes)
        else:
            rem_chain = _subregion_chain_to_destination(curr_sub, dest_sub, target_reg, current_speed, cfg, history_nodes)
        if not rem_chain:
            rem_chain = [dest_sub]
        reg_after = int(cfg.sub2reg[int(rem_chain[0]) - 1])
        new_rem_L = np.empty(len(rem_chain), dtype=float)
        for k, s_id in enumerate(rem_chain):
            base = cfg.subregion_data[s_id - 1].ave_trip_length
            new_rem_L[k] = float(base * rng.random()) if s_id == dest_sub else base
        vdata.replace_remaining_route(idx, rem_chain, new_rem_L, current_time, dt, reg_after)

    vdata.update_next_arrival_times(current_speed, current_time)
    return vdata


def _mfd_speed_from_accumulation(sd, n, cfg):
    return (sd.ave_trip_length * (sd.aMFD * n**2 + sd.bMFD * n + sd.cMFD)
            * compute_cav_penetration_factor(sd.dMFD, sd.eMFD, sd.fMFD, cfg.CAV_penetration)
            * getattr(cfg, "speed_unit_scale", 1.0))


def run_episode(data_dir, rl_agent=None, rng_seed=42, future_demand_scale=1.0,
             cav_penetration=None, ban_revisit=True, max_events=None,
             show_progress=True, progress_interval_seconds=30.0):
    """Run one episode on the 7-region network in ``data_dir``."""
    data_dir = str(data_dir)
    cfg = build_simulation_config(data_dir)
    if cav_penetration is not None:
        cfg.CAV_penetration = float(cav_penetration)
    cfg.ban_revisit = bool(ban_revisit)
    cfg.ban_revisit_count_current_region = True
    cfg.hdv_route_full_count = True
    use_rl_cav = rl_agent is not None
    rng = np.random.RandomState(rng_seed)
    nb = cfg.nb_region

    demand_od = create_od_demand_matrix(cfg, data_dir=data_dir) * future_demand_scale
    xinit = load_mat_file(find_data_file_path(f"xInit_ss_{nb}.mat", data_dir, __file__))[f"xInit_ss_{nb}"]
    initial_acc = cfg.initial_accumulation_scale * np.asarray(xinit, dtype=float)
    initial_sub_acc = initial_acc.sum(axis=1)
    speeds = np.array([_mfd_speed_from_accumulation(sd, initial_sub_acc[i], cfg) for i, sd in enumerate(cfg.subregion_data)])

    info = build_vehicle_lists(demand_od, initial_sub_acc, speeds, cfg.subregion_data, False, None, cfg, rng_seed)
    active = as_active_vehicles(info["traveler_info_initacc_Ri_all"], cfg)
    q_int = build_departure_queue(info["traveler_info_demand_Rii_all"])
    q_ext = build_departure_queue(info["traveler_info_demand_Rij_all"])
    total = len(active) + len(q_int) + len(q_ext)

    current_time, count, completed = cfg.t_init, 0, []
    start = _time.time()
    next_print = start + progress_interval_seconds
    while active or q_int or q_ext:
        if max_events is not None and count >= int(max_events):
            break
        t_int = next_departure_time(q_int, current_time, EVENT_TIME_EPS)
        t_ext = next_departure_time(q_ext, current_time, EVENT_TIME_EPS)
        nxt = active.next_region_arrival_times_after(current_time, EVENT_TIME_EPS)
        t_next = min(t_int, t_ext, float(nxt.min()) if nxt.size else float("inf"))
        if not np.isfinite(t_next):
            break
        prev_time, current_time = current_time, t_next
        if abs(current_time - t_int) < DEPARTURE_TOL:
            active.add_vehicles(pop_departures_due_at(q_int, current_time))
        if abs(current_time - t_ext) < DEPARTURE_TOL:
            active.add_vehicles(pop_departures_due_at(q_ext, current_time))

        sub_n = active.count_vehicles_per_subregion()
        speed = np.array([max(_mfd_speed_from_accumulation(sd, sub_n[i], cfg), MIN_SPEED) for i, sd in enumerate(cfg.subregion_data)])
        if use_rl_cav and hasattr(rl_agent, "observe_event_reward"):
            rl_agent.observe_event_reward(-(current_time - prev_time) * len(active))
        active = advance_vehicles_and_route(active, prev_time, current_time, speed, cfg.region_data,
                                           sub_n, cfg, rng, rl_agent, use_rl_cav)
        done, active = active.split_off_completed_vehicles(current_time)
        if use_rl_cav and hasattr(rl_agent, "observe_arrival"):
            for vehicle in done:
                if get_field_or_attribute(vehicle, "is_cav", False):
                    rl_agent.observe_arrival(vehicle)
        completed.extend(done)
        count += 1
        if show_progress and _time.time() >= next_print:
            _print_simulation_progress(count, current_time, len(completed), total, len(active),
                                len(q_int) + len(q_ext), start)
            next_print = _time.time() + progress_interval_seconds

    if use_rl_cav and hasattr(rl_agent, "end_episode"):
        rl_agent.end_episode()
    return {"vehicles": completed, "total_vehicles": int(total), "events": count,
            "sim_time": float(current_time), "elapsed_seconds": _time.time() - start}
