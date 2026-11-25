from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from config.config import SimulationConfig, build_simulation_config
from .demand import create_departure_times
from .matlab_io import load_mat_file, save_mat_file
from .utils import AttrDict, make_random_state


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    return [value]


def _normalize_loaded_vehicle_record(veh):
    for name in [
        "subregions_thru_expected",
        "regions_thru_expected",
        "dynamic_sub_seq",
        "trip_L_subreg",
        "trip_L_remain",
        "t_arr_subreg",
        "actual_path_recorded",
        "actual_region_recorded",
        "reg_boundary_times",
        "sub_entry_times",
        "sub_durations",
    ]:
        if name in veh:
            veh[name] = _ensure_list(veh[name])
    for name in [
        "veh_id",
        "uid",
        "Origin_sub",
        "Destination_sub",
        "current_location_sub",
        "Origin_reg",
        "Destination_reg",
        "current_location_reg",
        "curr_leg_idx",
    ]:
        if name in veh:
            veh[name] = int(veh[name])
    for name in ["t_dep", "t_arr", "t_next_arr"]:
        if name in veh:
            veh[name] = float(veh[name])
    if "is_cav" in veh:
        veh.is_cav = bool(veh.is_cav)
    if "vehicle_type" in veh and isinstance(veh.vehicle_type, np.ndarray):
        veh.vehicle_type = "".join(str(x) for x in veh.vehicle_type.reshape(-1).tolist())
    if "cohort" in veh and isinstance(veh.cohort, np.ndarray):
        veh.cohort = "".join(str(x) for x in veh.cohort.reshape(-1).tolist())
    return veh


COHORT_KEYS = {
    "init_acc": "traveler_info_initacc_Ri_all",
    "Rii": "traveler_info_demand_Rii_all",
    "Rij": "traveler_info_demand_Rij_all",
}
COHORT_ORDER = ("init_acc", "Rii", "Rij")
COHORT_CODE = {name: code for code, name in enumerate(COHORT_ORDER)}


def assign_vehicle_uids(traveler_info):
    """Stamp every traveler with a globally unique ``uid`` and its ``cohort`` name."""
    uid = 0
    for cohort in COHORT_ORDER:
        for veh in traveler_info.get(COHORT_KEYS[cohort], []):
            uid += 1
            veh.uid = uid
            veh.cohort = cohort
    return traveler_info


def _normalize_loaded_vehicle_records(data):
    out = {}
    for name in COHORT_KEYS.values():
        group = data.get(name, [])
        if isinstance(group, AttrDict):
            group = [group]
        elif not isinstance(group, list):
            group = _ensure_list(group)
        out[name] = [_normalize_loaded_vehicle_record(veh) for veh in group]

    if any("uid" not in veh for group in out.values() for veh in group):
        assign_vehicle_uids(out)
    return out


def _region_path_by_bfs(ori: int, dest: int, adj_matrix) -> list[int]:
    if ori == dest:
        return [ori]
    adj = np.asarray(adj_matrix)
    parent = {ori: 0}
    queue = deque([ori])
    while queue:
        u = queue.popleft()
        if u == dest:
            break
        for nb in np.where(adj[u - 1, :] > 0)[0] + 1:
            nb = int(nb)
            if nb not in parent:
                parent[nb] = u
                queue.append(nb)
    if dest not in parent:
        return [ori, dest]
    path = []
    cur = dest
    while cur != 0:
        path.insert(0, cur)
        cur = parent[cur]
    return path


def build_vehicle_records_from_demand(
    ori_sub: int,
    des_sub: int,
    flag_init_acc: int,
    flag_Qii: int,
    nb_vehs: int,
    departure_t_sub,
    sub_speed,
    n_sub_total,
    subregion_data,
    cfg: SimulationConfig | None = None,
    rng=None,
):
    cfg = cfg or build_simulation_config()
    rng = make_random_state(rng)
    nb_vehs = int(round(float(nb_vehs)))
    if nb_vehs <= 0:
        return []

    departure_t_sub = np.asarray(departure_t_sub, dtype=float).reshape(-1)
    sub_speed = np.asarray(sub_speed, dtype=float).reshape(-1)
    n_sub_total = np.asarray(n_sub_total, dtype=float).reshape(-1)

    num_cav = int(round(nb_vehs * cfg.CAV_penetration))
    num_fleet1 = int(round(num_cav * getattr(cfg, "fleet1_ratio", 0.5)))
    num_fleet2 = num_cav - num_fleet1
    num_hdv = max(0, nb_vehs - num_cav)

    veh_type_list = ["Fleet1"] * num_fleet1 + ["Fleet2"] * num_fleet2 + ["HDV"] * num_hdv
    if len(veh_type_list) > nb_vehs:
        veh_type_list = veh_type_list[:nb_vehs]
    while len(veh_type_list) < nb_vehs:
        veh_type_list.append("HDV")

    rng.shuffle(veh_type_list)

    generated_destination = []
    if flag_init_acc == 1:
        for m in range(1, cfg.nb_subregion + 1):
            count = int(round(n_sub_total[m - 1])) if m - 1 < n_sub_total.size else 0
            generated_destination.extend([m] * max(0, count))
        rng.shuffle(generated_destination)
        if len(generated_destination) < nb_vehs:
            fallback_dest = des_sub if des_sub > 0 else ori_sub
            generated_destination.extend([fallback_dest] * (nb_vehs - len(generated_destination)))

    ori_reg = int(cfg.sub2reg[ori_sub - 1])
    travelers = []
    for i in range(nb_vehs):
        veh = AttrDict()
        veh.veh_id = i + 1
        veh.vehicle_type = veh_type_list[i]
        veh.is_cav = veh.vehicle_type in ("Fleet1", "Fleet2")
        if flag_init_acc == 1:
            act_dest_sub = int(generated_destination[i]) if generated_destination else ori_sub
            veh.t_dep = 0.0
        else:
            act_dest_sub = ori_sub if flag_Qii == 1 else int(des_sub)
            veh.t_dep = float(departure_t_sub[i]) if i < departure_t_sub.size else 0.0
        if act_dest_sub <= 0:
            act_dest_sub = ori_sub
        des_reg = int(cfg.sub2reg[act_dest_sub - 1])

        veh.Origin_sub = int(ori_sub)
        veh.Destination_sub = int(act_dest_sub)
        veh.current_location_sub = int(ori_sub)
        veh.Origin_reg = ori_reg
        veh.Destination_reg = des_reg
        veh.current_location_reg = ori_reg

        subreg_seq = _region_path_by_bfs(int(ori_sub), int(act_dest_sub), cfg.sub_dgraph)
        veh.subregions_thru_expected = subreg_seq.copy()
        raw_reg_seq = [int(cfg.sub2reg[s - 1]) for s in subreg_seq]
        reg_thru = [raw_reg_seq[0]]
        for reg in raw_reg_seq[1:]:
            if reg != reg_thru[-1]:
                reg_thru.append(reg)
        veh.regions_thru_expected = reg_thru
        veh.dynamic_sub_seq = subreg_seq.copy()

        trip_L_vec = []
        for k, curr_s_id in enumerate(subreg_seq):
            base_len = subregion_data[curr_s_id - 1].ave_trip_length
            if len(subreg_seq) == 1:
                trip_L_vec.append(2 * base_len * rng.random())
            elif k == 0 or k == len(subreg_seq) - 1:
                trip_L_vec.append(2 * base_len * rng.random())
            else:
                trip_L_vec.append(base_len)
        veh.trip_L_subreg = list(map(float, trip_L_vec))
        veh.trip_L_remain = list(map(float, trip_L_vec))

        t_arr_vec = []
        t_accum = float(veh.t_dep)
        for s_id, length in zip(subreg_seq, trip_L_vec):
            v_val = max(0.5, float(sub_speed[s_id - 1]))
            t_accum += length / v_val
            t_arr_vec.append(float(t_accum))
        veh.t_arr_subreg = t_arr_vec
        veh.t_arr = t_arr_vec[-1]
        veh.actual_path_recorded = [int(ori_sub)]
        veh.actual_region_recorded = [ori_reg]
        veh.reg_boundary_times = []
        veh.sub_entry_times = [float(veh.t_dep)]
        veh.sub_durations = []
        veh.curr_leg_idx = 1
        veh.t_next_arr = t_arr_vec[0]
        travelers.append(veh)
    return travelers


def build_vehicle_lists(Q_S, n_sub_initial, V_n_sub, subregion_data, flag_change_travelers=True, cache_path: str | Path | None = None, cfg: SimulationConfig | None = None, rng_seed: int = 42):
    cfg = cfg or build_simulation_config()
    cache_path = Path(cache_path) if cache_path is not None else None
    if not flag_change_travelers and cache_path is not None and cache_path.exists():
        return _normalize_loaded_vehicle_records(load_mat_file(cache_path))

    rng = np.random.RandomState(rng_seed)
    departure_t_Rij = create_departure_times(Q_S, cfg.nb_subregion, 0.0, cfg.tfinq, rng_seed)
    init_all = []
    demand_rii_all = []
    demand_rij_all = []
    for w in range(1, cfg.nb_subregion + 1):
        init_all.extend(
            build_vehicle_records_from_demand(w, 0, 1, 0, n_sub_initial[w - 1], departure_t_Rij[w - 1][w - 1], V_n_sub, n_sub_initial, subregion_data, cfg, rng)
        )
        for v in range(1, cfg.nb_subregion + 1):
            deps = departure_t_Rij[w - 1][v - 1]
            travelers = build_vehicle_records_from_demand(w, v, 2, 1 if w == v else 2, len(deps), deps, V_n_sub, n_sub_initial, subregion_data, cfg, rng)
            if w == v:
                demand_rii_all.extend(travelers)
            else:
                demand_rij_all.extend(travelers)
    for i, veh in enumerate(init_all, start=1):
        veh.veh_id = i
    for i, veh in enumerate(demand_rii_all, start=1):
        veh.veh_id = i
    for i, veh in enumerate(demand_rij_all, start=1):
        veh.veh_id = i

    out = {
        "traveler_info_initacc_Ri_all": init_all,
        "traveler_info_demand_Rii_all": demand_rii_all,
        "traveler_info_demand_Rij_all": demand_rij_all,
    }
    assign_vehicle_uids(out)
    if cache_path is not None:
        save_mat_file(cache_path, **out)
    for group in out.values():
        if isinstance(group, list):
            for veh in group:
                veh.actual_path_recorded = [veh.Origin_sub]
    return out
