from __future__ import annotations

from pathlib import Path

import numpy as np

from config.config import SimulationConfig, build_simulation_config
from .matlab_io import find_data_file_path, load_mat_file, save_mat_file
from .mfd import draw_trip_length
from .utils import make_random_state


MACRO_DEMAND = np.array([[0.1, 0.2, 0.2, 0.4, 0.5, 0.45, 0.43, 0.3, 0.3, 0.2, 0.1, 0.1], [0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4, 0.35, 0.24, 0.14, 0.1], [0.1, 0.1, 0.16, 0.22, 0.4, 0.5, 0.5, 0.37, 0.27, 0.25, 0.15, 0.1], [0.2, 0.3, 0.3, 0.42, 0.5, 0.56, 0.45, 0.2, 0.1, 0.1, 0.09, 0.05], [0.25, 0.3, 0.31, 0.4, 0.45, 0.43, 0.32, 0.24, 0.19, 0.11, 0.1, 0.07], [0.1, 0.15, 0.15, 0.22, 0.31, 0.43, 0.42, 0.3, 0.22, 0.2, 0.1, 0.05], [0.3, 0.33, 0.35, 0.58, 0.5, 0.43, 0.32, 0.2, 0.1, 0.07, 0.06, 0.04], [0.1, 0.2, 0.25, 0.26, 0.45, 0.44, 0.4, 0.36, 0.32, 0.24, 0.14, 0.08], [0.2, 0.2, 0.3, 0.5, 0.51, 0.46, 0.42, 0.33, 0.27, 0.16, 0.13, 0.07], [0.2, 0.25, 0.27, 0.35, 0.45, 0.53, 0.58, 0.33, 0.25, 0.15, 0.12, 0.05], [0.2, 0.3, 0.3, 0.35, 0.45, 0.42, 0.31, 0.37, 0.26, 0.17, 0.17, 0.13], [0.1, 0.1, 0.2, 0.25, 0.3, 0.35, 0.46, 0.37, 0.23, 0.16, 0.15, 0.1], [0.2, 0.3, 0.3, 0.35, 0.65, 0.42, 0.21, 0.19, 0.11, 0.09, 0.04, 0.04], [0.1, 0.13, 0.25, 0.31, 0.43, 0.62, 0.41, 0.32, 0.25, 0.18, 0.12, 0.06], [0.1, 0.1, 0.15, 0.38, 0.35, 0.34, 0.22, 0.15, 0.13, 0.11, 0.09, 0.06], [0.2, 0.3, 0.56, 0.66, 0.65, 0.62, 0.41, 0.37, 0.32, 0.22, 0.13, 0.12], [0.4, 0.5, 0.58, 0.72, 0.66, 0.62, 0.51, 0.42, 0.32, 0.21, 0.12, 0.12], [0.3, 0.6, 0.73, 0.65, 0.61, 0.52, 0.37, 0.35, 0.27, 0.27, 0.17, 0.13], [0.1, 0.4, 0.47, 0.52, 0.41, 0.32, 0.21, 0.19, 0.13, 0.11, 0.11, 0.11], [0.2, 0.3, 0.49, 0.55, 0.49, 0.42, 0.34, 0.21, 0.21, 0.13, 0.07, 0.05], [1.2, 1.7, 1.91, 2.12, 2.22, 1.71, 1.52, 1.38, 0.98, 0.76, 0.62, 0.45], [0.2, 0.4, 0.5, 0.6, 0.6, 0.6, 0.42, 0.34, 0.28, 0.21, 0.15, 0.13], [0.2, 0.2, 0.5, 0.5, 0.4, 0.4, 0.31, 0.29, 0.23, 0.12, 0.11, 0.1], [0.3, 0.4, 0.7, 0.8, 0.7, 0.6, 0.53, 0.51, 0.34, 0.26, 0.12, 0.1], [0.1, 0.3, 0.8, 0.7, 0.9, 0.7, 0.61, 0.49, 0.21, 0.13, 0.11, 0.1], [0.1, 0.4, 0.7, 0.9, 0.7, 0.7, 0.56, 0.47, 0.24, 0.21, 0.11, 0.1], [0.2, 0.3, 0.3, 0.5, 0.4, 0.3, 0.26, 0.26, 0.19, 0.16, 0.11, 0.1], [1.3, 1.4, 1.6, 1.7, 1.8, 1.6, 1.46, 1.24, 0.95, 0.65, 0.41, 0.23], [0.1, 0.1, 0.3, 0.4, 0.4, 0.33, 0.32, 0.26, 0.22, 0.16, 0.11, 0.09], [0.2, 0.3, 0.4, 0.5, 0.5, 0.55, 0.41, 0.31, 0.27, 0.15, 0.11, 0.09], [0.1, 0.2, 0.5, 0.6, 0.5, 0.38, 0.33, 0.25, 0.24, 0.11, 0.11, 0.03], [0.3, 0.5, 0.7, 0.9, 0.8, 0.64, 0.54, 0.47, 0.39, 0.23, 0.16, 0.15], [0.2, 0.3, 0.5, 0.6, 0.6, 0.47, 0.31, 0.22, 0.24, 0.15, 0.13, 0.12], [0.3, 0.5, 0.7, 0.7, 0.6, 0.73, 0.67, 0.46, 0.39, 0.29, 0.15, 0.12], [0.9, 1.1, 1.2, 1.5, 1.4, 1.33, 1.19, 0.89, 0.52, 0.33, 0.21, 0.13], [0.2, 0.3, 0.5, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.07], [0.1, 0.1, 0.3, 0.4, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.07, 0.03], [0.3, 0.5, 0.5, 0.6, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.11], [0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.09, 0.03], [0.2, 0.3, 0.4, 0.6, 0.4, 0.2, 0.2, 0.2, 0.1, 0.1, 0.06, 0.03], [0.1, 0.3, 0.3, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2, 0.15, 0.08, 0.05], [0.1, 0.2, 0.2, 0.5, 0.4, 0.31, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.4, 0.3, 0.3, 0.2, 0.1], [0.2, 0.2, 0.3, 0.3, 0.4, 0.5, 0.5, 0.3, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.6, 0.3, 0.3, 0.2, 0.2], [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.3, 0.2, 0.2, 0.2], [0.1, 0.18, 0.28, 0.39, 0.45, 0.55, 0.44, 0.34, 0.21, 0.2, 0.2, 0.1], [0.1, 0.15, 0.16, 0.27, 0.36, 0.45, 0.42, 0.41, 0.3, 0.2, 0.1, 0.1], [0.2, 0.2, 0.22, 0.43, 0.38, 0.26, 0.23, 0.23, 0.2, 0.2, 0.1, 0.1]], dtype=float)


def _first_available_mat_value(data: dict, *names: str):
    for name in names:
        if name in data:
            return data[name]
    return None


def load_max_flow_and_steady_state_time_parameters(data_dir: str | Path | None = None, nb_region: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    param_path = find_data_file_path(f"qmax_het_{nb_region}.mat", data_dir, __file__)
    data = load_mat_file(param_path)
    qmax_het = _first_available_mat_value(data, f"qmax_het_{nb_region}", "qmax_het")
    if qmax_het is None:
        raise KeyError(f"{param_path} must contain qmax_het_{nb_region} or qmax_het")

    qmax_het = np.asarray(qmax_het, dtype=float)
    tu_ss = _first_available_mat_value(data, f"tu_ss_{nb_region}", "tu_ss")
    td_ss = _first_available_mat_value(data, f"td_ss_{nb_region}", "td_ss")
    tu_ss = 0.25 * np.ones_like(qmax_het) if tu_ss is None else np.asarray(tu_ss, dtype=float)
    td_ss = 0.75 * np.ones_like(qmax_het) if td_ss is None else np.asarray(td_ss, dtype=float)
    if qmax_het.shape != tu_ss.shape or qmax_het.shape != td_ss.shape:
        raise ValueError(f"qmax_het, tu_ss, and td_ss must have the same shape, got {qmax_het.shape}, {tu_ss.shape}, {td_ss.shape}")
    return qmax_het, tu_ss, td_ss


def create_departure_times(Q_S, nb_subregion: int = 7, t_plant: float = 0.0, tfinq: float = 3600.0, rng_seed: int = 42):
    Q_S = np.asarray(Q_S, dtype=float)
    rng = np.random.RandomState(rng_seed)
    departure_t_Rij = [[np.array([], dtype=float) for _ in range(nb_subregion)] for _ in range(nb_subregion)]
    d_t = 300.0
    t = np.arange(t_plant, tfinq + 1e-9, d_t)
    number_vehs = np.ceil(np.diff(t)[None, :] * Q_S[:, : len(t) - 1]).astype(int)
    for h in range(Q_S.shape[0]):
        od_times = []
        for j in range(len(t) - 1):
            n_vehs = number_vehs[h, j]
            if n_vehs > 0:
                od_times.add_vehicles(t[j] + d_t * rng.random(n_vehs))
        o = h // nb_subregion
        d = h % nb_subregion
        departure_t_Rij[o][d] = np.sort(np.asarray(od_times, dtype=float))
    return departure_t_Rij


def create_od_demand_matrix(config: SimulationConfig | None = None, rng_seed: int = 42, save_path: str | Path | None = None, plot: bool = False, data_dir: str | Path | None = None):
    cfg = config or build_simulation_config()
    qmax_het, tu_ss, td_ss = load_max_flow_and_steady_state_time_parameters(data_dir, cfg.nb_region)
    if qmax_het.shape != (cfg.nb_region, cfg.nb_region):
        raise ValueError(f"qmax_het must be {cfg.nb_region}x{cfg.nb_region}, got {qmax_het.shape}")

    time_steps = 12
    d_t = 300.0
    tfinq_local = d_t * time_steps
    t_mid = (np.arange(1, time_steps + 1, dtype=float) - 0.5) * d_t
    Q_S_subregions = np.zeros((cfg.nb_region * cfg.nb_region, time_steps), dtype=float)
    eps = np.finfo(float).eps

    for ori in range(cfg.nb_region):
        for des in range(cfg.nb_region):
            row_idx = ori * cfg.nb_region + des
            q_peak = max(0.0, float(qmax_het[ori, des]))
            t_up = float(tu_ss[ori, des])
            t_down = float(td_ss[ori, des])
            if t_up <= 1.5:
                t_up *= tfinq_local
            if t_down <= 1.5:
                t_down *= tfinq_local
            t_up = max(0.0, min(t_up, tfinq_local))
            t_down = max(t_up + eps, min(t_down, tfinq_local))

            for step, tau in enumerate(t_mid):
                if q_peak <= 0:
                    profile = 0.0
                elif tau < t_up:
                    profile = tau / max(t_up, eps)
                elif tau <= t_down:
                    profile = 1.0
                else:
                    profile = (tfinq_local - tau) / max(tfinq_local - t_down, eps)
                Q_S_subregions[row_idx, step] = q_peak * max(0.0, min(1.0, profile))

    if save_path is not None:
        save_mat_file(save_path, Q_S=Q_S_subregions, qmax_het=qmax_het, qmax_het_7=qmax_het, tu_ss=tu_ss, tu_ss_7=tu_ss, td_ss=td_ss, td_ss_7=td_ss)
    if plot:
        import matplotlib.pyplot as plt
        t = np.arange(0, 3600 + 300, 300)
        q_plot = np.c_[Q_S_subregions, Q_S_subregions[:, -1]]
        target_region = cfg.nb_region
        for r, label in [(1, "region 1 to region 7"), (5, "region 5 to region 7")]:
            row = (r - 1) * cfg.nb_region + (target_region - 1)
            plt.step(t, q_plot[row, :], where="post", linewidth=2, label=label)
        plt.xlabel("time [s]")
        plt.ylabel("Q [veh/s]")
        plt.legend()
        plt.grid(True)
        plt.show()
    return Q_S_subregions


