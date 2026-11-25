from __future__ import annotations

import numpy as np
import numba

from config.config import RegionData
from .utils import get_field_or_attribute, make_random_state


PEN_FACTOR_MIN_CAV_PENETRATION = 0.10


def compute_cav_penetration_factor(D: float, E: float, F: float, cav_penetration: float) -> float:
    """pen_factor with cav_penetration floored at PEN_FACTOR_MIN_CAV_PENETRATION."""
    p = max(float(cav_penetration), PEN_FACTOR_MIN_CAV_PENETRATION)
    return D + E * p + F * p**2


def draw_trip_length(flag: bool, mfd_L=2300.0, mfd_L_noise_SD=0.1, rng=None) -> float:
    rng = make_random_state(rng)
    if flag:
        return float(mfd_L * rng.random())
    return float(max(0.0, mfd_L + (mfd_L_noise_SD * mfd_L) * rng.standard_normal()))


@numba.njit(cache=True)
def _mfd_speed_kernel(values, A, B, C, pen_factor, speed_unit_scale, num_sub_per_reg):
    """g_output = G_i (veh/s); travel_time_HDV = t_i = n/G_i (Eq. 7); travel_time_CAV_SO ="""
    if values.size == 1:
        N_total = float(values[0])
        n_sub = N_total / num_sub_per_reg
        g_sub = (A * n_sub**3 + B * n_sub**2 + C * n_sub) * pen_factor * speed_unit_scale
        g_sub = max(g_sub, 0.0)
        g_output = num_sub_per_reg * g_sub
        dg_dN = (3 * A * n_sub**2 + 2 * B * n_sub + C) * pen_factor * speed_unit_scale
    else:
        N_total = float(np.sum(values))
        g_output = 0.0
        dg_dN_weighted = 0.0
        for n_sub in values:
            g_sub = (A * n_sub**3 + B * n_sub**2 + C * n_sub) * pen_factor * speed_unit_scale
            g_sub = max(float(g_sub), 0.0)
            g_output += g_sub
            dg_sub = (3 * A * n_sub**2 + 2 * B * n_sub + C) * pen_factor * speed_unit_scale
            if N_total > 0:
                dg_dN_weighted += dg_sub * (n_sub / N_total)
        dg_dN = dg_dN_weighted

    g_output = max(float(g_output), 1e-6)
    travel_time_HDV = 0.0 if N_total <= 0 else N_total / g_output
    if N_total <= 0:
        travel_time_CAV_SO = 0.0
    else:
        travel_time_CAV_SO = N_total * (2.0 / g_output - (N_total * dg_dN) / (g_output**2))
    return float(g_output), float(travel_time_HDV), float(max(travel_time_CAV_SO, 0.0))


def compute_region_production_and_travel_times(n_input, region_data: RegionData, subreg_struct=None, cav_penetration: float = 0.6,
                num_sub_per_reg: int = 1, *, speed_unit_scale: float):
    """Macro-region MFD production and HDV/CAV-SO travel times."""
    if isinstance(subreg_struct, (list, tuple)) and subreg_struct:
        s_data = subreg_struct[0]
    elif subreg_struct is not None and get_field_or_attribute(subreg_struct, "aMFD", None) is not None:
        s_data = subreg_struct
    else:
        s_data = RegionData()

    A = get_field_or_attribute(s_data, "aMFD", 4.787e-10)
    B = get_field_or_attribute(s_data, "bMFD", -1.528e-5)
    C = get_field_or_attribute(s_data, "cMFD", 1.257e-1)
    D = get_field_or_attribute(s_data, "dMFD", 3.636e-10)
    E = get_field_or_attribute(s_data, "eMFD", -4.802e-6)
    F = get_field_or_attribute(s_data, "fMFD", 8.346e-2)
    pen_factor = compute_cav_penetration_factor(D, E, F, cav_penetration)

    arr = np.asarray(n_input, dtype=np.float64).reshape(-1)
    return _mfd_speed_kernel(arr, A, B, C, pen_factor, float(speed_unit_scale), float(num_sub_per_reg))
