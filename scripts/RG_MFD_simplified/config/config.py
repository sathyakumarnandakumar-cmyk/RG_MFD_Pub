from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import mfd_profiles


@dataclass
class RegionData:
    ave_trip_length: float = 2300.0
    aMFD: float = 4.787e-10
    bMFD: float = -1.528e-5
    cMFD: float = 1.257e-1
    dMFD: float = 3.636e-10
    eMFD: float = -4.802e-6
    fMFD: float = 8.346e-2
    njamMFD: float = 15660.0
    ncriticalMFD: float = 0.0
    travel_length_2border: float = 1300.0


@dataclass
class SimulationConfig:
    Tstep: float = 60.0
    period_guidance: int = 1
    Tc: float = 60.0
    Hp: int = 6
    Hc: int = 2
    tHp: float = 360.0
    t_init: float = 0.0
    tfinq: float = 3600.0
    tfin: float = 9600.0
    nb_region: int = 7
    nb_subregion: int = 7
    reg_dgraph: np.ndarray = field(default_factory=lambda: np.zeros((7, 7)))
    sub_dgraph: np.ndarray = field(default_factory=lambda: np.zeros((7, 7)))
    sub2reg: np.ndarray = field(default_factory=lambda: np.arange(1, 8, dtype=int))
    subregion_data: list[RegionData] = field(default_factory=list)
    region_data: list[RegionData] = field(default_factory=list)
    min_flow: float = 0.015
    K_shortestpaths: int = 1
    parameter_logit: float = 0.1
    guidance_lb: float = 0.0
    guidance_ub: float = 1.0
    CAV_penetration: float = 0.6




    initial_accumulation_scale: float = 0.6


    speed_unit_scale: float = 1.0


    mfd_profile: str = "default"


    topology_source: str = "builtin 7-ring+hub"


    ban_revisit: bool = False


    ban_revisit_count_current_region: bool = True


    hdv_route_full_count: bool = True


    hdv_pretrip_route: bool = False


    agent_routes_all: bool = False





    hdv_all_or_nothing: bool = False
    control_number: int = 0
    route_guidance_indices: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


def _build_adjacency_matrix(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    graph = np.zeros((n, n), dtype=float)
    for u, v in edges:
        graph[u - 1, v - 1] = 1.0
        graph[v - 1, u - 1] = 1.0
    return graph


def build_simulation_config(data_dir: str | Path | None = None) -> SimulationConfig:
    """Build the default configuration."""


    if data_dir is None:
        _env_dir = os.environ.get("CAV_RG_DATA_DIR", "").strip()
        if _env_dir:
            data_dir = _env_dir

    cfg = SimulationConfig()
    cfg.Tc = cfg.Tstep * cfg.period_guidance
    cfg.tHp = cfg.Tc * cfg.Hp
    cfg.tfin = cfg.tfinq + 6000


    _topo = mfd_profiles.load_region_topology(data_dir)
    if _topo is not None:
        cfg.nb_region = cfg.nb_subregion = int(_topo["nb_region"])
        reg_edges = [tuple(e) for e in _topo["edges"]]
    else:
        outer_ring_edges = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 1),
        ]
        center_edges = [(region, 7) for region in range(1, 7)]
        reg_edges = outer_ring_edges + center_edges
    cfg.topology_source = _topo["_path"] if _topo is not None else "builtin 7-ring+hub"
    cfg.reg_dgraph = _build_adjacency_matrix(reg_edges, cfg.nb_region)


    cfg.sub2reg = np.arange(1, cfg.nb_region + 1, dtype=int)
    cfg.sub_dgraph = cfg.reg_dgraph.copy()


    _profile = ""
    _apply_profile = None
    _blocks = None
    if True:
        _net = mfd_profiles.load_mfd_profile(data_dir)
        if _net is not None:


            _blocks = mfd_profiles.get_region_parameter_blocks(_net, cfg.nb_subregion)
            cfg.speed_unit_scale = float(_net.get("speed_unit_scale", 1.0))
            cfg.mfd_profile = mfd_profiles.describe_mfd_profile(_net)

    cfg.subregion_data = []
    for _i in range(cfg.nb_subregion):
        sd = RegionData()
        if _apply_profile is not None:
            _apply_profile(sd)
        elif _blocks is not None:
            mfd_profiles.apply_region_parameter_block(sd, _blocks[_i])

        _declared_ncrit = _blocks[_i].get("ncriticalMFD") if _blocks is not None else None
        roots = np.roots([3 * sd.aMFD, 2 * sd.bMFD, sd.cMFD])
        positive = roots[np.isreal(roots)].real
        positive = positive[positive > 0]


        if _blocks is not None and not positive.size:
            raise mfd_profiles.MFDProfileError(
                f"{cfg.mfd_profile}: region {_i + 1} has no positive root for "
                f"roots(3a, 2b, c) with aMFD={sd.aMFD:g}, bMFD={sd.bMFD:g}, "
                f"cMFD={sd.cMFD:g} -- the completion curve never peaks, so there is no "
                f"critical accumulation. Check the coefficients came from one source; "
                f"verify with: python scripts/mfd_params.py --coeffs "
                f"{sd.aMFD:g} {sd.bMFD:g} {sd.cMFD:g}"
            )
        _derived_ncrit = float(np.min(positive)) if positive.size else 0.0
        if _declared_ncrit is None:
            sd.ncriticalMFD = _derived_ncrit
        else:


            _gap = abs(_declared_ncrit - _derived_ncrit) / max(_derived_ncrit, 1e-12)
            if _gap > mfd_profiles.NCRIT_TOLERANCE:
                raise mfd_profiles.MFDProfileError(
                    f"{cfg.mfd_profile}: region {_i + 1} declares "
                    f"ncriticalMFD={_declared_ncrit:g} but roots(3a, 2b, c) gives "
                    f"{_derived_ncrit:.1f} ({_gap:.1%} apart, tolerance "
                    f"{mfd_profiles.NCRIT_TOLERANCE:.0%}). Either the value or the "
                    f"coefficients are from a different source. Check with: "
                    f"python scripts/mfd_params.py --coeffs "
                    f"{sd.aMFD:g} {sd.bMFD:g} {sd.cMFD:g}"
                )
            sd.ncriticalMFD = float(_declared_ncrit)
        cfg.subregion_data.append(sd)

    cfg.region_data = []
    for reg in range(1, cfg.nb_region + 1):
        subregs = [cfg.subregion_data[i] for i in np.where(cfg.sub2reg == reg)[0]]
        cfg.region_data.append(
            RegionData(
                ave_trip_length=sum(s.ave_trip_length for s in subregs),
                njamMFD=sum(s.njamMFD for s in subregs),
                ncriticalMFD=sum(s.ncriticalMFD for s in subregs),
            )
        )

    cfg.control_number = cfg.nb_region * (cfg.nb_region - 1) * cfg.K_shortestpaths
    cfg.route_guidance_indices = {}
    curr = 0
    for i in range(1, cfg.nb_region + 1):
        for j in range(1, cfg.nb_region + 1):
            if i != j:
                cfg.route_guidance_indices[(i, j)] = np.arange(curr, curr + cfg.K_shortestpaths, dtype=int)
                curr += cfg.K_shortestpaths
    return cfg
