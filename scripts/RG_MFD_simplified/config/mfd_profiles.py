"""Per-network MFD calibrations, declared by an `mfd.json` inside each network directory."""
from __future__ import annotations

import json
from pathlib import Path


ALLOWED_REGION_KEYS = {
    "ave_trip_length",
    "aMFD", "bMFD", "cMFD",
    "dMFD", "eMFD", "fMFD",
    "njamMFD",
    "ncriticalMFD",
    "travel_length_2border",
}


NCRIT_TOLERANCE = 0.02

PROFILE_FILENAME = "mfd.json"
TOPOLOGY_FILENAME = "topology.json"


class MFDProfileError(ValueError):
    """Raised when an mfd.json is malformed or does not fit the topology being built."""


def mfd_profile_file_path(data_dir: str | Path | None) -> Path | None:
    """Path to the mfd.json for this data dir, or None if there is no such file."""
    if data_dir is None:
        return None
    p = Path(data_dir).expanduser() / PROFILE_FILENAME
    return p if p.is_file() else None


def load_mfd_profile(data_dir: str | Path | None) -> dict | None:
    """Read and validate the mfd.json in `data_dir`. Returns None when absent."""
    path = mfd_profile_file_path(data_dir)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            profile = json.load(fh)
    except json.JSONDecodeError as exc:
        raise MFDProfileError(f"{path}: invalid JSON -- {exc}") from exc
    if not isinstance(profile, dict):
        raise MFDProfileError(f"{path}: top level must be a JSON object")
    profile["_path"] = str(path)
    return profile


def topology_file_path(data_dir: str | Path | None) -> Path | None:
    """Path to the topology.json for this data dir, or None if there is no such file."""
    if data_dir is None:
        return None
    p = Path(data_dir).expanduser() / TOPOLOGY_FILENAME
    return p if p.is_file() else None


def _is_integer(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_region_topology(data_dir: str | Path | None) -> dict | None:
    """Read and validate the topology.json in `data_dir`. Returns None when absent."""
    path = topology_file_path(data_dir)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            topo = json.load(fh)
    except json.JSONDecodeError as exc:
        raise MFDProfileError(f"{path}: invalid JSON -- {exc}") from exc
    if not isinstance(topo, dict):
        raise MFDProfileError(f"{path}: top level must be a JSON object")

    nb = topo.get("nb_region")
    if not _is_integer(nb) or nb < 2:
        raise MFDProfileError(f"{path}: 'nb_region' must be an integer >= 2, got {nb!r}")

    edges = topo.get("edges")
    if not isinstance(edges, list) or not edges:
        raise MFDProfileError(f"{path}: 'edges' must be a non-empty list of [a, b] pairs")
    seen: set[frozenset] = set()
    degree = [0] * (nb + 1)
    for i, e in enumerate(edges):
        if not (isinstance(e, list) and len(e) == 2 and all(_is_integer(x) for x in e)):
            raise MFDProfileError(f"{path}: edges[{i}] must be a 2-element list of ints, got {e!r}")
        a, b = e
        if not (1 <= a <= nb and 1 <= b <= nb):
            raise MFDProfileError(
                f"{path}: edges[{i}]={e} has a region id outside 1..{nb}"
            )
        if a == b:
            raise MFDProfileError(f"{path}: edges[{i}]={e} is a self-loop")
        key = frozenset((a, b))
        if key in seen:
            raise MFDProfileError(
                f"{path}: edges[{i}]={e} duplicates an earlier edge (edges are undirected)"
            )
        seen.add(key)
        degree[a] += 1
        degree[b] += 1
    isolated = [r for r in range(1, nb + 1) if degree[r] == 0]
    if isolated:
        raise MFDProfileError(
            f"{path}: region(s) {isolated} have no edges; every region must be connected "
            f"(an isolated region makes shortest-path routing fail)"
        )
    topo["_path"] = str(path)
    return topo


def _validate_region_parameter_block(block: dict, path: str, where: str) -> dict:
    if not isinstance(block, dict):
        raise MFDProfileError(f"{path}: {where} must be a JSON object")
    unknown = set(block) - ALLOWED_REGION_KEYS
    if unknown:
        raise MFDProfileError(
            f"{path}: {where} has unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_REGION_KEYS)}."
        )
    for k, v in block.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise MFDProfileError(f"{path}: {where}.{k} must be a number, got {v!r}")
    return block


def get_region_parameter_blocks(profile: dict, nb_region: int) -> list[dict]:
    """Per-region override dicts, one per region, validated against the topology."""
    path = profile.get("_path", "<mfd.json>")

    declared = profile.get("nb_region")
    if declared is not None and int(declared) != int(nb_region):
        raise MFDProfileError(
            f"{path}: declares nb_region={declared} but the configuration is building "
            f"{nb_region} regions. This network's parameters do not fit this topology -- "
            f"check {TOPOLOGY_FILENAME} in the same directory (absent => builtin "
            f"7-region ring+hub)."
        )

    regions = profile.get("regions")
    if regions is None:
        raise MFDProfileError(f"{path}: missing required key 'regions'")

    if isinstance(regions, dict):
        _validate_region_parameter_block(regions, path, "regions")
        return [dict(regions) for _ in range(nb_region)]

    if isinstance(regions, list):
        if len(regions) != nb_region:
            raise MFDProfileError(
                f"{path}: 'regions' is a list of {len(regions)} entries but "
                f"{nb_region} regions are being built. Provide exactly one entry per "
                f"region, or a single object to apply to all."
            )
        return [dict(_validate_region_parameter_block(b, path, f"regions[{i}]"))
                for i, b in enumerate(regions)]

    raise MFDProfileError(f"{path}: 'regions' must be an object or a list of objects")


def apply_region_parameter_block(region_data, block: dict) -> None:
    """Overwrite one RegionData instance in place with the values from `block`."""
    for key, value in block.items():
        setattr(region_data, key, float(value))


def describe_mfd_profile(profile: dict | None) -> str:
    """Short label for logging / result metadata."""
    if profile is None:
        return "default"
    return str(profile.get("name") or Path(profile.get("_path", "mfd.json")).parent.name)
