from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


class AttrDict(dict):
    """Dictionary with attribute access, convenient for translated MATLAB structs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def get_field_or_attribute(obj: Any, name: str, default: Any = None) -> Any:


    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            return obj[name]
        except (KeyError, TypeError):
            return default


def make_random_state(rng=None, seed: int | None = None):
    if rng is not None:
        return rng
    return np.random.RandomState(seed)


def convert_to_plain_python(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, AttrDict):
        return dict(obj)
    if isinstance(obj, list):
        return [convert_to_plain_python(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_plain_python(x) for x in obj)
    return obj


def candidate_data_directories(package_file: str | Path | None = None) -> list[Path]:
    import os

    dirs: list[Path] = []
    env_dir = os.environ.get("CAV_RG_DATA_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    dirs.append(Path.cwd())
    if package_file is not None:
        p = Path(package_file).resolve()
        dirs.append(p.parent)
        dirs.append(p.parents[1])
        if len(p.parents) > 2:
            dirs.append(p.parents[2] / "Ye_CAV_RG_19_subreg")
            dirs.append(p.parents[2] / "Ye_CAV_RG_7_subreg_matlab")
    unique: list[Path] = []
    seen = set()
    for d in dirs:
        rd = d.resolve()
        if rd not in seen:
            unique.append(rd)
            seen.add(rd)
    return unique
