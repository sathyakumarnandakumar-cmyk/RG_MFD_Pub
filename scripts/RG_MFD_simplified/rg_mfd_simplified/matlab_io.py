from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import AttrDict, candidate_data_directories


def _convert_matlab_object_to_python(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_convert_matlab_object_to_python(x) for x in obj.reshape(-1)] if obj.ndim > 1 else [_convert_matlab_object_to_python(x) for x in obj]
        return obj
    if hasattr(obj, "_fieldnames"):
        out = AttrDict()
        for name in obj._fieldnames:
            out[name] = _convert_matlab_object_to_python(getattr(obj, name))
        return out
    return obj


def load_mat_file(path: str | Path, variables: Iterable[str] | None = None) -> dict:
    """Load MATLAB .mat files, with h5py fallback for v7.3 files."""
    path = Path(path)
    try:
        from scipy.io import loadmat as scipy_loadmat
    except ModuleNotFoundError:
        scipy_loadmat = None

    variable_names = None if variables is None else list(variables)
    if scipy_loadmat is not None:
        try:
            data = scipy_loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=variable_names)
            out = {k: _convert_matlab_object_to_python(v) for k, v in data.items() if not k.startswith("__")}
        except NotImplementedError:
            out = _load_hdf5_mat_file(path, variable_names)
    else:
        out = _load_hdf5_mat_file(path, variable_names)
    if variables is not None:
        keep = set(variables)
        out = {k: v for k, v in out.items() if k in keep}
    return out


def _load_hdf5_mat_file(path: Path, variables: Iterable[str] | None = None) -> dict:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install scipy for MATLAB v5 .mat files, or h5py for MATLAB v7.3 files") from exc
    out = {}
    keep = set(variables) if variables is not None else None
    with h5py.File(path, "r") as handle:
        for k in handle.keys():
            if keep is not None and k not in keep:
                continue
            value = np.array(handle[k])
            out[k] = value.T if value.ndim > 1 else value
    return out


def save_mat_file(path: str | Path, **kwargs) -> None:
    try:
        from scipy.io import savemat
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install scipy to write MATLAB .mat files") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    savemat(path, kwargs)


def find_data_file_path(name: str, data_dir: str | Path | None = None, package_file: str | Path | None = None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser())
    candidates.extend(candidate_data_directories(package_file))


    warn_fallback = data_dir is not None and not (Path(data_dir).expanduser() / name).exists()

    def _warn_and_return_fallback_path(path: Path) -> Path:
        if warn_fallback:
            print(f"WARNING: {name} not found directly in {data_dir}; using fallback {path}", file=sys.stderr)
        return path

    for directory in candidates:
        if not directory.exists():
            continue


        path = directory / name
        if path.exists():
            return _warn_and_return_fallback_path(path)


        if directory.is_dir():
            matches = list(directory.rglob(name))
            if matches:
                return _warn_and_return_fallback_path(matches[0])

    raise FileNotFoundError(f"Could not find {name} recursively in any of: {candidates}")
