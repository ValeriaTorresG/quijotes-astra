#!/usr/bin/env python3
"""Compute Quijote halo power spectra for all halos and ASTRA environments.

The numerical recipe follows ``power_spec.ipynb``:

* paint halo positions on a regular grid with Pylians' CIC assignment;
* convert the mesh to the density contrast;
* evaluate P0, P2, and P4 with ``Pk_library.Pk``;
* subtract the Poisson shot noise, V/N, from the monopole only;
* retain the native Pylians shells in the requested k interval.

Positions are read directly from the Quijote FoF catalogue.  This is equivalent
to selecting ``RANDITER == -1`` from ASTRA's raw FITS product, but avoids
loading all random-catalogue copies.  ASTRA's ``TARGETID`` is validated and
used to map each probability row back to its FoF halo.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "Quijotes" / "data"
DEFAULT_DATASETS = ("fiducial",)
DEFAULT_SNAPSHOT = 3
DEFAULT_BOX_SIZE = 1000.0
DEFAULT_GRID = 512
DEFAULT_MAS = "CIC"
DEFAULT_AXIS = 0
DEFAULT_THREADS = 1
DEFAULT_KMIN = 0.008
DEFAULT_KMAX = 0.5

SNAPSHOT_REDSHIFTS = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.5, 4: 0.0}
ENVIRONMENT_COLUMNS = (
    ("void", "PVOID"),
    ("sheet", "PSHEET"),
    ("filament", "PFILAMENT"),
    ("knot", "PKNOT"),
)

CSV_COLUMNS = (
    "dataset",
    "simulation_id",
    "snapshot",
    "redshift",
    "sample",
    "tracer",
    "n_objects",
    "n_k_shells",
    "number_density_h3_Mpc3",
    "fof_catalog_sha256",
    "n_astra_iterations",
    "astra_probability_sha256",
    "astra_release",
    "astra_random_seed",
    "astra_r_lower",
    "astra_r_med",
    "astra_r_upper",
    "astra_periodic",
    "astra_box_min_Mpc_h",
    "astra_box_max_Mpc_h",
    "box_size_Mpc_h",
    "grid",
    "mass_assignment",
    "los_axis",
    "threads",
    "k_min_h_Mpc",
    "k_max_h_Mpc",
    "k_nyquist_h_Mpc",
    "k_h_Mpc",
    "Pk0_raw_Mpc3_h3",
    "shot_noise_Mpc3_h3",
    "Pk0_shot_subtracted_Mpc3_h3",
    "Pk2_Mpc3_h3",
    "Pk4_Mpc3_h3",
    "sigma_Pk0_notebook_Mpc3_h3",
    "Nmodes",
)


class PowerSpectrumError(RuntimeError):
    """Raised when an input or a numerical result is not safe to use."""


@dataclass(frozen=True)
class Spectrum:
    """One masked Pylians power spectrum and its scalar sample metadata."""

    k: np.ndarray
    pk0_raw: np.ndarray
    pk0_shot_subtracted: np.ndarray
    pk2: np.ndarray
    pk4: np.ndarray
    sigma_pk0: np.ndarray
    nmodes: np.ndarray
    n_objects: int
    number_density: float
    shot_noise: float


@dataclass(frozen=True)
class EnvironmentAssignment:
    """FoF indices for the four hard ASTRA environment assignments."""

    indices: Mapping[str, np.ndarray]
    provenance: "AstraProvenance"


@dataclass(frozen=True)
class AstraProvenance:
    """Validated ASTRA configuration and identity of one probability product."""

    probability_path: Path
    probability_sha256: str
    release: str
    n_iterations: int
    random_seed: int
    r_lower: float
    r_med: float
    r_upper: float
    redshift: float
    periodic: bool
    box_min: float
    box_max: float


@dataclass
class Counters:
    """Mutable counters used for the final batch summary."""

    written: int = 0
    skipped: int = 0
    missing_probabilities: int = 0
    failures: int = 0


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _format_float(value: float) -> str:
    return f"{float(value):.12e}"


def _log(message: str) -> None:
    print(message, flush=True)


def _error(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _dataset_name(value: str) -> str:
    """Reject absolute or nested dataset values used to construct paths."""

    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise argparse.ArgumentTypeError(
            f"invalid dataset name {value!r}; use a directory name such as fiducial"
        )
    return name


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pylians P(k) for all Quijote FoF halos and for the four "
            "hard ASTRA environments, writing one CSV per sample and simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        type=_dataset_name,
        default=list(DEFAULT_DATASETS),
        metavar="NAME",
        help=(
            "catalogue families below --data-root; use 'all' to discover every "
            "family that has numeric realization directories"
        ),
    )
    parser.add_argument(
        "--simulation-ids",
        "--simulation-id",
        "--simulations",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "realization IDs applied to every dataset; omit this option, or pass "
            "'all', to discover every available numeric directory"
        ),
    )
    parser.add_argument(
        "--snapnum",
        type=_nonnegative_int,
        default=DEFAULT_SNAPSHOT,
        help="Quijote snapshot number",
    )
    parser.add_argument(
        "--data-root",
        type=_path,
        default=DEFAULT_DATA_ROOT,
        help="root containing fiducial/, Om_m/, Om_p/, s8_m/, ...",
    )
    parser.add_argument(
        "--astra-root",
        type=_path,
        default=None,
        help="root containing ASTRA run directories; defaults to DATA_ROOT/astra",
    )
    parser.add_argument(
        "--output-root",
        type=_path,
        default=None,
        help="output root; defaults to DATA_ROOT/pk",
    )
    parser.add_argument(
        "--grid",
        type=_positive_int,
        default=DEFAULT_GRID,
        help="number of mesh cells per dimension",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=DEFAULT_BOX_SIZE,
        help="periodic box side in Mpc/h",
    )
    parser.add_argument(
        "--mas",
        choices=("NGP", "CIC", "TSC", "PCS"),
        default=DEFAULT_MAS,
        help="mass-assignment scheme passed to Pylians",
    )
    parser.add_argument(
        "--axis",
        type=int,
        choices=(0, 1, 2),
        default=DEFAULT_AXIS,
        help="line-of-sight axis used for P2 and P4 (0=x, 1=y, 2=z)",
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=DEFAULT_THREADS,
        help="OpenMP threads passed to Pk_library.Pk",
    )
    parser.add_argument(
        "--kmin",
        type=float,
        default=DEFAULT_KMIN,
        help="minimum retained k in h/Mpc",
    )
    parser.add_argument(
        "--kmax",
        type=float,
        default=DEFAULT_KMAX,
        help="maximum retained k in h/Mpc",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--matter-only",
        action="store_true",
        help="write only the all-halo spectrum under matter/",
    )
    sample_group.add_argument(
        "--environments-only",
        action="store_true",
        help="write only the four ASTRA environment spectra under env/",
    )
    parser.add_argument(
        "--allow-missing-probabilities",
        action="store_true",
        help=(
            "skip environment spectra without treating an unfinished ASTRA "
            "probability product as a batch failure"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSV products atomically",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failed simulation instead of continuing the batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show resolved inputs and outputs without reading catalogues or computing",
    )
    parser.add_argument(
        "--quiet-pylians",
        action="store_true",
        help="suppress verbose output from MAS_library and Pk_library",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not math.isfinite(args.box_size) or args.box_size <= 0.0:
        parser.error("--box-size must be finite and positive")
    if not math.isfinite(args.kmin) or not math.isfinite(args.kmax):
        parser.error("--kmin and --kmax must be finite")
    if args.kmin < 0.0 or args.kmax <= args.kmin:
        parser.error("--kmin must be non-negative and smaller than --kmax")
    if args.grid < 2:
        parser.error("--grid must be at least 2")
    k_nyquist = math.pi * args.grid / args.box_size
    if args.kmax > k_nyquist * (1.0 + 1.0e-12):
        parser.error(
            f"--kmax={args.kmax:g} exceeds the mesh Nyquist frequency "
            f"{k_nyquist:.6g} h/Mpc for --grid={args.grid} and "
            f"--box-size={args.box_size:g}"
        )

    values = args.simulation_ids
    if values is not None:
        lowered = [str(value).lower() for value in values]
        if "all" in lowered and len(lowered) != 1:
            parser.error("'all' cannot be combined with explicit simulation IDs")
        if lowered != ["all"]:
            try:
                parsed = [_nonnegative_int(str(value)) for value in values]
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(str(exc))
            args.simulation_ids = list(dict.fromkeys(parsed))
        else:
            args.simulation_ids = None

    lowered_datasets = [value.lower() for value in args.datasets]
    if "all" in lowered_datasets and len(lowered_datasets) != 1:
        parser.error("'all' cannot be combined with explicit dataset names")


def _has_snapshot(realization_dir: Path, snapnum: int) -> bool:
    tag = f"{snapnum:03d}"
    return (
        realization_dir
        / f"groups_{tag}"
        / f"group_tab_{tag}.0"
    ).is_file()


def discover_datasets(data_root: Path, snapnum: int) -> list[str]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    found = []
    for candidate in sorted(data_root.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir() or candidate.name in {"astra", "pk"}:
            continue
        if any(
            child.is_dir()
            and child.name.isdigit()
            and _has_snapshot(child, snapnum)
            for child in candidate.iterdir()
        ):
            found.append(candidate.name)
    if not found:
        raise FileNotFoundError(
            f"no Quijote dataset with snapshot {snapnum:03d} found below {data_root}"
        )
    return found


def discover_simulations(
    data_root: Path,
    dataset: str,
    snapnum: int,
) -> list[int]:
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    found = sorted(
        int(child.name)
        for child in dataset_dir.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and _has_snapshot(child, snapnum)
    )
    if not found:
        raise FileNotFoundError(
            f"no realization with groups_{snapnum:03d}/group_tab_"
            f"{snapnum:03d}.0 found in {dataset_dir}"
        )
    return found


def catalogue_directory(
    data_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> Path:
    directory = data_root / dataset / str(simulation_id)
    if not _has_snapshot(directory, snapnum):
        expected = (
            directory
            / f"groups_{snapnum:03d}"
            / f"group_tab_{snapnum:03d}.0"
        )
        raise FileNotFoundError(f"FoF catalogue part not found: {expected}")
    return directory


def _matching_run_directories(
    astra_root: Path,
    dataset: str,
    simulation_id: int,
) -> list[Path]:
    matcher = re.compile(rf"^{re.escape(dataset)}_(\d+)$")
    candidates: list[Path] = []
    if astra_root.is_dir():
        own_match = matcher.match(astra_root.name)
        if own_match and int(own_match.group(1)) == simulation_id:
            candidates.append(astra_root)
        for child in astra_root.iterdir():
            if not child.is_dir():
                continue
            match = matcher.match(child.name)
            if match and int(match.group(1)) == simulation_id:
                candidates.append(child)
    return sorted(set(candidates))


def resolve_probability_path(
    astra_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> Path:
    """Find the final real-data probability FITS, never iteration chunks."""

    sim_tag = f"sim{simulation_id:03d}"
    snap_tag = f"snap{snapnum:03d}"
    preferred_names = (
        f"zone_00_{sim_tag}_{snap_tag}_probability_iterdata.fits.gz",
        f"zone_00_{sim_tag}_{snap_tag}_probability_iterdata.fits",
    )
    run_directories = _matching_run_directories(
        astra_root, dataset, simulation_id
    )
    if not run_directories:
        expected = astra_root / f"{dataset}_{simulation_id}"
        raise FileNotFoundError(
            f"ASTRA run directory not found for {dataset} simulation "
            f"{simulation_id}; expected a directory such as {expected}"
        )

    matches_by_run: dict[Path, list[Path]] = {}
    for run_directory in run_directories:
        matches: list[Path] = []
        for name in preferred_names:
            matches.extend(run_directory.rglob(name))
        matches = sorted(set(path.resolve() for path in matches if path.is_file()))
        if matches:
            matches_by_run[run_directory] = matches
    if len(matches_by_run) > 1:
        joined = "\n  ".join(
            f"{run}: {', '.join(str(path) for path in matches)}"
            for run, matches in matches_by_run.items()
        )
        raise PowerSpectrumError(
            "more than one ASTRA run contains a final probability product; "
            f"pass a narrower --astra-root:\n  {joined}"
        )
    if len(matches_by_run) == 1:
        matches = next(iter(matches_by_run.values()))
        if len(matches) == 1:
            return matches[0]
        joined = "\n  ".join(str(path) for path in matches)
        raise PowerSpectrumError(
            "multiple final ASTRA probability products found in one run:\n  "
            f"{joined}"
        )

    searched = ", ".join(str(path) for path in run_directories)
    raise FileNotFoundError(
        f"final ASTRA *_probability_iterdata product not found for {dataset} "
        f"simulation {simulation_id}, snapshot {snapnum}; searched: {searched}"
    )


def output_paths(
    output_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> tuple[Path, dict[str, Path]]:
    stem = f"{dataset}_sim{simulation_id:03d}_snap{snapnum:03d}"
    matter = output_root / "matter" / f"{stem}_pk.csv"
    environments = {
        name: output_root / "env" / f"{stem}_{name}_pk.csv"
        for name, _ in ENVIRONMENT_COLUMNS
    }
    return matter, environments


def import_pylians() -> tuple[Any, Any, Any]:
    """Import the three Pylians modules with an actionable environment error."""

    modules = []
    missing = []
    for name in ("readfof", "MAS_library", "Pk_library"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise PowerSpectrumError(
            f"missing Pylians module(s): {joined}. Activate the requested "
            "environment first with `conda activate pylians-x86`."
        )
    return modules[0], modules[1], modules[2]


def _sha256_files(paths: Sequence[Path]) -> str:
    """Hash file names and contents in a deterministic order."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def fof_catalogue_parts(directory: Path, snapnum: int) -> list[Path]:
    """Return contiguous ``group_tab`` parts used by one readfof call."""

    tag = f"{snapnum:03d}"
    group_directory = directory / f"groups_{tag}"
    matcher = re.compile(rf"^group_tab_{tag}\.(\d+)$")
    numbered: list[tuple[int, Path]] = []
    if group_directory.is_dir():
        for path in group_directory.iterdir():
            match = matcher.match(path.name)
            if match and path.is_file():
                numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    numbers = [number for number, _ in numbered]
    if numbers != list(range(len(numbered))):
        raise PowerSpectrumError(
            f"FoF parts in {group_directory} are not contiguous from part 0: "
            f"{numbers}"
        )
    if not numbered:
        raise FileNotFoundError(
            f"no group_tab_{tag} parts found in {group_directory}"
        )
    return [path for _, path in numbered]


def read_fof_positions(
    readfof: Any,
    directory: Path,
    snapnum: int,
    box_size: float,
) -> tuple[np.ndarray, str]:
    """Read ``GroupPos`` in kpc/h and return contiguous positions in Mpc/h."""

    catalogue_parts = fof_catalogue_parts(directory, snapnum)
    catalogue_sha256 = _sha256_files(catalogue_parts)
    fof = readfof.FoF_catalog(
        str(directory),
        snapnum,
        long_ids=False,
        swap=False,
        SFR=False,
        read_IDs=False,
    )
    raw_positions = np.asarray(fof.GroupPos)
    if raw_positions.ndim != 2 or raw_positions.shape[1] != 3:
        raise PowerSpectrumError(
            f"readfof returned GroupPos with shape {raw_positions.shape}, expected (N, 3)"
        )
    positions = np.array(raw_positions, dtype=np.float32, order="C", copy=True)
    positions /= np.float32(1000.0)
    del fof, raw_positions

    if len(positions) == 0:
        raise PowerSpectrumError(f"FoF catalogue contains no halos: {directory}")
    if not np.all(np.isfinite(positions)):
        raise PowerSpectrumError(f"FoF positions contain NaN or infinity: {directory}")
    tolerance = max(1.0e-5, abs(box_size) * 1.0e-6)
    minimum = float(np.min(positions))
    maximum = float(np.max(positions))
    if minimum < -tolerance or maximum > box_size + tolerance:
        raise PowerSpectrumError(
            f"FoF positions [{minimum}, {maximum}] lie outside the requested "
            f"periodic box [0, {box_size}] Mpc/h"
        )
    # A coordinate equal to the upper boundary is the periodic image of zero.
    np.mod(positions, np.float32(box_size), out=positions)
    return (
        np.ascontiguousarray(positions, dtype=np.float32),
        catalogue_sha256,
    )


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any:
    for actual_key, value in metadata.items():
        if str(actual_key).upper() == key.upper():
            return value
    raise PowerSpectrumError(f"ASTRA probability header is missing {key}")


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = _metadata_value(metadata, key)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"ASTRA probability header {key}={value!r} is not an integer"
        ) from exc
    return parsed


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float:
    value = _metadata_value(metadata, key)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"ASTRA probability header {key}={value!r} is not numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise PowerSpectrumError(
            f"ASTRA probability header {key}={value!r} is not finite"
        )
    return parsed


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = _metadata_value(metadata, key)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "t"}:
        return True
    if normalized in {"0", "false", "no", "off", "f"}:
        return False
    raise PowerSpectrumError(
        f"ASTRA probability header {key}={value!r} is not boolean"
    )


def read_environment_assignment(
    probability_path: Path,
    n_halos: int,
    simulation_id: int,
    snapnum: int,
    box_size: float,
) -> EnvironmentAssignment:
    """Validate ASTRA probabilities and map hard classes to FoF row indices."""

    try:
        from astropy.table import Table
    except ImportError as exc:
        raise PowerSpectrumError(
            "astropy is required to read ASTRA probability FITS products"
        ) from exc

    table = Table.read(probability_path, hdu=1)
    required = ("TARGETID", "TRACERTYPE") + tuple(
        column for _, column in ENVIRONMENT_COLUMNS
    )
    missing = [column for column in required if column not in table.colnames]
    if missing:
        raise PowerSpectrumError(
            f"{probability_path} is missing columns: {', '.join(missing)}"
        )

    meta_simulation = _metadata_int(table.meta, "SIMID")
    meta_snapshot = _metadata_int(table.meta, "SNAPNUM")
    if meta_simulation != simulation_id:
        raise PowerSpectrumError(
            f"SIMID={meta_simulation} in {probability_path}, expected {simulation_id}"
        )
    if meta_snapshot != snapnum:
        raise PowerSpectrumError(
            f"SNAPNUM={meta_snapshot} in {probability_path}, expected {snapnum}"
        )
    release = str(_metadata_value(table.meta, "RELEASE")).strip()
    if release.upper() != "QUIJOTES":
        raise PowerSpectrumError(
            f"RELEASE={release!r} in {probability_path}, expected QUIJOTES"
        )
    n_iterations = _metadata_int(table.meta, "NITER")
    random_seed = _metadata_int(table.meta, "RNGSEED")
    if n_iterations <= 0:
        raise PowerSpectrumError(f"NITER must be positive, found {n_iterations}")
    if random_seed < 0:
        raise PowerSpectrumError(f"RNGSEED must be non-negative, found {random_seed}")
    r_lower = _metadata_float(table.meta, "RLOWER")
    r_med = _metadata_float(table.meta, "RMED")
    r_upper = _metadata_float(table.meta, "RUPPER")
    if not r_lower < r_med < r_upper:
        raise PowerSpectrumError(
            "ASTRA thresholds must satisfy RLOWER < RMED < RUPPER; found "
            f"{r_lower}, {r_med}, {r_upper}"
        )
    redshift = _metadata_float(table.meta, "REDSHFT")
    periodic = _metadata_bool(table.meta, "PERIODIC")
    if not periodic:
        raise PowerSpectrumError(
            "ASTRA probabilities are not marked PERIODIC for a Quijote box"
        )
    lower_bounds = np.array(
        [
            _metadata_float(table.meta, "BOXXMIN"),
            _metadata_float(table.meta, "BOXYMIN"),
            _metadata_float(table.meta, "BOXZMIN"),
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            _metadata_float(table.meta, "BOXXMAX"),
            _metadata_float(table.meta, "BOXYMAX"),
            _metadata_float(table.meta, "BOXZMAX"),
        ],
        dtype=np.float64,
    )
    if not np.allclose(lower_bounds, 0.0, rtol=0.0, atol=1.0e-8):
        raise PowerSpectrumError(
            f"ASTRA box lower bounds are {lower_bounds.tolist()}, expected [0, 0, 0]"
        )
    if not np.allclose(
        upper_bounds, box_size, rtol=0.0, atol=max(1.0e-8, box_size * 1.0e-10)
    ):
        raise PowerSpectrumError(
            f"ASTRA box upper bounds are {upper_bounds.tolist()}, expected "
            f"[{box_size}, {box_size}, {box_size}] Mpc/h"
        )

    tracer_values = np.asarray(table["TRACERTYPE"]).astype(str)
    if tracer_values.ndim != 1 or not np.all(
        np.char.upper(np.char.strip(tracer_values)) == "HALO"
    ):
        unique = np.unique(tracer_values).tolist()
        raise PowerSpectrumError(
            f"ASTRA probabilities contain non-HALO TRACERTYPE values: {unique}"
        )

    masked_ids = np.ma.asarray(table["TARGETID"])
    if np.any(np.ma.getmaskarray(masked_ids)):
        raise PowerSpectrumError("TARGETID contains masked values")
    raw_ids = np.asarray(masked_ids.data)
    try:
        ids_as_float = np.asarray(raw_ids, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PowerSpectrumError("TARGETID cannot be converted to integers") from exc
    if ids_as_float.ndim != 1 or not np.all(np.isfinite(ids_as_float)):
        raise PowerSpectrumError("TARGETID contains masked, NaN, or infinite values")
    if not np.all(ids_as_float == np.floor(ids_as_float)):
        raise PowerSpectrumError("TARGETID contains non-integer values")
    target_ids = ids_as_float.astype(np.int64)

    if len(target_ids) != n_halos:
        raise PowerSpectrumError(
            f"ASTRA probabilities contain {len(target_ids)} rows, but readfof "
            f"contains {n_halos} halos; refusing a partial environment split"
        )
    if len(np.unique(target_ids)) != len(target_ids):
        raise PowerSpectrumError("ASTRA probability TARGETID values are not unique")
    expected_ids = np.arange(1, n_halos + 1, dtype=np.int64)
    if not np.array_equal(np.sort(target_ids), expected_ids):
        raise PowerSpectrumError(
            f"TARGETID must be the complete range 1..{n_halos} to map onto readfof"
        )

    probability_columns = []
    for _, column in ENVIRONMENT_COLUMNS:
        masked_values = np.ma.asarray(table[column])
        if np.any(np.ma.getmaskarray(masked_values)):
            raise PowerSpectrumError(f"{column} contains masked values")
        probability_columns.append(
            np.asarray(masked_values.data, dtype=np.float64)
        )
    probabilities = np.column_stack(probability_columns)
    if not np.all(np.isfinite(probabilities)):
        raise PowerSpectrumError("ASTRA probabilities contain NaN or infinity")
    tolerance = 1.0e-6
    if np.any(probabilities < -tolerance) or np.any(
        probabilities > 1.0 + tolerance
    ):
        raise PowerSpectrumError("ASTRA probabilities lie outside [0, 1]")
    sums = np.sum(probabilities, axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1.0e-5):
        maximum_error = float(np.max(np.abs(sums - 1.0)))
        raise PowerSpectrumError(
            "ASTRA probabilities do not sum to one; maximum absolute error "
            f"is {maximum_error:.3e}"
        )

    # np.argmax deliberately reproduces the notebook's tie priority:
    # void -> sheet -> filament -> knot.
    hard_class = np.argmax(probabilities, axis=1)
    indices = {
        name: np.ascontiguousarray(
            target_ids[hard_class == class_index] - 1,
            dtype=np.int64,
        )
        for class_index, (name, _) in enumerate(ENVIRONMENT_COLUMNS)
    }
    if sum(len(value) for value in indices.values()) != n_halos:
        raise PowerSpectrumError("environment assignments do not cover every halo")

    probability_sha256 = _sha256_files([probability_path])
    del table, probabilities, probability_columns, hard_class
    return EnvironmentAssignment(
        indices=indices,
        provenance=AstraProvenance(
            probability_path=probability_path,
            probability_sha256=probability_sha256,
            release=release,
            n_iterations=n_iterations,
            random_seed=random_seed,
            r_lower=r_lower,
            r_med=r_med,
            r_upper=r_upper,
            redshift=redshift,
            periodic=periodic,
            box_min=float(lower_bounds[0]),
            box_max=float(upper_bounds[0]),
        ),
    )


def compute_spectrum(
    positions: np.ndarray,
    mas_library: Any,
    pk_library: Any,
    *,
    grid: int,
    box_size: float,
    mas: str,
    axis: int,
    threads: int,
    kmin: float,
    kmax: float,
    verbose: bool,
) -> Spectrum:
    """Run one sequential Pylians mesh and FFT calculation."""

    n_objects = int(len(positions))
    if n_objects == 0:
        raise PowerSpectrumError("cannot compute a power spectrum for an empty sample")
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise PowerSpectrumError(
            f"positions have shape {positions.shape}, expected (N, 3)"
        )

    delta = np.zeros((grid, grid, grid), dtype=np.float32)
    result = None
    try:
        mas_library.MA(
            positions,
            delta,
            box_size,
            mas,
            verbose=verbose,
        )
        mean_density = float(np.mean(delta, dtype=np.float64))
        if not math.isfinite(mean_density) or mean_density <= 0.0:
            raise PowerSpectrumError(
                f"invalid mean mesh density after {mas} assignment: {mean_density}"
            )
        delta /= mean_density
        delta -= np.float32(1.0)

        result = pk_library.Pk(
            delta,
            box_size,
            axis,
            mas,
            threads,
            verbose,
        )
        k = np.asarray(result.k3D, dtype=np.float64).copy()
        multipoles = np.asarray(result.Pk, dtype=np.float64)
        nmodes = np.asarray(result.Nmodes3D, dtype=np.float64).copy()
        if multipoles.ndim != 2 or multipoles.shape[1] < 3:
            raise PowerSpectrumError(
                f"Pk_library returned multipoles with shape {multipoles.shape}"
            )
        pk0 = multipoles[:, 0].copy()
        pk2 = multipoles[:, 1].copy()
        pk4 = multipoles[:, 2].copy()
    finally:
        del result
        del delta
        gc.collect()

    volume = float(box_size) ** 3
    number_density = n_objects / volume
    shot_noise = volume / n_objects
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pk0 = pk0 * np.sqrt(2.0 / nmodes)
    pk0_shot_subtracted = pk0 - shot_noise

    mask = (
        np.isfinite(k)
        & np.isfinite(pk0)
        & np.isfinite(pk2)
        & np.isfinite(pk4)
        & np.isfinite(sigma_pk0)
        & np.isfinite(nmodes)
        & (nmodes > 0)
        & (k >= kmin)
        & (k <= kmax)
    )
    if not np.any(mask):
        raise PowerSpectrumError(
            f"no finite Pylians shell lies in {kmin} <= k <= {kmax} h/Mpc"
        )

    rounded_modes = np.rint(nmodes[mask]).astype(np.int64)
    if not np.allclose(nmodes[mask], rounded_modes, rtol=0.0, atol=1.0e-6):
        raise PowerSpectrumError("Pk_library returned non-integer Nmodes values")

    return Spectrum(
        k=k[mask],
        pk0_raw=pk0[mask],
        pk0_shot_subtracted=pk0_shot_subtracted[mask],
        pk2=pk2[mask],
        pk4=pk4[mask],
        sigma_pk0=sigma_pk0[mask],
        nmodes=rounded_modes,
        n_objects=n_objects,
        number_density=number_density,
        shot_noise=shot_noise,
    )


def write_spectrum_csv(
    path: Path,
    spectrum: Spectrum,
    *,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    sample: str,
    fof_catalog_sha256: str,
    astra_provenance: AstraProvenance | None,
    box_size: float,
    grid: int,
    mas: str,
    axis: int,
    threads: int,
    kmin: float,
    kmax: float,
    overwrite: bool,
) -> None:
    """Write a standalone CSV atomically so interrupted runs leave no partial file."""

    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    redshift = (
        astra_provenance.redshift
        if astra_provenance is not None
        else SNAPSHOT_REDSHIFTS.get(snapnum)
    )
    k_nyquist = math.pi * grid / box_size
    common: dict[str, Any] = {
        "dataset": dataset,
        "simulation_id": simulation_id,
        "snapshot": snapnum,
        "redshift": "" if redshift is None else _format_float(redshift),
        "sample": sample,
        "tracer": "FoF_halo",
        "n_objects": spectrum.n_objects,
        "n_k_shells": len(spectrum.k),
        "number_density_h3_Mpc3": _format_float(spectrum.number_density),
        "fof_catalog_sha256": fof_catalog_sha256,
        "n_astra_iterations": (
            ""
            if astra_provenance is None
            else astra_provenance.n_iterations
        ),
        "astra_probability_sha256": (
            ""
            if astra_provenance is None
            else astra_provenance.probability_sha256
        ),
        "astra_release": (
            "" if astra_provenance is None else astra_provenance.release
        ),
        "astra_random_seed": (
            ""
            if astra_provenance is None
            else astra_provenance.random_seed
        ),
        "astra_r_lower": (
            ""
            if astra_provenance is None
            else _format_float(astra_provenance.r_lower)
        ),
        "astra_r_med": (
            ""
            if astra_provenance is None
            else _format_float(astra_provenance.r_med)
        ),
        "astra_r_upper": (
            ""
            if astra_provenance is None
            else _format_float(astra_provenance.r_upper)
        ),
        "astra_periodic": (
            "" if astra_provenance is None else str(astra_provenance.periodic)
        ),
        "astra_box_min_Mpc_h": (
            ""
            if astra_provenance is None
            else _format_float(astra_provenance.box_min)
        ),
        "astra_box_max_Mpc_h": (
            ""
            if astra_provenance is None
            else _format_float(astra_provenance.box_max)
        ),
        "box_size_Mpc_h": _format_float(box_size),
        "grid": grid,
        "mass_assignment": mas,
        "los_axis": axis,
        "threads": threads,
        "k_min_h_Mpc": _format_float(kmin),
        "k_max_h_Mpc": _format_float(kmax),
        "k_nyquist_h_Mpc": _format_float(k_nyquist),
        "shot_noise_Mpc3_h3": _format_float(spectrum.shot_noise),
    }

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row_index in range(len(spectrum.k)):
                row = dict(common)
                row.update(
                    {
                        "k_h_Mpc": _format_float(spectrum.k[row_index]),
                        "Pk0_raw_Mpc3_h3": _format_float(
                            spectrum.pk0_raw[row_index]
                        ),
                        "Pk0_shot_subtracted_Mpc3_h3": _format_float(
                            spectrum.pk0_shot_subtracted[row_index]
                        ),
                        "Pk2_Mpc3_h3": _format_float(spectrum.pk2[row_index]),
                        "Pk4_Mpc3_h3": _format_float(spectrum.pk4[row_index]),
                        "sigma_Pk0_notebook_Mpc3_h3": _format_float(
                            spectrum.sigma_pk0[row_index]
                        ),
                        "Nmodes": int(spectrum.nmodes[row_index]),
                    }
                )
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _csv_float(row: Mapping[str, str], column: str, path: Path) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"{path}: column {column} contains a non-numeric value"
        ) from exc
    if not math.isfinite(value):
        raise PowerSpectrumError(
            f"{path}: column {column} contains NaN or infinity"
        )
    return value


def _csv_int(row: Mapping[str, str], column: str, path: Path) -> int:
    try:
        text = row[column]
        value = int(text)
    except (KeyError, TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"{path}: column {column} contains a non-integer value"
        ) from exc
    return value


def _require_csv_float(
    row: Mapping[str, str],
    column: str,
    expected: float,
    path: Path,
) -> None:
    actual = _csv_float(row, column, path)
    if not math.isclose(actual, expected, rel_tol=1.0e-10, abs_tol=1.0e-10):
        raise PowerSpectrumError(
            f"{path}: {column}={actual} does not match requested {expected}"
        )


def validate_existing_spectrum_csv(
    path: Path,
    *,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    sample: str,
    n_objects: int,
    fof_catalog_sha256: str,
    astra_provenance: AstraProvenance | None,
    args: argparse.Namespace,
) -> None:
    """Require an existing CSV to be complete and reproducibly compatible."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CSV_COLUMNS):
                raise PowerSpectrumError(
                    f"{path}: columns do not match the current output schema"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PowerSpectrumError(f"{path}: cannot read CSV: {exc}") from exc
    if not rows:
        raise PowerSpectrumError(f"{path}: CSV contains no power-spectrum rows")

    expected_redshift = (
        astra_provenance.redshift
        if astra_provenance is not None
        else SNAPSHOT_REDSHIFTS.get(snapnum)
    )
    expected_strings = {
        "dataset": dataset,
        "sample": sample,
        "tracer": "FoF_halo",
        "fof_catalog_sha256": fof_catalog_sha256,
        "mass_assignment": args.mas,
        "astra_probability_sha256": (
            ""
            if astra_provenance is None
            else astra_provenance.probability_sha256
        ),
        "astra_release": (
            "" if astra_provenance is None else astra_provenance.release
        ),
        "astra_periodic": (
            "" if astra_provenance is None else str(astra_provenance.periodic)
        ),
    }
    expected_integers = {
        "simulation_id": simulation_id,
        "snapshot": snapnum,
        "n_objects": n_objects,
        "n_k_shells": len(rows),
        "grid": args.grid,
        "los_axis": args.axis,
        "threads": args.threads,
    }
    expected_optional_integers = {
        "n_astra_iterations": (
            None
            if astra_provenance is None
            else astra_provenance.n_iterations
        ),
        "astra_random_seed": (
            None
            if astra_provenance is None
            else astra_provenance.random_seed
        ),
    }
    expected_floats = {
        "number_density_h3_Mpc3": n_objects / args.box_size**3,
        "box_size_Mpc_h": args.box_size,
        "k_min_h_Mpc": args.kmin,
        "k_max_h_Mpc": args.kmax,
        "k_nyquist_h_Mpc": math.pi * args.grid / args.box_size,
    }
    expected_optional_floats = {
        "redshift": expected_redshift,
        "astra_r_lower": (
            None if astra_provenance is None else astra_provenance.r_lower
        ),
        "astra_r_med": (
            None if astra_provenance is None else astra_provenance.r_med
        ),
        "astra_r_upper": (
            None if astra_provenance is None else astra_provenance.r_upper
        ),
        "astra_box_min_Mpc_h": (
            None if astra_provenance is None else astra_provenance.box_min
        ),
        "astra_box_max_Mpc_h": (
            None if astra_provenance is None else astra_provenance.box_max
        ),
    }
    expected_shot_noise = args.box_size**3 / n_objects

    previous_k = -math.inf
    for row_number, row in enumerate(rows, start=2):
        for column, expected in expected_strings.items():
            if row.get(column) != expected:
                raise PowerSpectrumError(
                    f"{path}:{row_number}: {column}={row.get(column)!r} "
                    f"does not match {expected!r}"
                )
        for column, expected in expected_integers.items():
            if _csv_int(row, column, path) != expected:
                raise PowerSpectrumError(
                    f"{path}:{row_number}: {column} does not match {expected}"
                )
        for column, expected in expected_optional_integers.items():
            text = row.get(column, "")
            if expected is None:
                if text != "":
                    raise PowerSpectrumError(
                        f"{path}:{row_number}: {column} should be empty"
                    )
            elif _csv_int(row, column, path) != expected:
                raise PowerSpectrumError(
                    f"{path}:{row_number}: {column} does not match {expected}"
                )
        for column, expected in expected_floats.items():
            _require_csv_float(row, column, expected, path)
        for column, expected in expected_optional_floats.items():
            text = row.get(column, "")
            if expected is None:
                if text != "":
                    raise PowerSpectrumError(
                        f"{path}:{row_number}: {column} should be empty"
                    )
            else:
                _require_csv_float(row, column, expected, path)

        k = _csv_float(row, "k_h_Mpc", path)
        pk0 = _csv_float(row, "Pk0_raw_Mpc3_h3", path)
        shot_noise = _csv_float(row, "shot_noise_Mpc3_h3", path)
        pk0_subtracted = _csv_float(
            row, "Pk0_shot_subtracted_Mpc3_h3", path
        )
        _csv_float(row, "Pk2_Mpc3_h3", path)
        _csv_float(row, "Pk4_Mpc3_h3", path)
        sigma = _csv_float(row, "sigma_Pk0_notebook_Mpc3_h3", path)
        nmodes = _csv_int(row, "Nmodes", path)
        if nmodes <= 0:
            raise PowerSpectrumError(
                f"{path}:{row_number}: Nmodes must be positive"
            )
        if k <= previous_k or k < args.kmin or k > args.kmax:
            raise PowerSpectrumError(
                f"{path}:{row_number}: k values are not strictly increasing "
                "inside the requested interval"
            )
        if not math.isclose(
            shot_noise,
            expected_shot_noise,
            rel_tol=1.0e-10,
            abs_tol=1.0e-8,
        ):
            raise PowerSpectrumError(
                f"{path}:{row_number}: shot noise is incompatible with V/N"
            )
        if not math.isclose(
            pk0_subtracted,
            pk0 - shot_noise,
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        ):
            raise PowerSpectrumError(
                f"{path}:{row_number}: shot-noise-subtracted P0 is inconsistent"
            )
        expected_sigma = pk0 * math.sqrt(2.0 / nmodes)
        if not math.isclose(
            sigma,
            expected_sigma,
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        ):
            raise PowerSpectrumError(
                f"{path}:{row_number}: notebook sigma(P0) is inconsistent"
            )
        previous_k = k


def output_needs_computation(
    path: Path,
    *,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    sample: str,
    n_objects: int,
    fof_catalog_sha256: str,
    astra_provenance: AstraProvenance | None,
    args: argparse.Namespace,
    counters: Counters,
) -> bool:
    if not path.exists() or args.overwrite:
        return True
    try:
        validate_existing_spectrum_csv(
            path,
            dataset=dataset,
            simulation_id=simulation_id,
            snapnum=snapnum,
            sample=sample,
            n_objects=n_objects,
            fof_catalog_sha256=fof_catalog_sha256,
            astra_provenance=astra_provenance,
            args=args,
        )
    except PowerSpectrumError as exc:
        raise PowerSpectrumError(
            f"existing output is incompatible or corrupt: {exc}. "
            "Use --overwrite to replace it."
        ) from exc
    _log(f"[skip] validated existing product: {path}")
    counters.skipped += 1
    return False


def _compute_and_write(
    *,
    positions: np.ndarray,
    sample: str,
    output_path: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    fof_catalog_sha256: str,
    astra_provenance: AstraProvenance | None,
    args: argparse.Namespace,
    mas_library: Any,
    pk_library: Any,
    counters: Counters,
) -> None:
    start = time.perf_counter()
    _log(
        f"[compute] {dataset} sim={simulation_id} snap={snapnum} "
        f"sample={sample} N={len(positions):,}"
    )
    spectrum = compute_spectrum(
        positions,
        mas_library,
        pk_library,
        grid=args.grid,
        box_size=args.box_size,
        mas=args.mas,
        axis=args.axis,
        threads=args.threads,
        kmin=args.kmin,
        kmax=args.kmax,
        verbose=not args.quiet_pylians,
    )
    write_spectrum_csv(
        output_path,
        spectrum,
        dataset=dataset,
        simulation_id=simulation_id,
        snapnum=snapnum,
        sample=sample,
        fof_catalog_sha256=fof_catalog_sha256,
        astra_provenance=astra_provenance,
        box_size=args.box_size,
        grid=args.grid,
        mas=args.mas,
        axis=args.axis,
        threads=args.threads,
        kmin=args.kmin,
        kmax=args.kmax,
        overwrite=args.overwrite,
    )
    elapsed = time.perf_counter() - start
    _log(
        f"[write] {output_path} ({len(spectrum.k)} k shells, {elapsed:.1f} s)"
    )
    counters.written += 1
    del spectrum
    gc.collect()


def process_simulation(
    *,
    dataset: str,
    simulation_id: int,
    args: argparse.Namespace,
    readfof: Any | None,
    mas_library: Any | None,
    pk_library: Any | None,
    counters: Counters,
) -> None:
    catalogue = catalogue_directory(
        args.data_root, dataset, simulation_id, args.snapnum
    )
    matter_path, environment_paths = output_paths(
        args.output_root, dataset, simulation_id, args.snapnum
    )
    do_matter = not args.environments_only
    do_environments = not args.matter_only

    if args.dry_run:
        _log(
            f"[plan] {dataset} sim={simulation_id} snap={args.snapnum} "
            f"catalogue={catalogue}"
        )
        if do_matter:
            state = "overwrite" if matter_path.exists() else "write"
            if matter_path.exists() and not args.overwrite:
                state = "skip-existing"
            _log(f"       matter ({state}): {matter_path}")
        if do_environments:
            try:
                probability = resolve_probability_path(
                    args.astra_root,
                    dataset,
                    simulation_id,
                    args.snapnum,
                )
                _log(f"       probability: {probability}")
            except FileNotFoundError as exc:
                _log(f"       probability: MISSING ({exc})")
            for name, _ in ENVIRONMENT_COLUMNS:
                path = environment_paths[name]
                state = "overwrite" if path.exists() else "write"
                if path.exists() and not args.overwrite:
                    state = "skip-existing"
                _log(f"       {name} ({state}): {path}")
        return

    if readfof is None or mas_library is None or pk_library is None:
        raise AssertionError("Pylians modules were not loaded")
    positions, fof_catalog_sha256 = read_fof_positions(
        readfof, catalogue, args.snapnum, args.box_size
    )
    _log(
        f"[input] {dataset} sim={simulation_id}: read {len(positions):,} "
        f"FoF halo positions (sha256={fof_catalog_sha256[:12]}...)"
    )

    if do_matter:
        if output_needs_computation(
            matter_path,
            dataset=dataset,
            simulation_id=simulation_id,
            snapnum=args.snapnum,
            sample="all",
            n_objects=len(positions),
            fof_catalog_sha256=fof_catalog_sha256,
            astra_provenance=None,
            args=args,
            counters=counters,
        ):
            _compute_and_write(
                positions=positions,
                sample="all",
                output_path=matter_path,
                dataset=dataset,
                simulation_id=simulation_id,
                snapnum=args.snapnum,
                fof_catalog_sha256=fof_catalog_sha256,
                astra_provenance=None,
                args=args,
                mas_library=mas_library,
                pk_library=pk_library,
                counters=counters,
            )

    if not do_environments:
        del positions
        return

    try:
        probability_path = resolve_probability_path(
            args.astra_root,
            dataset,
            simulation_id,
            args.snapnum,
        )
    except FileNotFoundError:
        counters.missing_probabilities += 1
        del positions
        if args.allow_missing_probabilities:
            _log(
                f"[warning] ASTRA probabilities are not finished for {dataset} "
                f"sim={simulation_id}; environment spectra skipped"
            )
            return
        raise

    assignment = read_environment_assignment(
        probability_path,
        len(positions),
        simulation_id,
        args.snapnum,
        args.box_size,
    )
    counts = ", ".join(
        f"{name}={len(assignment.indices[name]):,}"
        for name, _ in ENVIRONMENT_COLUMNS
    )
    _log(
        f"[class] {probability_path} "
        f"(sha256={assignment.provenance.probability_sha256[:12]}..., {counts})"
    )

    sample_errors: list[str] = []
    for name, _ in ENVIRONMENT_COLUMNS:
        output_path = environment_paths[name]
        try:
            indices = assignment.indices[name]
            if len(indices) == 0:
                raise PowerSpectrumError(
                    "the hard ASTRA assignment contains no halos"
                )
            if not output_needs_computation(
                output_path,
                dataset=dataset,
                simulation_id=simulation_id,
                snapnum=args.snapnum,
                sample=name,
                n_objects=len(indices),
                fof_catalog_sha256=fof_catalog_sha256,
                astra_provenance=assignment.provenance,
                args=args,
                counters=counters,
            ):
                continue
            sample_positions = np.ascontiguousarray(
                positions[indices], dtype=np.float32
            )
            try:
                _compute_and_write(
                    positions=sample_positions,
                    sample=name,
                    output_path=output_path,
                    dataset=dataset,
                    simulation_id=simulation_id,
                    snapnum=args.snapnum,
                    fof_catalog_sha256=fof_catalog_sha256,
                    astra_provenance=assignment.provenance,
                    args=args,
                    mas_library=mas_library,
                    pk_library=pk_library,
                    counters=counters,
                )
            finally:
                del sample_positions
                gc.collect()
        except Exception as exc:
            message = (
                f"{dataset} sim={simulation_id} snap={args.snapnum} "
                f"sample={name}: {exc}"
            )
            _error(f"[error] {message}")
            sample_errors.append(message)
            if args.fail_fast:
                raise
    del assignment, positions
    gc.collect()
    if sample_errors:
        raise PowerSpectrumError(
            f"{len(sample_errors)} environment sample(s) failed; "
            "see the preceding errors"
        )


def selected_datasets(args: argparse.Namespace) -> list[str]:
    if [value.lower() for value in args.datasets] == ["all"]:
        return discover_datasets(args.data_root, args.snapnum)
    return list(dict.fromkeys(args.datasets))


def selected_simulations(
    args: argparse.Namespace,
    dataset: str,
) -> list[int]:
    if args.simulation_ids is None:
        return discover_simulations(args.data_root, dataset, args.snapnum)
    return list(args.simulation_ids)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    args.data_root = _path(args.data_root)
    args.astra_root = _path(
        args.astra_root if args.astra_root is not None else args.data_root / "astra"
    )
    args.output_root = _path(
        args.output_root if args.output_root is not None else args.data_root / "pk"
    )

    try:
        datasets = selected_datasets(args)
    except Exception as exc:
        _error(f"[error] {exc}")
        return 1

    readfof = mas_library = pk_library = None
    if not args.dry_run:
        try:
            readfof, mas_library, pk_library = import_pylians()
        except Exception as exc:
            _error(f"[error] {exc}")
            return 1

    counters = Counters()
    total_start = time.perf_counter()
    for dataset in datasets:
        try:
            simulations = selected_simulations(args, dataset)
        except Exception as exc:
            counters.failures += 1
            _error(f"[error] {dataset}: {exc}")
            if args.fail_fast:
                break
            continue

        for simulation_id in simulations:
            try:
                process_simulation(
                    dataset=dataset,
                    simulation_id=simulation_id,
                    args=args,
                    readfof=readfof,
                    mas_library=mas_library,
                    pk_library=pk_library,
                    counters=counters,
                )
            except Exception as exc:
                counters.failures += 1
                _error(
                    f"[error] {dataset} sim={simulation_id} "
                    f"snap={args.snapnum}: {exc}"
                )
                if args.fail_fast:
                    break
        if args.fail_fast and counters.failures:
            break

    elapsed = time.perf_counter() - total_start
    _log(
        "[summary] "
        f"written={counters.written}, skipped={counters.skipped}, "
        f"missing_probabilities={counters.missing_probabilities}, "
        f"failures={counters.failures}, elapsed={elapsed:.1f} s"
    )
    return 1 if counters.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
