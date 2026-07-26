#!/usr/bin/env python3
"""Compute Quijote power spectra for halos and ASTRA environment samples.

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

The random-void sample is selected from every split ASTRA classification FITS.
The selected ``TARGETID`` values are crossmatched against ASTRA's raw FITS and
their stored ``XCART``, ``YCART``, and ``ZCART`` coordinates are used directly.
Compressed raw products are expanded temporarily so they can be memory-mapped.
Once all spectra are available, their raw monopoles are also combined into the
dark-background normalized environment plot used in ``power_spec.ipynb``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass, replace
import gc
import gzip
import hashlib
import importlib
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence

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
RANDOM_VOID_SAMPLE = "random_void"
RANDOM_VOID_TRACER = "uniform_random"
RANDOM_VOID_SELECTION_RULE = (
    "ISDATA=False and (NDATA-NRAND)/(NDATA+NRAND)<RLOWER; "
    "ratio=0 when NDATA+NRAND=0; union by TARGETID"
)
NORMALIZED_PLOT_SAMPLES = (
    ("void", "Voids", "#17becf"),
    ("sheet", "Sheets", "orange"),
    ("filament", "Filaments", "limegreen"),
    ("knot", "Knots", "magenta"),
    (RANDOM_VOID_SAMPLE, "Random voids", "#ffd166"),
)
PLOT_SHARED_METADATA_COLUMNS = (
    "dataset",
    "simulation_id",
    "snapshot",
    "redshift",
    "fof_catalog_sha256",
    "box_size_Mpc_h",
    "grid",
    "mass_assignment",
    "los_axis",
    "threads",
    "k_min_h_Mpc",
    "k_max_h_Mpc",
    "k_nyquist_h_Mpc",
)
PLOT_ENVIRONMENT_METADATA_COLUMNS = (
    "n_astra_iterations",
    "astra_release",
    "astra_random_seed",
    "astra_r_lower",
    "astra_r_med",
    "astra_r_upper",
    "astra_periodic",
    "astra_box_min_Mpc_h",
    "astra_box_max_Mpc_h",
)
PLOT_METADATA_COLUMNS = (
    PLOT_SHARED_METADATA_COLUMNS
    + PLOT_ENVIRONMENT_METADATA_COLUMNS
    + ("astra_probability_sha256",)
)

LEGACY_CSV_COLUMNS = (
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
CSV_COLUMNS = LEGACY_CSV_COLUMNS + (
    "astra_classification_sha256",
    "astra_classification_files",
    "astra_raw_sha256",
    "selection_rule",
)


class PowerSpectrumError(RuntimeError):
    """Raised when an input or a numerical result is not safe to use."""


class AstraProductsIncompleteError(FileNotFoundError):
    """Raised when an otherwise valid ASTRA iteration series is unfinished."""


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
class RandomVoidSelection:
    """Unique random target IDs classified as void in at least one iteration."""

    target_ids: np.ndarray
    n_objects: int
    provenance: "AstraProvenance"


@dataclass(frozen=True)
class PowerSpectrumCurve:
    """Raw monopole and stable CSV metadata used by the normalized plot."""

    k: np.ndarray
    pk0_raw: np.ndarray
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class NormalizedPowerSpectra:
    """Common k shells and five positive raw-monopole ratios."""

    k: np.ndarray
    ratios: Mapping[str, np.ndarray]
    source_paths: tuple[Path, ...]


@dataclass(frozen=True)
class AstraProvenance:
    """Validated ASTRA configuration and source-product identities."""

    probability_path: Path | None
    probability_sha256: str
    classification_paths: tuple[Path, ...]
    classification_sha256: str
    raw_path: Path | None
    raw_sha256: str
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
    plots_written: int = 0
    plots_skipped: int = 0
    missing_probabilities: int = 0
    missing_classifications: int = 0
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
            "Compute Pylians P(k) for all Quijote FoF halos, the four hard "
            "ASTRA halo environments, and random targets classified as void "
            "in at least one iteration."
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
        help=(
            "root containing ASTRA runs as DATASET/SIMULATION or "
            "DATASET_SIMULATION; defaults to DATA_ROOT (an optional "
            "DATA_ROOT/astra subdirectory is also searched)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=_path,
        default=None,
        help="output root; defaults to DATA_ROOT/pk",
    )
    parser.add_argument(
        "--raw-temp-dir",
        type=_path,
        default=None,
        help=(
            "parent directory for temporary decompression of raw FITS files; "
            "defaults to the system temporary directory"
        ),
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
        help=(
            "write only the four ASTRA halo environments and random-void "
            "spectrum under env/"
        ),
    )
    sample_group.add_argument(
        "--random-void-only",
        action="store_true",
        help=(
            "compute only the random-void spectrum; refresh the normalized "
            "plot too when its other CSVs already exist"
        ),
    )
    parser.add_argument(
        "--skip-random-void",
        action="store_true",
        help="do not calculate the new random-void spectrum",
    )
    parser.add_argument(
        "--skip-normalized-plot",
        "--skip-plot",
        dest="skip_normalized_plot",
        action="store_true",
        help="do not create the normalized five-environment PNG",
    )
    parser.add_argument(
        "--allow-missing-astra",
        "--allow-missing-probabilities",
        dest="allow_missing_astra",
        action="store_true",
        help=(
            "skip samples whose final ASTRA probability or classification "
            "products are unfinished, without treating them as batch failures"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSV and derived PNG products atomically",
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
    if args.random_void_only and args.skip_random_void:
        parser.error("--random-void-only cannot be combined with --skip-random-void")
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
    """Find both ``dataset_sim`` and ``dataset/sim`` ASTRA run layouts."""

    matcher = re.compile(rf"^{re.escape(dataset)}_(\d+)$")
    candidates: list[Path] = []
    search_roots = [astra_root]
    nested_astra_root = astra_root / "astra"
    if nested_astra_root.is_dir():
        search_roots.append(nested_astra_root)
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        own_match = matcher.match(search_root.name)
        if own_match and int(own_match.group(1)) == simulation_id:
            candidates.append(search_root)
        if (
            search_root.name.isdigit()
            and int(search_root.name) == simulation_id
            and search_root.parent.name == dataset
        ):
            candidates.append(search_root)
        for child in search_root.iterdir():
            if not child.is_dir():
                continue
            match = matcher.match(child.name)
            if match and int(match.group(1)) == simulation_id:
                candidates.append(child)
            if (
                search_root.name == dataset
                and child.name.isdigit()
                and int(child.name) == simulation_id
            ):
                candidates.append(child)
        dataset_root = search_root / dataset
        if dataset_root.is_dir():
            candidates.extend(
                child
                for child in dataset_root.iterdir()
                if child.is_dir()
                and child.name.isdigit()
                and int(child.name) == simulation_id
            )
    return sorted(set(path.resolve() for path in candidates))


def _expected_astra_run_message(
    astra_root: Path,
    dataset: str,
    simulation_id: int,
) -> str:
    """Describe the two supported run-directory layouts in errors."""

    if (
        astra_root.name.isdigit()
        and int(astra_root.name) == simulation_id
        and astra_root.parent.name == dataset
    ):
        nested_example = astra_root
    elif astra_root.name == dataset:
        nested_example = astra_root / str(simulation_id)
    else:
        nested_example = astra_root / dataset / str(simulation_id)
    legacy_match = re.match(
        rf"^{re.escape(dataset)}_(\d+)$",
        astra_root.name,
    )
    if legacy_match and int(legacy_match.group(1)) == simulation_id:
        legacy_example = astra_root
    else:
        legacy_example = astra_root / f"{dataset}_{simulation_id}"
    return f"{nested_example} or {legacy_example}"


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
        raise FileNotFoundError(
            f"ASTRA run directory not found for {dataset} simulation "
            f"{simulation_id}; expected a directory such as "
            f"{_expected_astra_run_message(astra_root, dataset, simulation_id)}"
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


def resolve_classification_paths(
    astra_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> tuple[Path, ...]:
    """Find the final split classification FITS files for every iteration."""

    sim_tag = f"sim{simulation_id:03d}"
    snap_tag = f"snap{snapnum:03d}"
    filename_re = re.compile(
        rf"^zone_00_{sim_tag}_{snap_tag}_iter(\d+)\.fits(?:\.gz)?$"
    )
    run_directories = _matching_run_directories(
        astra_root, dataset, simulation_id
    )
    if not run_directories:
        raise FileNotFoundError(
            f"ASTRA run directory not found for {dataset} simulation "
            f"{simulation_id}; expected a directory such as "
            f"{_expected_astra_run_message(astra_root, dataset, simulation_id)}"
        )

    matches_by_run: dict[Path, dict[int, Path]] = {}
    for run_directory in run_directories:
        by_iteration: dict[int, Path] = {}
        pattern = f"zone_00_{sim_tag}_{snap_tag}_iter*.fits*"
        for path in run_directory.rglob(pattern):
            if not path.is_file():
                continue
            match = filename_re.match(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            resolved = path.resolve()
            previous = by_iteration.get(iteration)
            if previous is not None and previous != resolved:
                raise PowerSpectrumError(
                    f"multiple classification FITS files found for iteration "
                    f"{iteration}: {previous}, {resolved}"
                )
            by_iteration[iteration] = resolved
        if by_iteration:
            matches_by_run[run_directory] = by_iteration

    if len(matches_by_run) > 1:
        joined = "\n  ".join(str(path) for path in matches_by_run)
        raise PowerSpectrumError(
            "more than one ASTRA run contains split classifications; "
            f"pass a narrower --astra-root:\n  {joined}"
        )
    if not matches_by_run:
        searched = ", ".join(str(path) for path in run_directories)
        raise FileNotFoundError(
            f"final ASTRA split classification files not found for {dataset} "
            f"simulation {simulation_id}, snapshot {snapnum}; searched: {searched}"
        )

    by_iteration = next(iter(matches_by_run.values()))
    iterations = sorted(by_iteration)
    if iterations != list(range(len(iterations))):
        raise AstraProductsIncompleteError(
            "classification iteration files must be contiguous from zero; "
            f"found {iterations}"
        )
    return tuple(by_iteration[iteration] for iteration in iterations)


def resolve_raw_path(
    astra_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> Path:
    """Find the ASTRA raw FITS containing real and random Cartesian positions."""

    filename = (
        f"zone_00_sim{simulation_id:03d}_snap{snapnum:03d}.fits"
    )
    names = (f"{filename}.gz", filename)
    run_directories = _matching_run_directories(
        astra_root, dataset, simulation_id
    )
    if not run_directories:
        raise FileNotFoundError(
            f"ASTRA run directory not found for {dataset} simulation "
            f"{simulation_id}; expected a directory such as "
            f"{_expected_astra_run_message(astra_root, dataset, simulation_id)}"
        )
    matches_by_run: dict[Path, list[Path]] = {}
    for run_directory in run_directories:
        matches: list[Path] = []
        for name in names:
            matches.extend(run_directory.rglob(name))
        matches = sorted(set(path.resolve() for path in matches if path.is_file()))
        if matches:
            matches_by_run[run_directory] = matches
    if len(matches_by_run) > 1:
        joined = "\n  ".join(str(path) for path in matches_by_run)
        raise PowerSpectrumError(
            "more than one ASTRA run contains a raw FITS; pass a narrower "
            f"--astra-root:\n  {joined}"
        )
    if len(matches_by_run) == 1:
        matches = next(iter(matches_by_run.values()))
        if len(matches) == 1:
            return matches[0]
        joined = "\n  ".join(str(path) for path in matches)
        raise PowerSpectrumError(
            f"multiple ASTRA raw FITS products found:\n  {joined}"
        )
    searched = ", ".join(str(path) for path in run_directories)
    raise FileNotFoundError(
        f"ASTRA raw FITS not found for {dataset} simulation {simulation_id}, "
        f"snapshot {snapnum}; searched: {searched}"
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
    environments[RANDOM_VOID_SAMPLE] = (
        output_root / "env" / f"{stem}_{RANDOM_VOID_SAMPLE}_pk.csv"
    )
    return matter, environments


def normalized_plot_path(
    output_root: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
) -> Path:
    """Return the PNG path stored beside the environment spectra."""

    stem = f"{dataset}_sim{simulation_id:03d}_snap{snapnum:03d}"
    return output_root / "env" / f"{stem}_normalized_pk.png"


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
    raise PowerSpectrumError(f"ASTRA product header is missing {key}")


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = _metadata_value(metadata, key)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"ASTRA product header {key}={value!r} is not an integer"
        ) from exc
    return parsed


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float:
    value = _metadata_value(metadata, key)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PowerSpectrumError(
            f"ASTRA product header {key}={value!r} is not numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise PowerSpectrumError(
            f"ASTRA product header {key}={value!r} is not finite"
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
        f"ASTRA product header {key}={value!r} is not boolean"
    )


def _astra_provenance_from_metadata(
    metadata: Mapping[str, Any],
    *,
    simulation_id: int,
    snapnum: int,
    box_size: float,
    probability_path: Path | None = None,
    probability_sha256: str = "",
    classification_paths: Sequence[Path] = (),
    classification_sha256: str = "",
) -> AstraProvenance:
    """Validate shared Quijote ASTRA metadata and attach source identities."""

    meta_simulation = _metadata_int(metadata, "SIMID")
    meta_snapshot = _metadata_int(metadata, "SNAPNUM")
    if meta_simulation != simulation_id:
        raise PowerSpectrumError(
            f"SIMID={meta_simulation}, expected {simulation_id}"
        )
    if meta_snapshot != snapnum:
        raise PowerSpectrumError(
            f"SNAPNUM={meta_snapshot}, expected {snapnum}"
        )
    release = str(_metadata_value(metadata, "RELEASE")).strip()
    if release.upper() != "QUIJOTES":
        raise PowerSpectrumError(
            f"RELEASE={release!r}, expected QUIJOTES"
        )
    n_iterations = _metadata_int(metadata, "NITER")
    random_seed = _metadata_int(metadata, "RNGSEED")
    if n_iterations <= 0:
        raise PowerSpectrumError(f"NITER must be positive, found {n_iterations}")
    if random_seed < 0:
        raise PowerSpectrumError(
            f"RNGSEED must be non-negative, found {random_seed}"
        )
    r_lower = _metadata_float(metadata, "RLOWER")
    r_med = _metadata_float(metadata, "RMED")
    r_upper = _metadata_float(metadata, "RUPPER")
    if not r_lower < r_med < r_upper:
        raise PowerSpectrumError(
            "ASTRA thresholds must satisfy RLOWER < RMED < RUPPER; found "
            f"{r_lower}, {r_med}, {r_upper}"
        )
    redshift = _metadata_float(metadata, "REDSHFT")
    periodic = _metadata_bool(metadata, "PERIODIC")
    if not periodic:
        raise PowerSpectrumError(
            "ASTRA product is not marked PERIODIC for a Quijote box"
        )
    lower_bounds = np.array(
        [
            _metadata_float(metadata, "BOXXMIN"),
            _metadata_float(metadata, "BOXYMIN"),
            _metadata_float(metadata, "BOXZMIN"),
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            _metadata_float(metadata, "BOXXMAX"),
            _metadata_float(metadata, "BOXYMAX"),
            _metadata_float(metadata, "BOXZMAX"),
        ],
        dtype=np.float64,
    )
    if not np.allclose(lower_bounds, 0.0, rtol=0.0, atol=1.0e-8):
        raise PowerSpectrumError(
            f"ASTRA box lower bounds are {lower_bounds.tolist()}, expected [0, 0, 0]"
        )
    if not np.allclose(
        upper_bounds,
        box_size,
        rtol=0.0,
        atol=max(1.0e-8, box_size * 1.0e-10),
    ):
        raise PowerSpectrumError(
            f"ASTRA box upper bounds are {upper_bounds.tolist()}, expected "
            f"[{box_size}, {box_size}, {box_size}] Mpc/h"
        )
    return AstraProvenance(
        probability_path=probability_path,
        probability_sha256=probability_sha256,
        classification_paths=tuple(classification_paths),
        classification_sha256=classification_sha256,
        raw_path=None,
        raw_sha256="",
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

    probability_sha256 = _sha256_files([probability_path])
    provenance = _astra_provenance_from_metadata(
        table.meta,
        simulation_id=simulation_id,
        snapnum=snapnum,
        box_size=box_size,
        probability_path=probability_path,
        probability_sha256=probability_sha256,
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

    del table, probabilities, probability_columns, hard_class
    return EnvironmentAssignment(
        indices=indices,
        provenance=provenance,
    )


def read_random_void_selection(
    classification_paths: Sequence[Path],
    n_halos: int,
    simulation_id: int,
    snapnum: int,
    box_size: float,
) -> RandomVoidSelection:
    """Read all split classifications and select the union of random voids."""

    try:
        from astropy.table import Table
    except ImportError as exc:
        raise PowerSpectrumError(
            "astropy is required to read ASTRA classification FITS products"
        ) from exc

    paths = tuple(Path(path).resolve() for path in classification_paths)
    if not paths:
        raise PowerSpectrumError("no split ASTRA classification files were provided")
    classification_sha256 = _sha256_files(paths)
    required = (
        "TARGETID",
        "RANDITER",
        "ISDATA",
        "NDATA",
        "NRAND",
        "TRACERTYPE",
    )
    selected_id_chunks: list[np.ndarray] = []
    provenance: AstraProvenance | None = None
    total_selected = 0

    for iteration, path in enumerate(paths):
        table = Table.read(path, hdu=1)
        missing = [column for column in required if column not in table.colnames]
        if missing:
            raise PowerSpectrumError(
                f"{path} is missing columns: {', '.join(missing)}"
            )
        current_provenance = _astra_provenance_from_metadata(
            table.meta,
            simulation_id=simulation_id,
            snapnum=snapnum,
            box_size=box_size,
            classification_paths=paths,
            classification_sha256=classification_sha256,
        )
        if provenance is None:
            provenance = current_provenance
            if len(paths) != provenance.n_iterations:
                message = (
                    f"found {len(paths)} classification files, but NITER="
                    f"{provenance.n_iterations}"
                )
                if len(paths) < provenance.n_iterations:
                    raise AstraProductsIncompleteError(message)
                raise PowerSpectrumError(message)
        elif current_provenance != provenance:
            raise PowerSpectrumError(
                f"ASTRA metadata in {path} differs from the other "
                "classification iterations"
            )

        if len(table) != 2 * n_halos:
            raise PowerSpectrumError(
                f"{path} contains {len(table)} rows, expected "
                f"{2 * n_halos} (N data + N random)"
            )
        target_ids = np.asarray(table["TARGETID"], dtype=np.int64)
        randiters = np.asarray(table["RANDITER"], dtype=np.int64)
        is_data = np.asarray(table["ISDATA"], dtype=bool)
        ndata = np.asarray(table["NDATA"], dtype=np.int64)
        nrand = np.asarray(table["NRAND"], dtype=np.int64)
        if any(
            values.ndim != 1 or len(values) != len(table)
            for values in (target_ids, randiters, is_data, ndata, nrand)
        ):
            raise PowerSpectrumError(
                f"{path} contains a malformed classification column"
            )
        if not np.all(randiters == iteration):
            found = np.unique(randiters).tolist()
            raise PowerSpectrumError(
                f"{path} contains RANDITER={found}, expected only {iteration}"
            )
        if np.any(ndata < 0) or np.any(nrand < 0):
            raise PowerSpectrumError(
                f"{path} contains negative NDATA or NRAND values"
            )
        if int(np.count_nonzero(is_data)) != n_halos:
            raise PowerSpectrumError(
                f"{path} does not contain exactly {n_halos} data rows"
            )

        tracer_values = np.asarray(table["TRACERTYPE"]).astype(str)
        if not np.all(
            np.char.upper(np.char.strip(tracer_values)) == "HALO"
        ):
            unique = np.unique(tracer_values).tolist()
            raise PowerSpectrumError(
                f"{path} contains non-HALO TRACERTYPE values: {unique}"
            )

        expected_data_ids = np.arange(1, n_halos + 1, dtype=np.int64)
        data_ids = np.sort(target_ids[is_data])
        if not np.array_equal(data_ids, expected_data_ids):
            raise PowerSpectrumError(
                f"{path} data TARGETID values are not the complete range "
                f"1..{n_halos}"
            )
        random_start = n_halos * (iteration + 1) + 1
        random_stop = random_start + n_halos
        expected_random_ids = np.arange(
            random_start, random_stop, dtype=np.int64
        )
        random_mask = ~is_data
        random_ids = target_ids[random_mask]
        if not np.array_equal(np.sort(random_ids), expected_random_ids):
            raise PowerSpectrumError(
                f"{path} random TARGETID values are not the complete range "
                f"{random_start}..{random_stop - 1}"
            )

        ndata_random = ndata[random_mask].astype(np.float32, copy=False)
        nrand_random = nrand[random_mask].astype(np.float32, copy=False)
        denominator = ndata_random + nrand_random
        ratio = np.zeros_like(denominator, dtype=np.float32)
        np.divide(
            ndata_random - nrand_random,
            denominator,
            out=ratio,
            where=denominator > 0,
        )
        void_mask = ratio < np.float32(current_provenance.r_lower)
        selected_ids = np.ascontiguousarray(
            random_ids[void_mask], dtype=np.int64
        )
        # The complete, unique per-iteration ID ranges validated above are
        # disjoint, so concatenating is exactly the TARGETID union.
        selected_id_chunks.append(selected_ids)
        total_selected += len(selected_ids)
        del table, target_ids, randiters, is_data, ndata, nrand
        gc.collect()

    if provenance is None:
        raise AssertionError("classification provenance was not initialized")
    if total_selected == 0:
        raise PowerSpectrumError(
            "no random target was classified as void in any iteration"
        )
    target_ids = np.sort(np.concatenate(selected_id_chunks))
    if len(np.unique(target_ids)) != total_selected:
        raise PowerSpectrumError(
            "random-void TARGETID values are duplicated across iterations"
        )
    return RandomVoidSelection(
        target_ids=np.ascontiguousarray(target_ids, dtype=np.int64),
        n_objects=total_selected,
        provenance=provenance,
    )


@contextmanager
def _materialized_raw_fits(
    raw_path: Path,
    temp_parent: Path | None,
    raw_sha256: str | None = None,
) -> Iterator[tuple[Path, str]]:
    """Yield a memmap-compatible raw FITS path and clean temporary expansion."""

    raw_path = raw_path.resolve()
    if raw_sha256 is None:
        raw_sha256 = _sha256_files([raw_path])
    if raw_path.suffix.lower() != ".gz":
        yield raw_path, raw_sha256
        return

    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)
        parent_text = str(temp_parent)
    else:
        parent_text = None
    with tempfile.TemporaryDirectory(
        prefix="quijotes_pk_raw_",
        dir=parent_text,
    ) as temporary_directory:
        expanded = Path(temporary_directory) / raw_path.stem
        _log(
            f"[raw] temporarily decompressing {raw_path} -> {expanded}"
        )
        with gzip.open(raw_path, "rb") as source, expanded.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        yield expanded, raw_sha256


def read_random_void_positions_from_raw(
    raw_path: Path,
    selection: RandomVoidSelection,
    *,
    n_halos: int,
    simulation_id: int,
    snapnum: int,
    box_size: float,
    temp_parent: Path | None,
    raw_sha256: str | None = None,
) -> tuple[np.ndarray, AstraProvenance]:
    """Crossmatch selected TARGETIDs against raw and return their coordinates."""

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise PowerSpectrumError(
            "astropy is required to read the ASTRA raw FITS product"
        ) from exc

    with _materialized_raw_fits(raw_path, temp_parent, raw_sha256) as (
        readable_path,
        raw_sha256,
    ):
        with fits.open(readable_path, mode="readonly", memmap=True) as hdul:
            if len(hdul) < 2 or hdul[1].data is None:
                raise PowerSpectrumError(
                    f"{raw_path} does not contain a binary table in HDU 1"
                )
            header = hdul[1].header
            data = hdul[1].data
            names = set(data.names or ())
            required = {"TARGETID", "RANDITER", "XCART", "YCART", "ZCART"}
            missing = sorted(required - names)
            if missing:
                raise PowerSpectrumError(
                    f"{raw_path} is missing columns: {', '.join(missing)}"
                )

            raw_simulation = _metadata_int(header, "SIMID")
            raw_snapshot = _metadata_int(header, "SNAPNUM")
            raw_iterations = _metadata_int(header, "NITER")
            raw_seed = _metadata_int(header, "RNGSEED")
            raw_nreal = _metadata_int(header, "NREAL")
            raw_nrandpt = _metadata_int(header, "NRANDPT")
            raw_periodic = _metadata_bool(header, "PERIODIC")
            raw_release = str(_metadata_value(header, "RELEASE")).strip()
            raw_redshift = _metadata_float(header, "REDSHFT")
            if raw_simulation != simulation_id or raw_snapshot != snapnum:
                raise PowerSpectrumError(
                    f"{raw_path} identifies SIMID={raw_simulation}, "
                    f"SNAPNUM={raw_snapshot}; expected {simulation_id}, {snapnum}"
                )
            if raw_iterations != selection.provenance.n_iterations:
                raise PowerSpectrumError(
                    f"{raw_path} NITER={raw_iterations} differs from "
                    f"classification NITER={selection.provenance.n_iterations}"
                )
            if raw_seed != selection.provenance.random_seed:
                raise PowerSpectrumError(
                    f"{raw_path} RNGSEED={raw_seed} differs from "
                    f"classification RNGSEED={selection.provenance.random_seed}"
                )
            if raw_release.upper() != selection.provenance.release.upper():
                raise PowerSpectrumError(
                    f"{raw_path} RELEASE={raw_release!r} differs from "
                    f"classification RELEASE={selection.provenance.release!r}"
                )
            if not math.isclose(
                raw_redshift,
                selection.provenance.redshift,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise PowerSpectrumError(
                    f"{raw_path} REDSHFT={raw_redshift} differs from "
                    f"classification REDSHFT={selection.provenance.redshift}"
                )
            if raw_nreal != n_halos or raw_nrandpt != n_halos:
                raise PowerSpectrumError(
                    f"{raw_path} has NREAL={raw_nreal}, NRANDPT={raw_nrandpt}; "
                    f"readfof contains {n_halos} halos"
                )
            if not raw_periodic:
                raise PowerSpectrumError(
                    f"{raw_path} is not marked as a periodic catalogue"
                )
            expected_rows = n_halos * (raw_iterations + 1)
            if len(data) != expected_rows:
                raise PowerSpectrumError(
                    f"{raw_path} contains {len(data)} rows, expected {expected_rows}"
                )

            lower = np.array(
                [
                    _metadata_float(header, "BOXLOX"),
                    _metadata_float(header, "BOXLOY"),
                    _metadata_float(header, "BOXLOZ"),
                ],
                dtype=np.float64,
            )
            upper = np.array(
                [
                    _metadata_float(header, "BOXHIX"),
                    _metadata_float(header, "BOXHIY"),
                    _metadata_float(header, "BOXHIZ"),
                ],
                dtype=np.float64,
            )
            if not np.allclose(lower, 0.0, rtol=0.0, atol=1.0e-8):
                raise PowerSpectrumError(
                    f"{raw_path} lower box bounds are {lower.tolist()}, "
                    "expected [0, 0, 0]"
                )
            if not np.allclose(
                upper,
                box_size,
                rtol=0.0,
                atol=max(1.0e-8, box_size * 1.0e-10),
            ):
                raise PowerSpectrumError(
                    f"{raw_path} upper box bounds are {upper.tolist()}, "
                    f"expected [{box_size}, {box_size}, {box_size}]"
                )

            target_ids = selection.target_ids
            if (
                target_ids.ndim != 1
                or len(target_ids) != selection.n_objects
                or np.any(np.diff(target_ids) <= 0)
            ):
                raise PowerSpectrumError(
                    "random-void TARGETID union is not strictly increasing and unique"
                )
            row_indices = target_ids - 1
            if row_indices[0] < n_halos or row_indices[-1] >= len(data):
                raise PowerSpectrumError(
                    "random-void TARGETID values lie outside the random raw rows"
                )
            matched_ids = np.asarray(
                data["TARGETID"][row_indices], dtype=np.int64
            )
            if not np.array_equal(matched_ids, target_ids):
                raise PowerSpectrumError(
                    "raw TARGETID order is not 1-based contiguous; direct "
                    "classification-to-raw crossmatch failed"
                )
            matched_iterations = np.asarray(
                data["RANDITER"][row_indices], dtype=np.int64
            )
            expected_iterations = row_indices // n_halos - 1
            if not np.array_equal(matched_iterations, expected_iterations):
                raise PowerSpectrumError(
                    "raw RANDITER values do not agree with selected TARGETIDs"
                )

            positions = np.empty((len(target_ids), 3), dtype=np.float32)
            positions[:, 0] = np.asarray(
                data["XCART"][row_indices], dtype=np.float32
            )
            positions[:, 1] = np.asarray(
                data["YCART"][row_indices], dtype=np.float32
            )
            positions[:, 2] = np.asarray(
                data["ZCART"][row_indices], dtype=np.float32
            )

    if not np.all(np.isfinite(positions)):
        raise PowerSpectrumError(
            f"{raw_path} contains non-finite random-void coordinates"
        )
    tolerance = max(1.0e-5, box_size * 1.0e-6)
    if float(np.min(positions)) < -tolerance or float(np.max(positions)) > (
        box_size + tolerance
    ):
        raise PowerSpectrumError(
            f"{raw_path} random-void coordinates lie outside [0, {box_size}]"
        )
    np.mod(positions, np.float32(box_size), out=positions)
    provenance = replace(
        selection.provenance,
        raw_path=raw_path.resolve(),
        raw_sha256=raw_sha256,
    )
    return np.ascontiguousarray(positions), provenance


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
    tracer: str,
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
        "tracer": tracer,
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
        "astra_classification_sha256": (
            ""
            if astra_provenance is None
            else astra_provenance.classification_sha256
        ),
        "astra_classification_files": (
            ""
            if astra_provenance is None
            or not astra_provenance.classification_paths
            else len(astra_provenance.classification_paths)
        ),
        "astra_raw_sha256": (
            "" if astra_provenance is None else astra_provenance.raw_sha256
        ),
        "selection_rule": (
            RANDOM_VOID_SELECTION_RULE
            if sample == RANDOM_VOID_SAMPLE
            else ""
        ),
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


def read_power_spectrum_curve(
    path: Path,
    *,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    sample: str,
) -> PowerSpectrumCurve:
    """Read and validate the raw monopole columns needed by the plot."""

    required = set(PLOT_METADATA_COLUMNS) | {
        "sample",
        "n_k_shells",
        "k_h_Mpc",
        "Pk0_raw_Mpc3_h3",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing = sorted(required - fieldnames)
            if missing:
                raise PowerSpectrumError(
                    f"{path}: missing plot columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PowerSpectrumError(f"{path}: cannot read CSV for plot: {exc}") from exc
    if not rows:
        raise PowerSpectrumError(f"{path}: CSV contains no power-spectrum rows")

    first = rows[0]
    expected = {
        "dataset": dataset,
        "simulation_id": str(simulation_id),
        "snapshot": str(snapnum),
        "sample": sample,
    }
    for column, value in expected.items():
        if first.get(column) != value:
            raise PowerSpectrumError(
                f"{path}: {column}={first.get(column)!r}, expected {value!r}"
            )
    n_shells = _csv_int(first, "n_k_shells", path)
    if n_shells != len(rows):
        raise PowerSpectrumError(
            f"{path}: n_k_shells={n_shells}, but CSV contains {len(rows)} rows"
        )

    stable_columns = PLOT_METADATA_COLUMNS + ("sample", "n_k_shells")
    stable = {column: first[column] for column in stable_columns}
    k_values = np.empty(len(rows), dtype=np.float64)
    pk0_values = np.empty(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows, start=2):
        for column, value in stable.items():
            if row.get(column) != value:
                raise PowerSpectrumError(
                    f"{path}:{row_index}: {column} changes within the CSV"
                )
        k_values[row_index - 2] = _csv_float(row, "k_h_Mpc", path)
        pk0_values[row_index - 2] = _csv_float(
            row, "Pk0_raw_Mpc3_h3", path
        )
    if np.any(k_values <= 0.0) or np.any(np.diff(k_values) <= 0.0):
        raise PowerSpectrumError(
            f"{path}: k_h_Mpc must be positive and strictly increasing"
        )
    metadata = {column: first[column] for column in PLOT_METADATA_COLUMNS}
    return PowerSpectrumCurve(
        k=k_values,
        pk0_raw=pk0_values,
        metadata=metadata,
    )


def build_normalized_power_spectra(
    matter_path: Path,
    environment_paths: Mapping[str, Path],
    *,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    kmin: float,
    kmax: float,
) -> NormalizedPowerSpectra:
    """Build the notebook's five ``P0_raw(environment) / P0_raw(all)`` curves."""

    matter = read_power_spectrum_curve(
        matter_path,
        dataset=dataset,
        simulation_id=simulation_id,
        snapnum=snapnum,
        sample="all",
    )
    curves: dict[str, PowerSpectrumCurve] = {}
    source_paths = [matter_path]
    astra_metadata: Mapping[str, str] | None = None
    halo_probability_sha256: str | None = None
    for name, _, _ in NORMALIZED_PLOT_SAMPLES:
        try:
            path = environment_paths[name]
        except KeyError as exc:
            raise PowerSpectrumError(
                f"no CSV path was provided for plot sample {name!r}"
            ) from exc
        curve = read_power_spectrum_curve(
            path,
            dataset=dataset,
            simulation_id=simulation_id,
            snapnum=snapnum,
            sample=name,
        )
        for column in PLOT_SHARED_METADATA_COLUMNS:
            if curve.metadata[column] != matter.metadata[column]:
                raise PowerSpectrumError(
                    f"{path}: {column}={curve.metadata[column]!r} differs "
                    f"from all-halo CSV value {matter.metadata[column]!r}"
                )
        if astra_metadata is None:
            astra_metadata = {
                column: curve.metadata[column]
                for column in PLOT_ENVIRONMENT_METADATA_COLUMNS
            }
            missing_astra_metadata = [
                column
                for column, value in astra_metadata.items()
                if value == ""
            ]
            if missing_astra_metadata:
                raise PowerSpectrumError(
                    f"{path}: empty ASTRA plot metadata: "
                    f"{', '.join(missing_astra_metadata)}"
                )
        else:
            for column, value in astra_metadata.items():
                if curve.metadata[column] != value:
                    raise PowerSpectrumError(
                        f"{path}: {column}={curve.metadata[column]!r} differs "
                        f"from the other environment CSVs value {value!r}"
                    )
        if name != RANDOM_VOID_SAMPLE:
            probability_sha256 = curve.metadata[
                "astra_probability_sha256"
            ]
            if not probability_sha256:
                raise PowerSpectrumError(
                    f"{path}: astra_probability_sha256 is empty"
                )
            if halo_probability_sha256 is None:
                halo_probability_sha256 = probability_sha256
            elif probability_sha256 != halo_probability_sha256:
                raise PowerSpectrumError(
                    f"{path}: astra_probability_sha256 differs from the "
                    "other hard-environment CSVs"
                )
        if curve.k.shape != matter.k.shape or not np.allclose(
            curve.k,
            matter.k,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise PowerSpectrumError(
                f"{path}: k shells do not match {matter_path}; "
                "the normalized plot does not interpolate spectra"
            )
        curves[name] = curve
        source_paths.append(path)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = {
            name: curve.pk0_raw / matter.pk0_raw
            for name, curve in curves.items()
        }
    mask = (
        np.isfinite(matter.k)
        & np.isfinite(matter.pk0_raw)
        & (matter.pk0_raw > 0.0)
        & (matter.k >= kmin)
        & (matter.k <= kmax)
    )
    for ratio in ratios.values():
        mask &= np.isfinite(ratio) & (ratio > 0.0)
    if not np.any(mask):
        raise PowerSpectrumError(
            "no common positive finite shells remain for the normalized plot"
        )
    return NormalizedPowerSpectra(
        k=np.ascontiguousarray(matter.k[mask]),
        ratios={
            name: np.ascontiguousarray(ratio[mask])
            for name, ratio in ratios.items()
        },
        source_paths=tuple(path.resolve() for path in source_paths),
    )


def normalized_plot_needs_refresh(
    path: Path,
    source_paths: Sequence[Path],
    *,
    overwrite: bool,
) -> bool:
    """Return whether the PNG is missing, forced, or older than any source CSV."""

    if overwrite or not path.exists():
        return True
    if not path.is_file():
        raise PowerSpectrumError(f"{path}: normalized plot path is not a file")
    try:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return True
    except OSError:
        return True
    plot_mtime = path.stat().st_mtime_ns
    return any(source.stat().st_mtime_ns > plot_mtime for source in source_paths)


def draw_normalized_power_spectra(
    spectra: NormalizedPowerSpectra,
) -> tuple[Any, Any]:
    """Draw the normalized figure with the visual style of power_spec.ipynb."""

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PowerSpectrumError(
            "matplotlib is required to create the normalized power-spectrum plot"
        ) from exc

    with matplotlib.rc_context({"figure.dpi": 360.0, "text.usetex": False}):
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            for name, label, color in NORMALIZED_PLOT_SAMPLES:
                ax.plot(
                    spectra.k,
                    spectra.ratios[name],
                    ".-",
                    color=color,
                    label=label,
                )
            ax.axhline(
                1.0,
                color="white",
                linestyle="--",
                linewidth=1.2,
                label="All",
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
            ax.set_ylabel(
                r"$P_{0,\mathrm{env}}^{\mathrm{raw}}(k)"
                r"/P_{0,\mathrm{all}}^{\mathrm{raw}}(k)$"
            )
            ax.set_title(r"Normalized $P_0(k)$ by environment")
            ax.grid(linewidth=0.3)
            ax.legend()
            fig.tight_layout()
    return fig, ax


def write_normalized_power_spectra_plot(
    path: Path,
    spectra: NormalizedPowerSpectra,
) -> None:
    """Render the normalized PNG atomically for safe batch execution."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    fig = None
    plt = None
    try:
        fig, _ = draw_normalized_power_spectra(spectra)
        import matplotlib.pyplot as plt

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}.",
            suffix=".png",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
        fig.savefig(
            temporary_name,
            dpi=360,
            facecolor="black",
            edgecolor="none",
            bbox_inches="tight",
        )
        with Path(temporary_name).open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if fig is not None:
            if plt is None:
                try:
                    import matplotlib.pyplot as plt
                except ImportError:
                    plt = None
            if plt is not None:
                plt.close(fig)
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


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
    tracer: str,
    n_objects: int,
    fof_catalog_sha256: str,
    astra_provenance: AstraProvenance | None,
    args: argparse.Namespace,
) -> None:
    """Require an existing CSV to be complete and reproducibly compatible."""

    has_extended_columns = False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            has_extended_columns = reader.fieldnames == list(CSV_COLUMNS)
            is_legacy = reader.fieldnames == list(LEGACY_CSV_COLUMNS)
            if not has_extended_columns and not is_legacy:
                raise PowerSpectrumError(
                    f"{path}: columns do not match the current output schema"
                )
            if sample == RANDOM_VOID_SAMPLE and not has_extended_columns:
                raise PowerSpectrumError(
                    f"{path}: random-void products require classification/raw "
                    "provenance columns"
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
        "tracer": tracer,
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
    if has_extended_columns:
        expected_strings.update(
            {
                "astra_classification_sha256": (
                    ""
                    if astra_provenance is None
                    else astra_provenance.classification_sha256
                ),
                "astra_raw_sha256": (
                    ""
                    if astra_provenance is None
                    else astra_provenance.raw_sha256
                ),
                "selection_rule": (
                    RANDOM_VOID_SELECTION_RULE
                    if sample == RANDOM_VOID_SAMPLE
                    else ""
                ),
            }
        )
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
    if has_extended_columns:
        expected_optional_integers["astra_classification_files"] = (
            None
            if astra_provenance is None
            or not astra_provenance.classification_paths
            else len(astra_provenance.classification_paths)
        )
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
    tracer: str,
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
            tracer=tracer,
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
    tracer: str,
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
        tracer=tracer,
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


def maybe_write_normalized_plot(
    *,
    matter_path: Path,
    environment_paths: Mapping[str, Path],
    output_path: Path,
    dataset: str,
    simulation_id: int,
    snapnum: int,
    args: argparse.Namespace,
    counters: Counters,
) -> None:
    """Create the normalized PNG once all six compatible CSVs are present."""

    required_paths = [("all", matter_path)]
    required_paths.extend(
        (name, environment_paths[name])
        for name, _, _ in NORMALIZED_PLOT_SAMPLES
    )
    missing = [name for name, path in required_paths if not path.is_file()]
    if missing:
        _log(
            f"[plot] normalized P(k) skipped for {dataset} "
            f"sim={simulation_id}; missing CSVs: {', '.join(missing)}"
        )
        return

    spectra = build_normalized_power_spectra(
        matter_path,
        environment_paths,
        dataset=dataset,
        simulation_id=simulation_id,
        snapnum=snapnum,
        kmin=args.kmin,
        kmax=args.kmax,
    )
    if not normalized_plot_needs_refresh(
        output_path,
        spectra.source_paths,
        overwrite=args.overwrite,
    ):
        _log(f"[plot-skip] normalized plot is current: {output_path}")
        counters.plots_skipped += 1
        return
    start = time.perf_counter()
    write_normalized_power_spectra_plot(output_path, spectra)
    elapsed = time.perf_counter() - start
    _log(
        f"[plot-write] {output_path} "
        f"({len(spectra.k)} common k shells, {elapsed:.1f} s)"
    )
    counters.plots_written += 1


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
    plot_path = normalized_plot_path(
        args.output_root, dataset, simulation_id, args.snapnum
    )
    do_matter = not args.environments_only and not args.random_void_only
    do_halo_environments = not args.matter_only and not args.random_void_only
    do_random_void = (
        not args.matter_only
        and not args.skip_random_void
    )
    do_normalized_plot = (
        not args.matter_only
        and not args.skip_random_void
        and not args.skip_normalized_plot
    )

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
        if do_halo_environments:
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
        if do_random_void:
            try:
                classifications = resolve_classification_paths(
                    args.astra_root,
                    dataset,
                    simulation_id,
                    args.snapnum,
                )
                _log(
                    f"       classifications: {len(classifications)} files "
                    f"({classifications[0].parent})"
                )
            except FileNotFoundError as exc:
                _log(f"       classifications: MISSING ({exc})")
            try:
                raw_path = resolve_raw_path(
                    args.astra_root,
                    dataset,
                    simulation_id,
                    args.snapnum,
                )
                _log(f"       raw: {raw_path}")
            except FileNotFoundError as exc:
                _log(f"       raw: MISSING ({exc})")
            random_path = environment_paths[RANDOM_VOID_SAMPLE]
            state = "overwrite" if random_path.exists() else "write"
            if random_path.exists() and not args.overwrite:
                state = "validate-existing"
            _log(f"       {RANDOM_VOID_SAMPLE} ({state}): {random_path}")
        if do_normalized_plot:
            state = "overwrite" if args.overwrite else "write-or-refresh"
            _log(f"       normalized plot ({state}): {plot_path}")
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
            tracer="FoF_halo",
            n_objects=len(positions),
            fof_catalog_sha256=fof_catalog_sha256,
            astra_provenance=None,
            args=args,
            counters=counters,
        ):
            _compute_and_write(
                positions=positions,
                sample="all",
                tracer="FoF_halo",
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

    sample_errors: list[str] = []
    if do_halo_environments:
        try:
            probability_path = resolve_probability_path(
                args.astra_root,
                dataset,
                simulation_id,
                args.snapnum,
            )
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
                f"(sha256={assignment.provenance.probability_sha256[:12]}..., "
                f"{counts})"
            )
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
                        tracer="FoF_halo",
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
                            tracer="FoF_halo",
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
            del assignment
        except FileNotFoundError as exc:
            counters.missing_probabilities += 1
            if args.allow_missing_astra:
                _log(
                    f"[warning] ASTRA probabilities are not finished for "
                    f"{dataset} sim={simulation_id}; halo environments skipped"
                )
            else:
                message = (
                    f"{dataset} sim={simulation_id} snap={args.snapnum}: {exc}"
                )
                _error(f"[error] {message}")
                sample_errors.append(message)
                if args.fail_fast:
                    raise

    if do_random_void:
        try:
            classification_paths = resolve_classification_paths(
                args.astra_root,
                dataset,
                simulation_id,
                args.snapnum,
            )
            raw_path = resolve_raw_path(
                args.astra_root,
                dataset,
                simulation_id,
                args.snapnum,
            )
            selection = read_random_void_selection(
                classification_paths,
                len(positions),
                simulation_id,
                args.snapnum,
                args.box_size,
            )
            raw_sha256 = _sha256_files([raw_path])
            random_provenance = replace(
                selection.provenance,
                raw_path=raw_path.resolve(),
                raw_sha256=raw_sha256,
            )
            _log(
                f"[random-void] selected {selection.n_objects:,} unique "
                f"TARGETIDs from {len(classification_paths)} iterations "
                f"(class sha256="
                f"{selection.provenance.classification_sha256[:12]}...)"
            )
            output_path = environment_paths[RANDOM_VOID_SAMPLE]
            if output_needs_computation(
                output_path,
                dataset=dataset,
                simulation_id=simulation_id,
                snapnum=args.snapnum,
                sample=RANDOM_VOID_SAMPLE,
                tracer=RANDOM_VOID_TRACER,
                n_objects=selection.n_objects,
                fof_catalog_sha256=fof_catalog_sha256,
                astra_provenance=random_provenance,
                args=args,
                counters=counters,
            ):
                random_positions, random_provenance = (
                    read_random_void_positions_from_raw(
                        raw_path,
                        selection,
                        n_halos=len(positions),
                        simulation_id=simulation_id,
                        snapnum=args.snapnum,
                        box_size=args.box_size,
                        temp_parent=args.raw_temp_dir,
                        raw_sha256=raw_sha256,
                    )
                )
                try:
                    _compute_and_write(
                        positions=random_positions,
                        sample=RANDOM_VOID_SAMPLE,
                        tracer=RANDOM_VOID_TRACER,
                        output_path=output_path,
                        dataset=dataset,
                        simulation_id=simulation_id,
                        snapnum=args.snapnum,
                        fof_catalog_sha256=fof_catalog_sha256,
                        astra_provenance=random_provenance,
                        args=args,
                        mas_library=mas_library,
                        pk_library=pk_library,
                        counters=counters,
                    )
                finally:
                    del random_positions
                    gc.collect()
            del selection
        except FileNotFoundError as exc:
            counters.missing_classifications += 1
            if args.allow_missing_astra:
                _log(
                    f"[warning] ASTRA classification/raw products are not "
                    f"finished for {dataset} sim={simulation_id}; "
                    f"{RANDOM_VOID_SAMPLE} skipped"
                )
            else:
                message = (
                    f"{dataset} sim={simulation_id} snap={args.snapnum}: {exc}"
                )
                _error(f"[error] {message}")
                sample_errors.append(message)
                if args.fail_fast:
                    raise
        except Exception as exc:
            message = (
                f"{dataset} sim={simulation_id} snap={args.snapnum} "
                f"sample={RANDOM_VOID_SAMPLE}: {exc}"
            )
            _error(f"[error] {message}")
            sample_errors.append(message)
            if args.fail_fast:
                raise

    if do_normalized_plot:
        if sample_errors:
            _log(
                f"[plot] normalized P(k) skipped for {dataset} "
                f"sim={simulation_id} because a requested spectrum failed"
            )
        else:
            try:
                maybe_write_normalized_plot(
                    matter_path=matter_path,
                    environment_paths=environment_paths,
                    output_path=plot_path,
                    dataset=dataset,
                    simulation_id=simulation_id,
                    snapnum=args.snapnum,
                    args=args,
                    counters=counters,
                )
            except Exception as exc:
                message = (
                    f"{dataset} sim={simulation_id} snap={args.snapnum} "
                    f"normalized plot: {exc}"
                )
                _error(f"[error] {message}")
                sample_errors.append(message)
                if args.fail_fast:
                    raise

    del positions
    gc.collect()
    if sample_errors:
        raise PowerSpectrumError(
            f"{len(sample_errors)} requested sample group(s) failed; "
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
        args.astra_root if args.astra_root is not None else args.data_root
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
        f"plots_written={counters.plots_written}, "
        f"plots_skipped={counters.plots_skipped}, "
        f"missing_probabilities={counters.missing_probabilities}, "
        f"missing_classification_or_raw={counters.missing_classifications}, "
        f"failures={counters.failures}, elapsed={elapsed:.1f} s"
    )
    return 1 if counters.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
