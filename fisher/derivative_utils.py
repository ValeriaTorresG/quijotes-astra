"""Shared CSV discovery, validation, aggregation, and writing helpers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_PK_ROOT = Path.home() / "Desktop" / "Quijotes" / "data" / "pk"
DEFAULT_OBSERVABLE_COLUMNS = (
    "Pk0_raw_Mpc3_h3",
    "Pk0_shot_subtracted_Mpc3_h3",
    "Pk2_Mpc3_h3",
    "Pk4_Mpc3_h3",
)
KNOWN_SAMPLES = ("all", "void", "sheet", "filament", "knot", "random_void")

TEXT_COMPATIBILITY_COLUMNS = (
    "mass_assignment",
    "binning_mode",
    "astra_release",
    "astra_periodic",
    "selection_rule",
)
INTEGER_COMPATIBILITY_COLUMNS = (
    "grid",
    "los_axis",
    "threads",
    "n_astra_iterations",
    "astra_random_seed",
)
FLOAT_COMPATIBILITY_COLUMNS = (
    "redshift",
    "box_size_Mpc_h",
    "k_min_h_Mpc",
    "k_max_h_Mpc",
    "k_nyquist_h_Mpc",
    "k_fundamental_h_Mpc",
    "k_bin_width_h_Mpc",
    "astra_r_lower",
    "astra_r_med",
    "astra_r_upper",
    "astra_box_min_Mpc_h",
    "astra_box_max_Mpc_h",
)


class DerivativeError(RuntimeError):
    """Raised when spectra cannot be paired without ambiguity."""


@dataclass(frozen=True, order=True)
class ProductKey:
    """Identity of one spectrum, excluding its cosmological dataset."""

    category: str
    sample: str
    snapshot: int
    simulation_id: int

    def describe(self) -> str:
        return (
            f"{self.category}/{self.sample}, sim={self.simulation_id}, "
            f"snap={self.snapshot}"
        )


@dataclass
class SpectrumCSV:
    """Validated arrays and stable metadata read from one P(k) CSV."""

    path: Path
    dataset: str
    key: ProductKey
    tracer: str
    metadata: dict[str, str]
    k: np.ndarray
    nmodes: np.ndarray
    k_bin_index: np.ndarray | None
    k_bin_min: np.ndarray | None
    k_bin_max: np.ndarray | None
    values: dict[str, np.ndarray]


@dataclass
class DerivativeProduct:
    """Per-realization derivatives on one validated k grid."""

    key: ProductKey
    grid: SpectrumCSV
    derivatives: dict[str, np.ndarray]


def path_arg(value: str | os.PathLike[str]) -> Path:
    """Resolve a command-line filesystem path."""

    return Path(value).expanduser().resolve()


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def parse_simulation_ids(
    values: Sequence[str] | None,
    parser: argparse.ArgumentParser,
) -> set[int] | None:
    """Parse ``--simulation-ids`` while accepting the single token ``all``."""

    if values is None:
        return None
    lowered = [str(value).lower() for value in values]
    if "all" in lowered:
        if lowered != ["all"]:
            parser.error("'all' cannot be combined with explicit simulation IDs")
        return None
    try:
        parsed = [nonnegative_int(str(value)) for value in values]
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    return set(parsed)


def validate_observable_names(
    columns: Sequence[str],
    parser: argparse.ArgumentParser,
) -> tuple[str, ...]:
    """Reject duplicate or metadata columns requested as observables."""

    selected = tuple(columns)
    if not selected:
        parser.error("--columns needs at least one CSV column")
    if len(set(selected)) != len(selected):
        parser.error("--columns contains duplicate names")
    forbidden = {
        "dataset",
        "simulation_id",
        "snapshot",
        "sample",
        "tracer",
        "k_h_Mpc",
        "Nmodes",
    }
    overlap = forbidden.intersection(selected)
    if overlap:
        parser.error(
            "--columns must contain data-vector values, not metadata: "
            + ", ".join(sorted(overlap))
        )
    return selected


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the input, selection, output, and aggregation arguments."""

    parser.add_argument(
        "--pk-root",
        type=path_arg,
        default=DEFAULT_PK_ROOT,
        help="root containing the matter/ and env/ P(k) CSV directories",
    )
    parser.add_argument(
        "--output-root",
        type=path_arg,
        default=None,
        help="output root; defaults to PK_ROOT/fisher",
    )
    parser.add_argument(
        "--snapnum",
        "--snapshot",
        dest="snapnum",
        type=nonnegative_int,
        default=3,
        help="snapshot to process",
    )
    parser.add_argument(
        "--simulation-ids",
        "--simulation-id",
        nargs="+",
        default=None,
        metavar="ID",
        help="realization IDs to process; omit, or pass 'all', to discover them",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=KNOWN_SAMPLES,
        default=None,
        metavar="SAMPLE",
        help=(
            "samples to process; omit to use every discovered product "
            "('all' here means the all-halo matter sample)"
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_OBSERVABLE_COLUMNS),
        metavar="COLUMN",
        help="numeric data-vector columns to differentiate",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=(0, 1),
        default=1,
        help="degrees of freedom used for the across-realization dispersion",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "process only the exact product intersection when some datasets "
            "lack counterparts; by default any missing counterpart is an error"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing derivative and summary CSVs atomically",
    )


def finish_common_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Normalize shared parsed arguments."""

    args.output_root = (
        path_arg(args.output_root)
        if args.output_root is not None
        else args.pk_root / "fisher"
    )
    args.simulation_ids = parse_simulation_ids(args.simulation_ids, parser)
    args.samples = None if args.samples is None else set(args.samples)
    args.columns = validate_observable_names(args.columns, parser)


def _dataset_patterns(dataset: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(dataset)
    matter = re.compile(
        rf"^{escaped}_sim(?P<sim>\d+)_snap(?P<snap>\d+)_pk\.csv$"
    )
    environment = re.compile(
        rf"^{escaped}_sim(?P<sim>\d+)_snap(?P<snap>\d+)"
        rf"_(?P<sample>.+)_pk\.csv$"
    )
    return matter, environment


def discover_dataset_products(
    pk_root: Path,
    dataset: str,
    snapshot: int,
) -> dict[ProductKey, Path]:
    """Discover products using the exact filenames emitted by the P(k) script."""

    matter_pattern, environment_pattern = _dataset_patterns(dataset)
    found: dict[ProductKey, Path] = {}
    searches = (
        ("matter", pk_root / "matter", matter_pattern),
        ("env", pk_root / "env", environment_pattern),
    )
    for category, directory, pattern in searches:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(f"{dataset}_sim*_snap*_pk.csv")):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            file_snapshot = int(match.group("snap"))
            if file_snapshot != snapshot:
                continue
            sample = "all" if category == "matter" else match.group("sample")
            key = ProductKey(
                category=category,
                sample=sample,
                snapshot=file_snapshot,
                simulation_id=int(match.group("sim")),
            )
            if key in found:
                raise DerivativeError(
                    f"duplicate {dataset} spectrum for {key.describe()}: "
                    f"{found[key]} and {path}"
                )
            found[key] = path.resolve()
    if not found:
        raise DerivativeError(
            f"no {dataset} P(k) CSVs found for snapshot {snapshot} below "
            f"{pk_root / 'matter'} or {pk_root / 'env'}"
        )
    return found


def _filter_products(
    products: Mapping[ProductKey, Path],
    simulation_ids: set[int] | None,
    samples: set[str] | None,
) -> dict[ProductKey, Path]:
    return {
        key: path
        for key, path in products.items()
        if (
            simulation_ids is None or key.simulation_id in simulation_ids
        )
        and (samples is None or key.sample in samples)
    }


def discover_matched_products(
    pk_root: Path,
    datasets: Sequence[str],
    snapshot: int,
    *,
    simulation_ids: set[int] | None,
    samples: set[str] | None,
    allow_incomplete: bool,
) -> tuple[dict[str, dict[ProductKey, Path]], list[ProductKey], list[str]]:
    """Return only keys shared by every required cosmological dataset."""

    maps = {
        dataset: _filter_products(
            discover_dataset_products(pk_root, dataset, snapshot),
            simulation_ids,
            samples,
        )
        for dataset in datasets
    }
    sets = {dataset: set(products) for dataset, products in maps.items()}
    union = set().union(*sets.values())
    common = set.intersection(*sets.values()) if sets else set()

    if simulation_ids is not None:
        seen_ids = {key.simulation_id for key in union}
        missing_ids = simulation_ids - seen_ids
        if missing_ids:
            raise DerivativeError(
                "requested simulation IDs not found in the selected products: "
                + ", ".join(str(value) for value in sorted(missing_ids))
            )
    if samples is not None:
        seen_samples = {key.sample for key in union}
        missing_samples = samples - seen_samples
        if missing_samples:
            raise DerivativeError(
                "requested samples not found in the selected products: "
                + ", ".join(sorted(missing_samples))
            )
    if not union:
        raise DerivativeError(
            f"no selected P(k) products remain for snapshot {snapshot}"
        )

    incomplete = union - common
    warnings: list[str] = []
    if incomplete and not allow_incomplete:
        details = []
        for key in sorted(incomplete):
            missing = [dataset for dataset in datasets if key not in sets[dataset]]
            details.append(
                f"{key.describe()} missing [{', '.join(missing)}]"
            )
        preview = "; ".join(details[:12])
        if len(details) > 12:
            preview += f"; ... and {len(details) - 12} more"
        raise DerivativeError(
            "cosmological datasets do not have identical paired products; "
            + preview
            + ". Use --simulation-ids/--samples to select a complete subset, "
            "or --allow-incomplete to process only the intersection."
        )
    if incomplete:
        warnings.append(
            f"skipping {len(incomplete)} unmatched product(s); "
            f"processing the {len(common)}-product intersection"
        )
    if not common:
        raise DerivativeError(
            "the required cosmological datasets have no matched products"
        )
    return maps, sorted(common), warnings


def _constant_value(
    rows: Sequence[dict[str, str]],
    column: str,
    path: Path,
) -> str:
    values = [row.get(column) for row in rows]
    if any(value is None for value in values):
        raise DerivativeError(
            f"{path}: one or more rows are missing a value for {column!r}"
        )
    stripped = [str(value).strip() for value in values]
    first = stripped[0]
    if any(value != first for value in stripped[1:]):
        raise DerivativeError(
            f"{path}: metadata column {column!r} changes between k rows"
        )
    return first


def _parse_int(value: str, *, path: Path, column: str, row: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DerivativeError(
            f"{path}: invalid integer in {column!r}, row {row}: {value!r}"
        ) from exc
    return parsed


def _parse_float(value: str, *, path: Path, column: str, row: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DerivativeError(
            f"{path}: invalid float in {column!r}, row {row}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise DerivativeError(
            f"{path}: non-finite value in {column!r}, row {row}: {value!r}"
        )
    return parsed


def _optional_float_array(
    rows: Sequence[dict[str, str]],
    column: str,
    path: Path,
) -> np.ndarray | None:
    if column not in rows[0]:
        return None
    values = [
        "" if row.get(column) is None else str(row[column]).strip()
        for row in rows
    ]
    if all(value == "" for value in values):
        return None
    if any(value == "" for value in values):
        raise DerivativeError(
            f"{path}: {column!r} is only populated for some k rows"
        )
    return np.asarray(
        [
            _parse_float(value, path=path, column=column, row=index + 2)
            for index, value in enumerate(values)
        ],
        dtype=np.float64,
    )


def _optional_int_array(
    rows: Sequence[dict[str, str]],
    column: str,
    path: Path,
) -> np.ndarray | None:
    if column not in rows[0]:
        return None
    values = [
        "" if row.get(column) is None else str(row[column]).strip()
        for row in rows
    ]
    if all(value == "" for value in values):
        return None
    if any(value == "" for value in values):
        raise DerivativeError(
            f"{path}: {column!r} is only populated for some k rows"
        )
    return np.asarray(
        [
            _parse_int(value, path=path, column=column, row=index + 2)
            for index, value in enumerate(values)
        ],
        dtype=np.int64,
    )


def read_spectrum_csv(
    path: Path,
    *,
    expected_dataset: str,
    expected_key: ProductKey,
    columns: Sequence[str],
) -> SpectrumCSV:
    """Read one spectrum and verify filename identity against CSV metadata."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise DerivativeError(f"{path}: CSV header is missing")
            required = {
                "dataset",
                "simulation_id",
                "snapshot",
                "sample",
                "tracer",
                "n_k_shells",
                "k_h_Mpc",
                "Nmodes",
                *columns,
            }
            missing = required - set(fieldnames)
            if missing:
                raise DerivativeError(
                    f"{path}: missing required columns: "
                    + ", ".join(sorted(missing))
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise DerivativeError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise DerivativeError(f"{path}: spectrum CSV has no k rows")

    constant_columns = {
        "dataset",
        "simulation_id",
        "snapshot",
        "sample",
        "tracer",
        "n_k_shells",
        *TEXT_COMPATIBILITY_COLUMNS,
        *INTEGER_COMPATIBILITY_COLUMNS,
        *FLOAT_COMPATIBILITY_COLUMNS,
    }
    metadata = {
        column: _constant_value(rows, column, path)
        for column in constant_columns
        if column in rows[0]
    }
    dataset = metadata["dataset"]
    if dataset != expected_dataset:
        raise DerivativeError(
            f"{path}: dataset={dataset!r}, expected {expected_dataset!r}"
        )
    simulation_id = _parse_int(
        metadata["simulation_id"],
        path=path,
        column="simulation_id",
        row=2,
    )
    snapshot = _parse_int(
        metadata["snapshot"],
        path=path,
        column="snapshot",
        row=2,
    )
    sample = metadata["sample"]
    if (
        simulation_id != expected_key.simulation_id
        or snapshot != expected_key.snapshot
        or sample != expected_key.sample
    ):
        raise DerivativeError(
            f"{path}: CSV identity is sample={sample}, sim={simulation_id}, "
            f"snap={snapshot}; filename selection expected "
            f"{expected_key.describe()}"
        )
    expected_shells = _parse_int(
        metadata["n_k_shells"],
        path=path,
        column="n_k_shells",
        row=2,
    )
    if expected_shells != len(rows):
        raise DerivativeError(
            f"{path}: n_k_shells={expected_shells}, but CSV has {len(rows)} rows"
        )

    k = np.asarray(
        [
            _parse_float(
                row["k_h_Mpc"],
                path=path,
                column="k_h_Mpc",
                row=index + 2,
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.float64,
    )
    if np.any(np.diff(k) <= 0.0):
        raise DerivativeError(f"{path}: k_h_Mpc must be strictly increasing")
    nmodes = np.asarray(
        [
            _parse_int(
                row["Nmodes"],
                path=path,
                column="Nmodes",
                row=index + 2,
            )
            for index, row in enumerate(rows)
        ],
        dtype=np.int64,
    )
    if np.any(nmodes <= 0):
        raise DerivativeError(f"{path}: Nmodes must be positive")
    values = {
        column: np.asarray(
            [
                _parse_float(
                    row[column],
                    path=path,
                    column=column,
                    row=index + 2,
                )
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        for column in columns
    }
    return SpectrumCSV(
        path=path,
        dataset=dataset,
        key=expected_key,
        tracer=metadata["tracer"],
        metadata=metadata,
        k=k,
        nmodes=nmodes,
        k_bin_index=_optional_int_array(rows, "k_bin_index", path),
        k_bin_min=_optional_float_array(rows, "k_bin_min_h_Mpc", path),
        k_bin_max=_optional_float_array(rows, "k_bin_max_h_Mpc", path),
        values=values,
    )


def _optional_metadata_equal(
    reference: SpectrumCSV,
    candidate: SpectrumCSV,
    column: str,
    *,
    numeric: str | None = None,
) -> bool:
    left = reference.metadata.get(column, "").strip()
    right = candidate.metadata.get(column, "").strip()
    if left == "" and right == "":
        return True
    if (left == "") != (right == ""):
        return False
    if numeric is None:
        return left == right
    try:
        if numeric == "int":
            return int(left) == int(right)
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    except ValueError:
        return False


def _arrays_match(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    exact: bool,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if left.shape != right.shape:
        return False
    if exact:
        return bool(np.array_equal(left, right))
    return bool(
        np.allclose(left, right, rtol=1.0e-10, atol=1.0e-12)
    )


def validate_compatible_spectra(
    reference: SpectrumCSV,
    candidate: SpectrumCSV,
    *,
    same_realization: bool,
) -> None:
    """Require the same data-vector identity and k grid without interpolation."""

    if (
        reference.key.category != candidate.key.category
        or reference.key.sample != candidate.key.sample
        or reference.key.snapshot != candidate.key.snapshot
        or (
            same_realization
            and reference.key.simulation_id != candidate.key.simulation_id
        )
    ):
        raise DerivativeError(
            f"incompatible spectrum identities: {reference.path} and "
            f"{candidate.path}"
        )
    if reference.tracer != candidate.tracer:
        raise DerivativeError(
            f"tracer mismatch between {reference.path} and {candidate.path}"
        )
    if reference.k.shape != candidate.k.shape or not np.allclose(
        reference.k,
        candidate.k,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise DerivativeError(
            f"k grid mismatch between {reference.path} and {candidate.path}; "
            "spectra are never interpolated implicitly"
        )
    if not np.array_equal(reference.nmodes, candidate.nmodes):
        raise DerivativeError(
            f"Nmodes mismatch between {reference.path} and {candidate.path}"
        )
    optional_arrays = (
        ("k_bin_index", reference.k_bin_index, candidate.k_bin_index, True),
        ("k_bin_min", reference.k_bin_min, candidate.k_bin_min, False),
        ("k_bin_max", reference.k_bin_max, candidate.k_bin_max, False),
    )
    for label, left, right, exact in optional_arrays:
        if not _arrays_match(left, right, exact=exact):
            raise DerivativeError(
                f"{label} mismatch between {reference.path} and {candidate.path}"
            )
    for column in TEXT_COMPATIBILITY_COLUMNS:
        if not _optional_metadata_equal(reference, candidate, column):
            raise DerivativeError(
                f"{column} mismatch between {reference.path} and "
                f"{candidate.path}"
            )
    for column in INTEGER_COMPATIBILITY_COLUMNS:
        if not _optional_metadata_equal(
            reference, candidate, column, numeric="int"
        ):
            raise DerivativeError(
                f"{column} mismatch between {reference.path} and "
                f"{candidate.path}"
            )
    for column in FLOAT_COMPATIBILITY_COLUMNS:
        if not _optional_metadata_equal(
            reference, candidate, column, numeric="float"
        ):
            raise DerivativeError(
                f"{column} mismatch between {reference.path} and "
                f"{candidate.path}"
            )


def derivative_output_path(
    output_root: Path,
    scheme: str,
    key: ProductKey,
) -> Path:
    """Return one per-realization derivative path."""

    suffix = "" if key.category == "matter" else f"_{key.sample}"
    return (
        output_root
        / scheme
        / key.category
        / (
            f"sim{key.simulation_id:03d}_snap{key.snapshot:03d}"
            f"{suffix}_derivatives.csv"
        )
    )


def summary_output_path(
    output_root: Path,
    scheme: str,
    key: ProductKey,
) -> Path:
    """Return the mean-and-dispersion path for a product family."""

    suffix = "" if key.category == "matter" else f"_{key.sample}"
    return (
        output_root
        / scheme
        / key.category
        / f"snap{key.snapshot:03d}{suffix}_derivatives_mean_std.csv"
    )


def grid_fieldnames() -> list[str]:
    return [
        "simulation_id",
        "snapshot",
        "sample",
        "tracer",
        "redshift",
        "k_h_Mpc",
        "Nmodes",
        "binning_mode",
        "k_bin_width_h_Mpc",
        "k_bin_index",
        "k_bin_min_h_Mpc",
        "k_bin_max_h_Mpc",
    ]


def _format_float(value: float) -> str:
    return f"{float(value):.12e}"


def _optional_array_value(
    values: np.ndarray | None,
    index: int,
    *,
    integer: bool,
) -> str | int:
    if values is None:
        return ""
    return int(values[index]) if integer else _format_float(values[index])


def grid_row(grid: SpectrumCSV, index: int) -> dict[str, str | int]:
    """Build common output columns for one k row."""

    return {
        "simulation_id": grid.key.simulation_id,
        "snapshot": grid.key.snapshot,
        "sample": grid.key.sample,
        "tracer": grid.tracer,
        "redshift": grid.metadata.get("redshift", ""),
        "k_h_Mpc": _format_float(grid.k[index]),
        "Nmodes": int(grid.nmodes[index]),
        "binning_mode": grid.metadata.get("binning_mode", ""),
        "k_bin_width_h_Mpc": grid.metadata.get("k_bin_width_h_Mpc", ""),
        "k_bin_index": _optional_array_value(
            grid.k_bin_index, index, integer=True
        ),
        "k_bin_min_h_Mpc": _optional_array_value(
            grid.k_bin_min, index, integer=False
        ),
        "k_bin_max_h_Mpc": _optional_array_value(
            grid.k_bin_max, index, integer=False
        ),
    }


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    overwrite: bool,
) -> None:
    """Write a CSV atomically."""

    if path.exists() and not overwrite:
        raise DerivativeError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def check_output_collisions(paths: Iterable[Path], overwrite: bool) -> None:
    """Fail before computation so a rerun cannot leave a partial new batch."""

    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        preview = ", ".join(str(path) for path in existing[:5])
        if len(existing) > 5:
            preview += f", ... and {len(existing) - 5} more"
        raise DerivativeError(
            f"{len(existing)} output(s) already exist: {preview}; "
            "pass --overwrite to replace them"
        )


def _summary_groups(
    products: Sequence[DerivativeProduct],
) -> dict[tuple[str, str, int], list[DerivativeProduct]]:
    groups: dict[tuple[str, str, int], list[DerivativeProduct]] = {}
    for product in products:
        group_key = (
            product.key.category,
            product.key.sample,
            product.key.snapshot,
        )
        groups.setdefault(group_key, []).append(product)
    return groups


def build_summary_rows(
    products: Sequence[DerivativeProduct],
    *,
    derivative_columns: Sequence[str],
    ddof: int,
    common_metadata: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    """Aggregate per-realization arrays only after derivatives are formed."""

    if not products:
        raise DerivativeError("cannot summarize an empty derivative collection")
    ordered = sorted(products, key=lambda item: item.key.simulation_id)
    reference = ordered[0].grid
    for product in ordered[1:]:
        validate_compatible_spectra(
            reference,
            product.grid,
            same_realization=False,
        )
    n_realizations = len(ordered)
    realization_ids = ",".join(
        str(product.key.simulation_id) for product in ordered
    )
    means: dict[str, np.ndarray] = {}
    dispersions: dict[str, np.ndarray] = {}
    for column in derivative_columns:
        stacked = np.stack(
            [product.derivatives[column] for product in ordered],
            axis=0,
        )
        means[column] = np.mean(stacked, axis=0)
        if n_realizations <= ddof:
            dispersions[column] = np.full(
                reference.k.shape,
                np.nan,
                dtype=np.float64,
            )
        else:
            dispersions[column] = np.std(stacked, axis=0, ddof=ddof)

    fields = [
        "snapshot",
        "sample",
        "tracer",
        "redshift",
        "n_realizations",
        "realization_ids",
        "dispersion_ddof",
        "k_h_Mpc",
        "Nmodes",
        "binning_mode",
        "k_bin_width_h_Mpc",
        "k_bin_index",
        "k_bin_min_h_Mpc",
        "k_bin_max_h_Mpc",
        *common_metadata.keys(),
    ]
    for column in derivative_columns:
        fields.extend((f"mean_{column}", f"std_{column}"))

    rows: list[dict[str, object]] = []
    for index in range(len(reference.k)):
        base = grid_row(reference, index)
        base.pop("simulation_id")
        row: dict[str, object] = {
            **base,
            "n_realizations": n_realizations,
            "realization_ids": realization_ids,
            "dispersion_ddof": ddof,
            **common_metadata,
        }
        for column in derivative_columns:
            row[f"mean_{column}"] = _format_float(means[column][index])
            dispersion = dispersions[column][index]
            row[f"std_{column}"] = (
                "" if np.isnan(dispersion) else _format_float(dispersion)
            )
        rows.append(row)
    return fields, rows


def group_products_for_summary(
    products: Sequence[DerivativeProduct],
) -> list[list[DerivativeProduct]]:
    """Return deterministic category/sample groups."""

    groups = _summary_groups(products)
    return [groups[key] for key in sorted(groups)]
