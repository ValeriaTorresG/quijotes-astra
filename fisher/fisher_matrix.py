#!/usr/bin/env python3
"""Build Fisher matrices from mean derivatives and fiducial covariances."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np

if __package__:
    from .derivative_utils import (
        DEFAULT_PK_ROOT,
        COMBINED_CATEGORY,
        COMBINED_SAMPLE,
        DerivativeError,
        KNOWN_SAMPLES,
        atomic_write_csv,
        check_output_collisions,
        nonnegative_int,
        path_arg,
    )
else:
    from derivative_utils import (  # type: ignore[no-redef]
        DEFAULT_PK_ROOT,
        COMBINED_CATEGORY,
        COMBINED_SAMPLE,
        DerivativeError,
        KNOWN_SAMPLES,
        atomic_write_csv,
        check_output_collisions,
        nonnegative_int,
        path_arg,
    )


DEFAULT_FISHER_ROOT = DEFAULT_PK_ROOT / "fisher"
DEFAULT_PARAMETERS = ("Om", "h", "ns", "s8", "Mnu")
CENTRAL_PARAMETERS = frozenset(("Om", "h", "ns", "s8"))
DATASET = "fiducial"
FISHER_SAMPLES = (*KNOWN_SAMPLES, COMBINED_SAMPLE)


@dataclass(frozen=True, order=True)
class FisherKey:
    """Identity shared by covariance and derivative summary products."""

    category: str
    sample: str
    snapshot: int


@dataclass(frozen=True)
class DataVectorComponent:
    """One labelled row/column of the covariance matrix."""

    label: str
    position: int
    observable: str
    k_index: int
    k: float
    nmodes: int
    tracer: str
    binning_mode: str
    k_bin_width: str
    k_bin_index: str
    k_bin_min: str
    k_bin_max: str


@dataclass
class CovarianceData:
    """Dense covariance plus its exact component mapping."""

    key: FisherKey
    components: list[DataVectorComponent]
    matrix: np.ndarray
    n_realizations: int
    ddof: int
    realization_ids: list[int]

    @property
    def size(self) -> int:
        return len(self.components)


@dataclass
class DerivativeSummary:
    """Mean derivatives and grid metadata from one summary CSV."""

    path: Path
    key: FisherKey
    tracer: str
    n_realizations: int
    k: np.ndarray
    nmodes: np.ndarray
    binning_mode: str
    k_bin_width: str
    k_bin_index: list[str]
    k_bin_min: list[str]
    k_bin_max: list[str]
    values: dict[str, np.ndarray]


@dataclass
class FisherResult:
    """All numerical products for one matter/environment sample."""

    key: FisherKey
    covariance: CovarianceData
    parameters: tuple[str, ...]
    derivative_matrix: np.ndarray
    fisher_matrix: np.ndarray
    precision_factor: float
    covariance_rank: int
    covariance_condition: float
    correlation_condition: float
    covariance_eigen_min: float
    covariance_eigen_max: float
    fisher_rank: int
    fisher_condition: float
    central_realizations: int | None
    mnu_realizations: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Fisher matrix per matter/environment sample from the "
            "mean paired derivatives and the fiducial covariance products."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fisher-root",
        "--input-root",
        dest="fisher_root",
        type=path_arg,
        default=DEFAULT_FISHER_ROOT,
        help="root containing central/, forward/, and covariance/",
    )
    parser.add_argument(
        "--output-root",
        type=path_arg,
        default=None,
        help="output root; defaults to FISHER_ROOT/matrices",
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
        "--samples",
        nargs="+",
        choices=FISHER_SAMPLES,
        default=None,
        metavar="SAMPLE",
        help=(
            "samples to process; omit to use every discovered covariance "
            "('all' means the all-halo matter sample)"
        ),
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=DEFAULT_PARAMETERS,
        default=list(DEFAULT_PARAMETERS),
        metavar="PARAM",
        help="ordered parameter list used for Fisher rows and columns",
    )
    parser.add_argument(
        "--mnu-order",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="forward-difference accuracy order used for the Mnu derivative",
    )
    parser.add_argument(
        "--precision-correction",
        choices=("hartlap", "none"),
        default="none",
        help=(
            "optional finite-simulation correction applied to the inverse "
            "covariance"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing Fisher products atomically",
    )
    return parser


def finish_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    args.output_root = (
        path_arg(args.output_root)
        if args.output_root is not None
        else args.fisher_root / "matrices"
    )
    args.samples = None if args.samples is None else set(args.samples)
    if len(set(args.parameters)) != len(args.parameters):
        parser.error("--parameters contains duplicate names")
    args.parameters = tuple(args.parameters)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DerivativeError(f"{path}: CSV header is missing")
            fieldnames = list(reader.fieldnames)
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise DerivativeError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise DerivativeError(f"{path}: CSV has no data rows")
    return fieldnames, rows


def _require_columns(
    path: Path,
    fieldnames: Sequence[str],
    required: Sequence[str] | set[str],
) -> None:
    missing = set(required) - set(fieldnames)
    if missing:
        raise DerivativeError(
            f"{path}: missing required columns: "
            + ", ".join(sorted(missing))
        )


def _parse_int(value: str, path: Path, column: str, row: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DerivativeError(
            f"{path}: invalid integer in {column!r}, row {row}: {value!r}"
        ) from exc


def _parse_float(value: str, path: Path, column: str, row: int) -> float:
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


def _constant(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    column: str,
) -> str:
    values = [str(row.get(column, "")).strip() for row in rows]
    if any(value != values[0] for value in values[1:]):
        raise DerivativeError(
            f"{path}: {column!r} changes between rows"
        )
    return values[0]


def _covariance_stem(key: FisherKey) -> str:
    suffix = "" if key.category == "matter" else f"_{key.sample}"
    return f"{DATASET}_snap{key.snapshot:03d}{suffix}"


def covariance_paths(
    fisher_root: Path,
    key: FisherKey,
) -> tuple[Path, Path, Path]:
    stem = _covariance_stem(key)
    directory = fisher_root / "covariance" / key.category
    return (
        directory / f"{stem}_data_vector.csv",
        directory / f"{stem}_covariance.csv",
        directory / f"{stem}_realizations.csv",
    )


def derivative_summary_path(
    fisher_root: Path,
    scheme: str,
    key: FisherKey,
) -> Path:
    suffix = "" if key.category == "matter" else f"_{key.sample}"
    return (
        fisher_root
        / scheme
        / key.category
        / (
            f"snap{key.snapshot:03d}{suffix}"
            "_derivatives_mean_std.csv"
        )
    )


def discover_covariance_keys(
    fisher_root: Path,
    snapshot: int,
    samples: set[str] | None,
) -> list[FisherKey]:
    keys: list[FisherKey] = []
    matter_name = f"{DATASET}_snap{snapshot:03d}_data_vector.csv"
    matter_path = fisher_root / "covariance" / "matter" / matter_name
    if matter_path.is_file() and (samples is None or "all" in samples):
        keys.append(FisherKey("matter", "all", snapshot))

    environment_directory = fisher_root / "covariance" / "env"
    pattern = re.compile(
        rf"^{DATASET}_snap{snapshot:03d}_(?P<sample>.+)_data_vector\.csv$"
    )
    if environment_directory.is_dir():
        for path in sorted(environment_directory.glob(
            f"{DATASET}_snap{snapshot:03d}_*_data_vector.csv"
        )):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            sample = match.group("sample")
            if samples is None or sample in samples:
                keys.append(FisherKey("env", sample, snapshot))
    combined_name = (
        f"{DATASET}_snap{snapshot:03d}_{COMBINED_SAMPLE}_data_vector.csv"
    )
    combined_path = (
        fisher_root
        / "covariance"
        / COMBINED_CATEGORY
        / combined_name
    )
    if combined_path.is_file() and (
        samples is None or COMBINED_SAMPLE in samples
    ):
        keys.append(
            FisherKey(COMBINED_CATEGORY, COMBINED_SAMPLE, snapshot)
        )
    if samples is not None:
        seen = {key.sample for key in keys}
        missing = samples - seen
        if missing:
            raise DerivativeError(
                "requested covariance samples not found: "
                + ", ".join(sorted(missing))
            )
    if not keys:
        raise DerivativeError(
            f"no covariance data-vector products found for snapshot {snapshot} "
            f"below {fisher_root / 'covariance'}"
        )
    return sorted(keys)


def _parse_data_vector(
    path: Path,
    key: FisherKey,
) -> tuple[list[DataVectorComponent], int, int]:
    fieldnames, rows = _read_csv(path)
    required = {
        "data_vector_index",
        "vector_position",
        "dataset",
        "n_realizations",
        "snapshot",
        "category",
        "sample",
        "tracer",
        "data_vector_order",
        "observable",
        "k_index",
        "k_h_Mpc",
        "Nmodes",
        "binning_mode",
        "k_bin_width_h_Mpc",
        "k_bin_index",
        "k_bin_min_h_Mpc",
        "k_bin_max_h_Mpc",
        "covariance_ddof",
        "data_vector_size",
    }
    _require_columns(path, fieldnames, required)
    if _constant(path, rows, "dataset") != DATASET:
        raise DerivativeError(f"{path}: covariance dataset must be fiducial")
    if _constant(path, rows, "category") != key.category:
        raise DerivativeError(f"{path}: category does not match its path")
    if _constant(path, rows, "sample") != key.sample:
        raise DerivativeError(f"{path}: sample does not match its path")
    if _constant(path, rows, "data_vector_order") != "observable_then_k":
        raise DerivativeError(
            f"{path}: unsupported data-vector order; expected observable_then_k"
        )
    snapshot = _parse_int(
        _constant(path, rows, "snapshot"),
        path,
        "snapshot",
        2,
    )
    if snapshot != key.snapshot:
        raise DerivativeError(
            f"{path}: snapshot={snapshot}, expected {key.snapshot}"
        )
    n_realizations = _parse_int(
        _constant(path, rows, "n_realizations"),
        path,
        "n_realizations",
        2,
    )
    ddof = _parse_int(
        _constant(path, rows, "covariance_ddof"),
        path,
        "covariance_ddof",
        2,
    )
    declared_size = _parse_int(
        _constant(path, rows, "data_vector_size"),
        path,
        "data_vector_size",
        2,
    )
    if declared_size != len(rows):
        raise DerivativeError(
            f"{path}: data_vector_size={declared_size}, but CSV has "
            f"{len(rows)} rows"
        )

    components: list[DataVectorComponent] = []
    for index, row in enumerate(rows):
        csv_row = index + 2
        label = row["data_vector_index"].strip()
        expected_label = f"v{index:06d}"
        position = _parse_int(
            row["vector_position"],
            path,
            "vector_position",
            csv_row,
        )
        if label != expected_label or position != index:
            raise DerivativeError(
                f"{path}: row {csv_row} has index {label}/{position}; "
                f"expected {expected_label}/{index}"
            )
        components.append(
            DataVectorComponent(
                label=label,
                position=position,
                observable=row["observable"].strip(),
                k_index=_parse_int(
                    row["k_index"],
                    path,
                    "k_index",
                    csv_row,
                ),
                k=_parse_float(
                    row["k_h_Mpc"],
                    path,
                    "k_h_Mpc",
                    csv_row,
                ),
                nmodes=_parse_int(
                    row["Nmodes"],
                    path,
                    "Nmodes",
                    csv_row,
                ),
                tracer=row["tracer"].strip(),
                binning_mode=row["binning_mode"].strip(),
                k_bin_width=row["k_bin_width_h_Mpc"].strip(),
                k_bin_index=row["k_bin_index"].strip(),
                k_bin_min=row["k_bin_min_h_Mpc"].strip(),
                k_bin_max=row["k_bin_max_h_Mpc"].strip(),
            )
        )
    if any(not component.observable for component in components):
        raise DerivativeError(f"{path}: observable names cannot be empty")
    return components, n_realizations, ddof


def _parse_covariance_matrix(
    path: Path,
    components: Sequence[DataVectorComponent],
    expected_n: int,
    expected_ddof: int,
) -> np.ndarray:
    fieldnames, rows = _read_csv(path)
    labels = [component.label for component in components]
    required = {
        "data_vector_index",
        "n_realizations",
        "covariance_ddof",
        *labels,
    }
    _require_columns(path, fieldnames, required)
    if len(rows) != len(labels):
        raise DerivativeError(
            f"{path}: covariance has {len(rows)} rows, expected {len(labels)}"
        )
    n_realizations = _parse_int(
        _constant(path, rows, "n_realizations"),
        path,
        "n_realizations",
        2,
    )
    ddof = _parse_int(
        _constant(path, rows, "covariance_ddof"),
        path,
        "covariance_ddof",
        2,
    )
    if n_realizations != expected_n or ddof != expected_ddof:
        raise DerivativeError(
            f"{path}: covariance metadata N={n_realizations}, ddof={ddof}; "
            f"mapping has N={expected_n}, ddof={expected_ddof}"
        )
    matrix = np.empty((len(labels), len(labels)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        if row["data_vector_index"].strip() != labels[row_index]:
            raise DerivativeError(
                f"{path}: covariance row {row_index + 2} is not "
                f"{labels[row_index]}"
            )
        for column_index, label in enumerate(labels):
            matrix[row_index, column_index] = _parse_float(
                row[label],
                path,
                label,
                row_index + 2,
            )
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(
        matrix,
        matrix.T,
        rtol=1.0e-11,
        atol=1.0e-13 * scale,
    ):
        raise DerivativeError(f"{path}: covariance matrix is not symmetric")
    return 0.5 * (matrix + matrix.T)


def _parse_realizations(
    path: Path,
    key: FisherKey,
    expected_n: int,
) -> list[int]:
    fieldnames, rows = _read_csv(path)
    required = {
        "realization_order",
        "simulation_id",
        "dataset",
        "snapshot",
        "category",
        "sample",
    }
    _require_columns(path, fieldnames, required)
    if len(rows) != expected_n:
        raise DerivativeError(
            f"{path}: has {len(rows)} realizations, expected {expected_n}"
        )
    ids = []
    for index, row in enumerate(rows):
        csv_row = index + 2
        order = _parse_int(
            row["realization_order"],
            path,
            "realization_order",
            csv_row,
        )
        if order != index:
            raise DerivativeError(
                f"{path}: realization order {order}, expected {index}"
            )
        if (
            row["dataset"].strip() != DATASET
            or row["category"].strip() != key.category
            or row["sample"].strip() != key.sample
            or _parse_int(
                row["snapshot"],
                path,
                "snapshot",
                csv_row,
            )
            != key.snapshot
        ):
            raise DerivativeError(
                f"{path}: realization row {csv_row} has incompatible identity"
            )
        ids.append(
            _parse_int(
                row["simulation_id"],
                path,
                "simulation_id",
                csv_row,
            )
        )
    if len(set(ids)) != len(ids):
        raise DerivativeError(f"{path}: duplicate simulation IDs")
    return ids


def read_covariance(
    fisher_root: Path,
    key: FisherKey,
) -> CovarianceData:
    vector_path, matrix_path, realizations_path = covariance_paths(
        fisher_root,
        key,
    )
    components, n_realizations, ddof = _parse_data_vector(
        vector_path,
        key,
    )
    matrix = _parse_covariance_matrix(
        matrix_path,
        components,
        n_realizations,
        ddof,
    )
    realization_ids = _parse_realizations(
        realizations_path,
        key,
        n_realizations,
    )
    return CovarianceData(
        key=key,
        components=components,
        matrix=matrix,
        n_realizations=n_realizations,
        ddof=ddof,
        realization_ids=realization_ids,
    )


def _summary_column(
    parameter: str,
    observable: str,
    mnu_order: int,
) -> str:
    if parameter == "Mnu":
        return (
            f"mean_d_{observable}_dMnu_forward_order{mnu_order}"
        )
    return f"mean_d_{observable}_d{parameter}"


def read_derivative_summary(
    path: Path,
    key: FisherKey,
    required_value_columns: Sequence[str],
) -> DerivativeSummary:
    fieldnames, rows = _read_csv(path)
    required = {
        "snapshot",
        "sample",
        "tracer",
        "n_realizations",
        "k_h_Mpc",
        "Nmodes",
        "binning_mode",
        "k_bin_width_h_Mpc",
        "k_bin_index",
        "k_bin_min_h_Mpc",
        "k_bin_max_h_Mpc",
        *required_value_columns,
    }
    _require_columns(path, fieldnames, required)
    snapshot = _parse_int(
        _constant(path, rows, "snapshot"),
        path,
        "snapshot",
        2,
    )
    sample = _constant(path, rows, "sample")
    tracer = _constant(path, rows, "tracer")
    if snapshot != key.snapshot or sample != key.sample:
        raise DerivativeError(
            f"{path}: derivative identity sample={sample}, snap={snapshot}; "
            f"expected sample={key.sample}, snap={key.snapshot}"
        )
    n_realizations = _parse_int(
        _constant(path, rows, "n_realizations"),
        path,
        "n_realizations",
        2,
    )
    binning_mode = _constant(path, rows, "binning_mode")
    k_bin_width = _constant(path, rows, "k_bin_width_h_Mpc")
    k = np.asarray(
        [
            _parse_float(row["k_h_Mpc"], path, "k_h_Mpc", index + 2)
            for index, row in enumerate(rows)
        ]
    )
    if np.any(np.diff(k) <= 0.0):
        raise DerivativeError(f"{path}: k grid must be strictly increasing")
    nmodes = np.asarray(
        [
            _parse_int(row["Nmodes"], path, "Nmodes", index + 2)
            for index, row in enumerate(rows)
        ],
        dtype=np.int64,
    )
    values = {
        column: np.asarray(
            [
                _parse_float(row[column], path, column, index + 2)
                for index, row in enumerate(rows)
            ]
        )
        for column in required_value_columns
    }
    return DerivativeSummary(
        path=path,
        key=key,
        tracer=tracer,
        n_realizations=n_realizations,
        k=k,
        nmodes=nmodes,
        binning_mode=binning_mode,
        k_bin_width=k_bin_width,
        k_bin_index=[row["k_bin_index"].strip() for row in rows],
        k_bin_min=[row["k_bin_min_h_Mpc"].strip() for row in rows],
        k_bin_max=[row["k_bin_max_h_Mpc"].strip() for row in rows],
        values=values,
    )


def _optional_float_matches(left: str, right: str) -> bool:
    if left == "" or right == "":
        return left == right
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        )
    except ValueError:
        return False


def validate_summary_alignment(
    covariance: CovarianceData,
    summary: DerivativeSummary,
) -> None:
    """Align every covariance component to one derivative-summary k row."""

    for component in covariance.components:
        index = component.k_index
        if index < 0 or index >= len(summary.k):
            raise DerivativeError(
                f"{summary.path}: covariance component {component.label} "
                f"requests missing k_index={index}"
            )
        if component.tracer != summary.tracer:
            raise DerivativeError(
                f"{summary.path}: tracer mismatch for {component.label}"
            )
        if not math.isclose(
            component.k,
            float(summary.k[index]),
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise DerivativeError(
                f"{summary.path}: k mismatch for {component.label}"
            )
        if component.nmodes != int(summary.nmodes[index]):
            raise DerivativeError(
                f"{summary.path}: Nmodes mismatch for {component.label}"
            )
        if (
            component.binning_mode != summary.binning_mode
            or not _optional_float_matches(
                component.k_bin_width,
                summary.k_bin_width,
            )
            or component.k_bin_index != summary.k_bin_index[index]
            or not _optional_float_matches(
                component.k_bin_min,
                summary.k_bin_min[index],
            )
            or not _optional_float_matches(
                component.k_bin_max,
                summary.k_bin_max[index],
            )
        ):
            raise DerivativeError(
                f"{summary.path}: binning mismatch for {component.label}"
            )


def build_derivative_matrix(
    covariance: CovarianceData,
    central: DerivativeSummary | None,
    forward: DerivativeSummary | None,
    parameters: Sequence[str],
    mnu_order: int,
) -> np.ndarray:
    matrix = np.empty(
        (covariance.size, len(parameters)),
        dtype=np.float64,
    )
    for parameter_index, parameter in enumerate(parameters):
        summary = forward if parameter == "Mnu" else central
        if summary is None:
            raise DerivativeError(
                f"missing derivative summary for parameter {parameter}"
            )
        for component in covariance.components:
            column = _summary_column(
                parameter,
                component.observable,
                mnu_order,
            )
            matrix[component.position, parameter_index] = (
                summary.values[column][component.k_index]
            )
    return matrix


def _precision_factor(
    correction: str,
    n_realizations: int,
    size: int,
    ddof: int,
) -> float:
    if correction == "none":
        return 1.0
    if n_realizations <= size + 2:
        raise DerivativeError(
            f"Hartlap correction requires N_realizations > data_vector_size + 2; "
            f"got N={n_realizations}, size={size}. Reduce the vector, add "
            "realizations, or omit --precision-correction hartlap."
        )
    denominator = n_realizations - ddof
    if denominator <= 0:
        raise DerivativeError(
            f"invalid covariance normalization N-ddof={denominator}"
        )
    return (n_realizations - size - 2.0) / denominator


def calculate_fisher(
    covariance: CovarianceData,
    derivative_matrix: np.ndarray,
    *,
    precision_correction: str,
) -> tuple[
    np.ndarray,
    float,
    int,
    float,
    float,
    float,
    float,
    int,
    float,
]:
    matrix = covariance.matrix
    condition = float(np.linalg.cond(matrix))
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigen_min = float(eigenvalues[0])
    eigen_max = float(eigenvalues[-1])
    diagonal = np.diag(matrix)
    if np.any(diagonal <= 0.0):
        raise DerivativeError(
            f"{covariance.key.category}/{covariance.key.sample}: covariance "
            "has a non-positive diagonal"
        )
    standard_deviation = np.sqrt(diagonal)
    correlation = matrix / np.outer(
        standard_deviation,
        standard_deviation,
    )
    correlation = 0.5 * (correlation + correlation.T)
    rank = int(np.linalg.matrix_rank(correlation))
    correlation_condition = float(np.linalg.cond(correlation))
    if rank < covariance.size:
        raise DerivativeError(
            f"{covariance.key.category}/{covariance.key.sample}: covariance "
            f"rank {rank} < data-vector size {covariance.size}; refusing to "
            "use a pseudoinverse"
        )
    try:
        cholesky = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as exc:
        raise DerivativeError(
            f"{covariance.key.category}/{covariance.key.sample}: covariance "
            "is not positive definite; refusing to use a pseudoinverse"
        ) from exc
    factor = _precision_factor(
        precision_correction,
        covariance.n_realizations,
        covariance.size,
        covariance.ddof,
    )
    scaled_derivatives = derivative_matrix / standard_deviation[:, None]
    whitened = np.linalg.solve(cholesky, scaled_derivatives)
    fisher = factor * (whitened.T @ whitened)
    fisher = 0.5 * (fisher + fisher.T)
    if not np.all(np.isfinite(fisher)):
        raise DerivativeError("the Fisher matrix contains non-finite values")
    fisher_rank = int(np.linalg.matrix_rank(fisher))
    fisher_condition = float(np.linalg.cond(fisher))
    return (
        fisher,
        factor,
        rank,
        condition,
        correlation_condition,
        eigen_min,
        eigen_max,
        fisher_rank,
        fisher_condition,
    )


def compute_result(
    fisher_root: Path,
    key: FisherKey,
    parameters: tuple[str, ...],
    mnu_order: int,
    precision_correction: str,
) -> FisherResult:
    covariance = read_covariance(fisher_root, key)
    observables = tuple(
        dict.fromkeys(
            component.observable for component in covariance.components
        )
    )
    central_parameters = [
        parameter for parameter in parameters if parameter in CENTRAL_PARAMETERS
    ]
    central = None
    if central_parameters:
        columns = [
            _summary_column(parameter, observable, mnu_order)
            for parameter in central_parameters
            for observable in observables
        ]
        central = read_derivative_summary(
            derivative_summary_path(fisher_root, "central", key),
            key,
            columns,
        )
        validate_summary_alignment(covariance, central)
    forward = None
    if "Mnu" in parameters:
        columns = [
            _summary_column("Mnu", observable, mnu_order)
            for observable in observables
        ]
        forward = read_derivative_summary(
            derivative_summary_path(fisher_root, "forward", key),
            key,
            columns,
        )
        validate_summary_alignment(covariance, forward)
    derivative_matrix = build_derivative_matrix(
        covariance,
        central,
        forward,
        parameters,
        mnu_order,
    )
    (
        fisher,
        factor,
        covariance_rank,
        covariance_condition,
        correlation_condition,
        eigen_min,
        eigen_max,
        fisher_rank,
        fisher_condition,
    ) = calculate_fisher(
        covariance,
        derivative_matrix,
        precision_correction=precision_correction,
    )
    return FisherResult(
        key=key,
        covariance=covariance,
        parameters=parameters,
        derivative_matrix=derivative_matrix,
        fisher_matrix=fisher,
        precision_factor=factor,
        covariance_rank=covariance_rank,
        covariance_condition=covariance_condition,
        correlation_condition=correlation_condition,
        covariance_eigen_min=eigen_min,
        covariance_eigen_max=eigen_max,
        fisher_rank=fisher_rank,
        fisher_condition=fisher_condition,
        central_realizations=(
            None if central is None else central.n_realizations
        ),
        mnu_realizations=(
            None if forward is None else forward.n_realizations
        ),
    )


def output_paths(
    output_root: Path,
    key: FisherKey,
    parameters: Sequence[str],
    mnu_order: int,
    precision_correction: str,
) -> tuple[Path, Path, Path]:
    suffix = (
        ""
        if key.category == "matter"
        else f"_{key.sample}"
    )
    order_suffix = (
        f"_mnu_order{mnu_order}"
        if "Mnu" in parameters
        else ""
    )
    parameter_suffix = f"_params_{'-'.join(parameters)}"
    precision_suffix = f"_precision_{precision_correction}"
    stem = (
        f"{DATASET}_snap{key.snapshot:03d}{suffix}"
        f"{parameter_suffix}{order_suffix}{precision_suffix}"
    )
    directory = output_root / key.category
    return (
        directory / f"{stem}_fisher.csv",
        directory / f"{stem}_derivative_matrix.csv",
        directory / f"{stem}_diagnostics.csv",
    )


def _format_float(value: float) -> str:
    return f"{float(value):.17e}"


def fisher_rows(
    result: FisherResult,
) -> tuple[list[str], list[dict[str, object]]]:
    fields = ["parameter", *result.parameters]
    rows = []
    for row_index, parameter in enumerate(result.parameters):
        row: dict[str, object] = {"parameter": parameter}
        row.update(
            {
                column_parameter: _format_float(
                    result.fisher_matrix[row_index, column_index]
                )
                for column_index, column_parameter in enumerate(
                    result.parameters
                )
            }
        )
        rows.append(row)
    return fields, rows


def derivative_rows(
    result: FisherResult,
) -> tuple[list[str], list[dict[str, object]]]:
    derivative_fields = [
        f"dD_d{parameter}" for parameter in result.parameters
    ]
    fields = [
        "data_vector_index",
        "vector_position",
        "observable",
        "k_index",
        "k_h_Mpc",
        "Nmodes",
        *derivative_fields,
    ]
    rows = []
    for component in result.covariance.components:
        row: dict[str, object] = {
            "data_vector_index": component.label,
            "vector_position": component.position,
            "observable": component.observable,
            "k_index": component.k_index,
            "k_h_Mpc": _format_float(component.k),
            "Nmodes": component.nmodes,
        }
        for parameter_index, field in enumerate(derivative_fields):
            row[field] = _format_float(
                result.derivative_matrix[
                    component.position,
                    parameter_index,
                ]
            )
        rows.append(row)
    return fields, rows


def diagnostic_rows(
    result: FisherResult,
    *,
    mnu_order: int,
    precision_correction: str,
) -> tuple[list[str], list[dict[str, object]]]:
    fields = [
        "dataset",
        "snapshot",
        "category",
        "sample",
        "parameters",
        "mnu_forward_order",
        "precision_correction",
        "precision_factor",
        "covariance_realizations",
        "covariance_ddof",
        "data_vector_size",
        "covariance_rank",
        "covariance_condition_number",
        "correlation_condition_number",
        "covariance_eigenvalue_min",
        "covariance_eigenvalue_max",
        "central_derivative_realizations",
        "mnu_derivative_realizations",
        "fisher_rank",
        "fisher_condition_number",
    ]
    row = {
        "dataset": DATASET,
        "snapshot": result.key.snapshot,
        "category": result.key.category,
        "sample": result.key.sample,
        "parameters": ",".join(result.parameters),
        "mnu_forward_order": (
            mnu_order if "Mnu" in result.parameters else ""
        ),
        "precision_correction": precision_correction,
        "precision_factor": _format_float(result.precision_factor),
        "covariance_realizations": result.covariance.n_realizations,
        "covariance_ddof": result.covariance.ddof,
        "data_vector_size": result.covariance.size,
        "covariance_rank": result.covariance_rank,
        "covariance_condition_number": _format_float(
            result.covariance_condition
        ),
        "correlation_condition_number": _format_float(
            result.correlation_condition
        ),
        "covariance_eigenvalue_min": _format_float(
            result.covariance_eigen_min
        ),
        "covariance_eigenvalue_max": _format_float(
            result.covariance_eigen_max
        ),
        "central_derivative_realizations": (
            ""
            if result.central_realizations is None
            else result.central_realizations
        ),
        "mnu_derivative_realizations": (
            ""
            if result.mnu_realizations is None
            else result.mnu_realizations
        ),
        "fisher_rank": result.fisher_rank,
        "fisher_condition_number": _format_float(
            result.fisher_condition
        ),
    }
    return fields, [row]


def run(args: argparse.Namespace) -> tuple[int, int]:
    keys = discover_covariance_keys(
        args.fisher_root,
        args.snapnum,
        args.samples,
    )
    prospective_paths = []
    for key in keys:
        prospective_paths.extend(
            output_paths(
                args.output_root,
                key,
                args.parameters,
                args.mnu_order,
                args.precision_correction,
            )
        )
    check_output_collisions(prospective_paths, args.overwrite)

    prepared = []
    for key in keys:
        result = compute_result(
            args.fisher_root,
            key,
            args.parameters,
            args.mnu_order,
            args.precision_correction,
        )
        paths = output_paths(
            args.output_root,
            result.key,
            result.parameters,
            args.mnu_order,
            args.precision_correction,
        )
        prepared.append(
            (
                result,
                paths,
                fisher_rows(result),
                derivative_rows(result),
                diagnostic_rows(
                    result,
                    mnu_order=args.mnu_order,
                    precision_correction=args.precision_correction,
                ),
            )
        )

    for result, paths, fisher_data, derivative_data, diagnostic_data in prepared:
        for path, (fields, rows) in zip(
            paths,
            (fisher_data, derivative_data, diagnostic_data),
        ):
            atomic_write_csv(
                path,
                fields,
                rows,
                overwrite=args.overwrite,
            )
            print(f"[write] {path}")
        if result.fisher_rank < len(result.parameters):
            print(
                f"[warning] {result.key.category}/{result.key.sample}: "
                f"Fisher rank {result.fisher_rank} < parameter count "
                f"{len(result.parameters)}",
                file=sys.stderr,
            )
    return len(prepared), sum(
        result.covariance.size for result, *_ in prepared
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finish_args(args, parser)
    try:
        matrices, total_size = run(args)
    except (DerivativeError, OSError, np.linalg.LinAlgError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"[done] wrote {matrices} Fisher product set(s); "
        f"combined data-vector size across products={total_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
