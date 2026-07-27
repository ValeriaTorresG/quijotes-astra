#!/usr/bin/env python3
"""Estimate P(k) covariance matrices from fiducial realizations only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

if __package__:
    from .derivative_utils import (
        DEFAULT_PK_ROOT,
        DerivativeError,
        KNOWN_SAMPLES,
        ProductKey,
        SpectrumCSV,
        atomic_write_csv,
        check_output_collisions,
        discover_dataset_products,
        nonnegative_int,
        parse_simulation_ids,
        path_arg,
        read_spectrum_csv,
        validate_compatible_spectra,
        validate_observable_names,
    )
else:
    from derivative_utils import (  # type: ignore[no-redef]
        DEFAULT_PK_ROOT,
        DerivativeError,
        KNOWN_SAMPLES,
        ProductKey,
        SpectrumCSV,
        atomic_write_csv,
        check_output_collisions,
        discover_dataset_products,
        nonnegative_int,
        parse_simulation_ids,
        path_arg,
        read_spectrum_csv,
        validate_compatible_spectra,
        validate_observable_names,
    )


DATASET = "fiducial"
DEFAULT_COVARIANCE_COLUMNS = ("Pk0_shot_subtracted_Mpc3_h3",)
SCHEME_DIRECTORY = "covariance"


@dataclass
class CovarianceResult:
    """One sample's ordered data vector and sample covariance."""

    key: ProductKey
    grid: SpectrumCSV
    observables: tuple[str, ...]
    realization_ids: list[int]
    mean: np.ndarray
    covariance: np.ndarray
    rank: int

    @property
    def size(self) -> int:
        return int(self.mean.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate covariance matrices across standard fiducial Quijote "
            "realizations. Each matter/environment sample is treated as a "
            "separate data vector; selected observable blocks are concatenated "
            "in command-line order, with increasing k inside each block."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="fiducial realization IDs; omit, or pass 'all', to discover them",
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
        default=list(DEFAULT_COVARIANCE_COLUMNS),
        metavar="COLUMN",
        help=(
            "ordered observable blocks in the data vector; all cross-covariances "
            "between selected blocks are retained"
        ),
    )
    parser.add_argument(
        "--ddof",
        type=int,
        choices=(0, 1),
        default=1,
        help="degrees of freedom in the covariance denominator N-ddof",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "when explicit IDs are requested, let each sample use its available "
            "subset; by default every requested ID must exist for every sample"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing covariance products atomically",
    )
    return parser


def finish_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    args.output_root = (
        path_arg(args.output_root)
        if args.output_root is not None
        else args.pk_root / "fisher"
    )
    args.simulation_ids = parse_simulation_ids(args.simulation_ids, parser)
    args.samples = None if args.samples is None else set(args.samples)
    args.columns = validate_observable_names(args.columns, parser)


def _group_selected_paths(
    paths: Mapping[ProductKey, Path],
    *,
    simulation_ids: set[int] | None,
    samples: set[str] | None,
    allow_incomplete: bool,
) -> list[tuple[tuple[str, str, int], list[tuple[ProductKey, Path]]]]:
    selected = {
        key: path
        for key, path in paths.items()
        if (
            simulation_ids is None or key.simulation_id in simulation_ids
        )
        and (samples is None or key.sample in samples)
    }
    if not selected:
        raise DerivativeError("no fiducial products match the requested selection")

    seen_samples = {key.sample for key in selected}
    if samples is not None:
        missing_samples = samples - seen_samples
        if missing_samples:
            raise DerivativeError(
                "requested fiducial samples not found: "
                + ", ".join(sorted(missing_samples))
            )
    seen_ids = {key.simulation_id for key in selected}
    if simulation_ids is not None:
        missing_ids = simulation_ids - seen_ids
        if missing_ids:
            raise DerivativeError(
                "requested fiducial simulation IDs not found: "
                + ", ".join(str(value) for value in sorted(missing_ids))
            )

    grouped: dict[
        tuple[str, str, int],
        list[tuple[ProductKey, Path]],
    ] = {}
    for key, path in selected.items():
        group = (key.category, key.sample, key.snapshot)
        grouped.setdefault(group, []).append((key, path))

    if simulation_ids is not None and not allow_incomplete:
        failures = []
        for group, products in sorted(grouped.items()):
            available = {key.simulation_id for key, _ in products}
            missing = simulation_ids - available
            if missing:
                failures.append(
                    f"{group[0]}/{group[1]} missing IDs "
                    + ",".join(str(value) for value in sorted(missing))
                )
        if failures:
            raise DerivativeError(
                "requested realization set is incomplete: "
                + "; ".join(failures)
                + ". Pass --allow-incomplete to use each available subset."
            )

    return [
        (
            group,
            sorted(products, key=lambda item: item[0].simulation_id),
        )
        for group, products in sorted(grouped.items())
    ]


def covariance_paths(
    output_root: Path,
    key: ProductKey,
) -> tuple[Path, Path, Path]:
    suffix = "" if key.category == "matter" else f"_{key.sample}"
    stem = f"{DATASET}_snap{key.snapshot:03d}{suffix}"
    directory = output_root / SCHEME_DIRECTORY / key.category
    return (
        directory / f"{stem}_data_vector.csv",
        directory / f"{stem}_covariance.csv",
        directory / f"{stem}_realizations.csv",
    )


def _concatenate_observables(
    spectrum: SpectrumCSV,
    observables: Sequence[str],
) -> np.ndarray:
    return np.concatenate(
        [spectrum.values[observable] for observable in observables]
    )


def estimate_covariance(
    spectra: Sequence[SpectrumCSV],
    observables: Sequence[str],
    *,
    ddof: int,
) -> CovarianceResult:
    """Calculate C = X_centered.T X_centered / (N-ddof)."""

    if len(spectra) < 2:
        raise DerivativeError(
            f"{spectra[0].key.category}/{spectra[0].key.sample} needs at least "
            "two fiducial realizations to estimate a covariance"
        )
    reference = spectra[0]
    for candidate in spectra[1:]:
        validate_compatible_spectra(
            reference,
            candidate,
            same_realization=False,
        )
    denominator = len(spectra) - ddof
    if denominator <= 0:
        raise DerivativeError(
            f"covariance denominator N-ddof is not positive: "
            f"N={len(spectra)}, ddof={ddof}"
        )
    data = np.stack(
        [
            _concatenate_observables(spectrum, observables)
            for spectrum in spectra
        ],
        axis=0,
    )
    mean = np.mean(data, axis=0)
    centered = data - mean
    covariance = centered.T @ centered / denominator
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise DerivativeError("the estimated covariance contains non-finite values")
    rank = int(np.linalg.matrix_rank(covariance))
    return CovarianceResult(
        key=reference.key,
        grid=reference,
        observables=tuple(observables),
        realization_ids=[
            spectrum.key.simulation_id for spectrum in spectra
        ],
        mean=mean,
        covariance=covariance,
        rank=rank,
    )


def _format_float(value: float) -> str:
    return f"{float(value):.17e}"


def _vector_label(index: int) -> str:
    return f"v{index:06d}"


def data_vector_rows(
    result: CovarianceResult,
    *,
    ddof: int,
) -> tuple[list[str], list[dict[str, object]]]:
    """Map covariance indices back to observable and k."""

    fields = [
        "data_vector_index",
        "vector_position",
        "dataset",
        "n_realizations",
        "snapshot",
        "category",
        "sample",
        "tracer",
        "data_vector_order",
        "observable_index",
        "observable",
        "k_index",
        "k_h_Mpc",
        "Nmodes",
        "binning_mode",
        "k_bin_width_h_Mpc",
        "k_bin_index",
        "k_bin_min_h_Mpc",
        "k_bin_max_h_Mpc",
        "mean",
        "std",
        "covariance_ddof",
        "data_vector_size",
        "covariance_rank",
    ]
    diagonal = np.diag(result.covariance)
    if np.any(diagonal < -1.0e-12 * max(1.0, float(np.max(np.abs(diagonal))))):
        raise DerivativeError("the covariance has a significantly negative diagonal")
    standard_deviation = np.sqrt(np.maximum(diagonal, 0.0))

    rows: list[dict[str, object]] = []
    n_k = len(result.grid.k)
    for observable_index, observable in enumerate(result.observables):
        for k_index in range(n_k):
            vector_position = observable_index * n_k + k_index
            rows.append(
                {
                    "data_vector_index": _vector_label(vector_position),
                    "vector_position": vector_position,
                    "dataset": DATASET,
                    "n_realizations": len(result.realization_ids),
                    "snapshot": result.key.snapshot,
                    "category": result.key.category,
                    "sample": result.key.sample,
                    "tracer": result.grid.tracer,
                    "data_vector_order": "observable_then_k",
                    "observable_index": observable_index,
                    "observable": observable,
                    "k_index": k_index,
                    "k_h_Mpc": _format_float(result.grid.k[k_index]),
                    "Nmodes": int(result.grid.nmodes[k_index]),
                    "binning_mode": result.grid.metadata.get(
                        "binning_mode",
                        "",
                    ),
                    "k_bin_width_h_Mpc": result.grid.metadata.get(
                        "k_bin_width_h_Mpc",
                        "",
                    ),
                    "k_bin_index": (
                        ""
                        if result.grid.k_bin_index is None
                        else int(result.grid.k_bin_index[k_index])
                    ),
                    "k_bin_min_h_Mpc": (
                        ""
                        if result.grid.k_bin_min is None
                        else _format_float(result.grid.k_bin_min[k_index])
                    ),
                    "k_bin_max_h_Mpc": (
                        ""
                        if result.grid.k_bin_max is None
                        else _format_float(result.grid.k_bin_max[k_index])
                    ),
                    "mean": _format_float(result.mean[vector_position]),
                    "std": _format_float(
                        standard_deviation[vector_position]
                    ),
                    "covariance_ddof": ddof,
                    "data_vector_size": result.size,
                    "covariance_rank": result.rank,
                }
            )
    return fields, rows


def covariance_matrix_rows(
    result: CovarianceResult,
    *,
    ddof: int,
) -> tuple[list[str], list[dict[str, object]]]:
    """Build a dense, index-labelled covariance matrix CSV."""

    labels = [_vector_label(index) for index in range(result.size)]
    fields = [
        "data_vector_index",
        "n_realizations",
        "covariance_ddof",
        *labels,
    ]
    rows: list[dict[str, object]] = []
    for row_index, row_values in enumerate(result.covariance):
        row: dict[str, object] = {
            "data_vector_index": labels[row_index],
            "n_realizations": len(result.realization_ids),
            "covariance_ddof": ddof,
        }
        row.update(
            {
                label: _format_float(value)
                for label, value in zip(labels, row_values)
            }
        )
        rows.append(row)
    return fields, rows


def realization_rows(
    result: CovarianceResult,
) -> tuple[list[str], list[dict[str, object]]]:
    """Record the exact ordered realization set without repeating long IDs."""

    fields = [
        "realization_order",
        "simulation_id",
        "dataset",
        "snapshot",
        "category",
        "sample",
    ]
    rows = [
        {
            "realization_order": order,
            "simulation_id": simulation_id,
            "dataset": DATASET,
            "snapshot": result.key.snapshot,
            "category": result.key.category,
            "sample": result.key.sample,
        }
        for order, simulation_id in enumerate(result.realization_ids)
    ]
    return fields, rows


def run(args: argparse.Namespace) -> tuple[int, int]:
    paths = discover_dataset_products(args.pk_root, DATASET, args.snapnum)
    groups = _group_selected_paths(
        paths,
        simulation_ids=args.simulation_ids,
        samples=args.samples,
        allow_incomplete=args.allow_incomplete,
    )
    output_pairs = [
        covariance_paths(args.output_root, products[0][0])
        for _, products in groups
    ]
    check_output_collisions(
        [path for pair in output_pairs for path in pair],
        args.overwrite,
    )

    prepared = []
    for _, products in groups:
        spectra = [
            read_spectrum_csv(
                path,
                expected_dataset=DATASET,
                expected_key=key,
                columns=args.columns,
            )
            for key, path in products
        ]
        result = estimate_covariance(
            spectra,
            args.columns,
            ddof=args.ddof,
        )
        vector_fields, vector_rows = data_vector_rows(
            result,
            ddof=args.ddof,
        )
        covariance_fields, covariance_rows = covariance_matrix_rows(
            result,
            ddof=args.ddof,
        )
        realization_fields, realization_data = realization_rows(result)
        vector_path, covariance_path, realization_path = covariance_paths(
            args.output_root,
            result.key,
        )
        prepared.append(
            (
                result,
                vector_path,
                vector_fields,
                vector_rows,
                covariance_path,
                covariance_fields,
                covariance_rows,
                realization_path,
                realization_fields,
                realization_data,
            )
        )

    for (
        result,
        vector_path,
        vector_fields,
        vector_rows,
        covariance_path,
        covariance_fields,
        covariance_rows,
        realization_path,
        realization_fields,
        realization_data,
    ) in prepared:
        atomic_write_csv(
            vector_path,
            vector_fields,
            vector_rows,
            overwrite=args.overwrite,
        )
        atomic_write_csv(
            covariance_path,
            covariance_fields,
            covariance_rows,
            overwrite=args.overwrite,
        )
        atomic_write_csv(
            realization_path,
            realization_fields,
            realization_data,
            overwrite=args.overwrite,
        )
        print(f"[write] {vector_path}")
        print(f"[write] {covariance_path}")
        print(f"[write] {realization_path}")
        if result.rank < result.size:
            print(
                f"[warning] {result.key.category}/{result.key.sample}: "
                f"covariance rank {result.rank} < data-vector size "
                f"{result.size}; the matrix is singular",
                file=sys.stderr,
            )
    return len(prepared), sum(result.size for result, *_ in prepared)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finish_args(args, parser)
    if {
        "Pk0_raw_Mpc3_h3",
        "Pk0_shot_subtracted_Mpc3_h3",
    }.issubset(args.columns):
        print(
            "[warning] both raw and shot-noise-subtracted P0 were selected; "
            "this may produce a redundant or ill-conditioned data vector",
            file=sys.stderr,
        )
    try:
        groups, total_size = run(args)
    except (DerivativeError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"[done] wrote {groups} fiducial covariance product set(s); "
        f"combined data-vector size across products={total_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
