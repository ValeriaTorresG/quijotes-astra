#!/usr/bin/env python3
"""Compute paired forward P(k) derivatives with respect to neutrino mass."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

if __package__:
    from .derivative_utils import (
        DerivativeError,
        DerivativeProduct,
        ProductKey,
        SpectrumCSV,
        add_common_cli_arguments,
        atomic_write_csv,
        build_summary_rows,
        check_output_collisions,
        derivative_output_path,
        discover_matched_products,
        finish_common_args,
        grid_fieldnames,
        grid_row,
        group_products_for_summary,
        read_spectrum_csv,
        summary_output_path,
        validate_compatible_spectra,
    )
else:
    from derivative_utils import (  # type: ignore[no-redef]
        DerivativeError,
        DerivativeProduct,
        ProductKey,
        SpectrumCSV,
        add_common_cli_arguments,
        atomic_write_csv,
        build_summary_rows,
        check_output_collisions,
        derivative_output_path,
        discover_matched_products,
        finish_common_args,
        grid_fieldnames,
        grid_row,
        group_products_for_summary,
        read_spectrum_csv,
        summary_output_path,
        validate_compatible_spectra,
    )


SCHEME_NAME = "forward"
MNU_STEP_EV = 0.1
MNU_DATASETS = (
    ("fiducial_ZA", 0.0),
    ("Mnu_p", 0.1),
    ("Mnu_pp", 0.2),
    ("Mnu_ppp", 0.4),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute three paired forward finite-difference approximations "
            "to dP(k)/dMnu at Mnu=0. The stencils have first-, second-, and "
            "third-order accuracy and use fiducial_ZA, Mnu_p, Mnu_pp, and "
            "Mnu_ppp from the same realization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_cli_arguments(parser)
    return parser


def _datasets() -> list[str]:
    return [dataset for dataset, _ in MNU_DATASETS]


def derivative_column(accuracy_order: int, observable: str) -> str:
    return f"d_{observable}_dMnu_forward_order{accuracy_order}"


def derivative_columns(observables: Sequence[str]) -> list[str]:
    return [
        derivative_column(order, observable)
        for order in (1, 2, 3)
        for observable in observables
    ]


def scheme_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "finite_difference_scheme": "paired_forward_first_derivative",
        "Mnu_evaluation_eV": f"{0.0:.12e}",
        "Mnu_step_eV": f"{MNU_STEP_EV:.12e}",
        "forward_order1_truncation": "O(h)",
        "forward_order2_truncation": "O(h^2)",
        "forward_order3_truncation": "O(h^3)",
    }
    for dataset, mass in MNU_DATASETS:
        metadata[f"{dataset}_Mnu_eV"] = f"{mass:.12e}"
    return metadata


def forward_stencils(
    fiducial_za: np.ndarray,
    mnu_p: np.ndarray,
    mnu_pp: np.ndarray,
    mnu_ppp: np.ndarray,
    *,
    step: float = MNU_STEP_EV,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return O(h), O(h^2), and O(h^3) estimates of the first derivative."""

    if not np.isfinite(step) or step <= 0.0:
        raise DerivativeError("the neutrino-mass step must be finite and positive")
    d1 = (mnu_p - fiducial_za) / step
    d2 = (-3.0 * fiducial_za + 4.0 * mnu_p - mnu_pp) / (2.0 * step)
    d3 = (
        -21.0 * fiducial_za
        + 32.0 * mnu_p
        - 12.0 * mnu_pp
        + mnu_ppp
    ) / (12.0 * step)
    return d1, d2, d3


def compute_products(
    maps: dict[str, dict[ProductKey, Path]],
    keys: Sequence[ProductKey],
    observables: Sequence[str],
) -> list[DerivativeProduct]:
    """Form all three Mnu stencils separately for every realization."""

    products: list[DerivativeProduct] = []
    for key in keys:
        spectra: dict[str, SpectrumCSV] = {
            dataset: read_spectrum_csv(
                maps[dataset][key],
                expected_dataset=dataset,
                expected_key=key,
                columns=observables,
            )
            for dataset in _datasets()
        }
        reference = spectra["fiducial_ZA"]
        for dataset in _datasets()[1:]:
            validate_compatible_spectra(
                reference,
                spectra[dataset],
                same_realization=True,
            )

        derivatives: dict[str, np.ndarray] = {}
        for observable in observables:
            estimates = forward_stencils(
                reference.values[observable],
                spectra["Mnu_p"].values[observable],
                spectra["Mnu_pp"].values[observable],
                spectra["Mnu_ppp"].values[observable],
            )
            for order, values in enumerate(estimates, start=1):
                derivatives[derivative_column(order, observable)] = values
        products.append(
            DerivativeProduct(
                key=key,
                grid=reference,
                derivatives=derivatives,
            )
        )
    return products


def write_product(
    product: DerivativeProduct,
    output_root: Path,
    *,
    columns: Sequence[str],
    metadata: Mapping[str, object],
    overwrite: bool,
) -> Path:
    path = derivative_output_path(output_root, SCHEME_NAME, product.key)
    fieldnames = [*grid_fieldnames(), *metadata.keys(), *columns]
    rows = []
    for index in range(len(product.grid.k)):
        row: dict[str, object] = {
            **grid_row(product.grid, index),
            **metadata,
        }
        for column in columns:
            row[column] = f"{product.derivatives[column][index]:.12e}"
        rows.append(row)
    atomic_write_csv(path, fieldnames, rows, overwrite=overwrite)
    return path


def run(args: argparse.Namespace) -> tuple[int, int]:
    maps, keys, warnings = discover_matched_products(
        args.pk_root,
        _datasets(),
        args.snapnum,
        simulation_ids=args.simulation_ids,
        samples=args.samples,
        allow_incomplete=args.allow_incomplete,
    )
    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    output_paths = [
        derivative_output_path(args.output_root, SCHEME_NAME, key)
        for key in keys
    ]
    group_keys: dict[tuple[str, str, int], ProductKey] = {}
    for key in keys:
        group_keys.setdefault(
            (key.category, key.sample, key.snapshot),
            key,
        )
    summary_paths = [
        summary_output_path(args.output_root, SCHEME_NAME, key)
        for key in group_keys.values()
    ]
    check_output_collisions(
        [*output_paths, *summary_paths],
        args.overwrite,
    )

    products = compute_products(maps, keys, args.columns)
    result_columns = derivative_columns(args.columns)
    metadata = scheme_metadata()
    prepared_summaries = []
    for group in group_products_for_summary(products):
        fields, rows = build_summary_rows(
            group,
            derivative_columns=result_columns,
            ddof=args.ddof,
            common_metadata=metadata,
        )
        path = summary_output_path(
            args.output_root,
            SCHEME_NAME,
            group[0].key,
        )
        prepared_summaries.append((path, fields, rows))

    for product in products:
        path = write_product(
            product,
            args.output_root,
            columns=result_columns,
            metadata=metadata,
            overwrite=args.overwrite,
        )
        print(f"[write] {path}")

    for path, fields, rows in prepared_summaries:
        atomic_write_csv(path, fields, rows, overwrite=args.overwrite)
        print(f"[write] {path}")
    return len(products), len(prepared_summaries)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finish_common_args(args, parser)
    try:
        products, summaries = run(args)
    except (DerivativeError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"[done] wrote {products} paired realization derivative CSV(s) "
        f"and {summaries} mean/dispersion CSV(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
