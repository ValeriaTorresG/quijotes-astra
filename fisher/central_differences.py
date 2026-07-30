#!/usr/bin/env python3
"""Compute paired central P(k) derivatives for the Quijote cosmologies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

if __package__:
    from .derivative_utils import (
        DerivativeError,
        DerivativeProduct,
        COMBINED_CATEGORY,
        COMBINED_ENVIRONMENT_SAMPLES,
        ProductKey,
        add_common_cli_arguments,
        atomic_write_csv,
        build_summary_rows,
        check_output_collisions,
        combine_environment_derivatives,
        combined_observable,
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
        COMBINED_CATEGORY,
        COMBINED_ENVIRONMENT_SAMPLES,
        ProductKey,
        add_common_cli_arguments,
        atomic_write_csv,
        build_summary_rows,
        check_output_collisions,
        combine_environment_derivatives,
        combined_observable,
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


SCHEME_NAME = "central"


@dataclass(frozen=True)
class CentralParameter:
    """One symmetric Quijote parameter variation."""

    name: str
    minus_dataset: str
    minus_value: float
    plus_dataset: str
    plus_value: float

    @property
    def denominator(self) -> float:
        return self.plus_value - self.minus_value


CENTRAL_PARAMETERS = (
    CentralParameter("Om", "Om_m", 0.3075, "Om_p", 0.3275),
    CentralParameter("h", "h_m", 0.6511, "h_p", 0.6911),
    CentralParameter("ns", "ns_m", 0.9424, "ns_p", 0.9824),
    CentralParameter("s8", "s8_m", 0.819, "s8_p", 0.849),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute central finite-difference derivatives of Quijote P(k) "
            "observables. Every plus/minus combination is formed for the same "
            "realization before the realization mean and dispersion are computed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_cli_arguments(parser)
    return parser


def _datasets() -> list[str]:
    return [
        dataset
        for parameter in CENTRAL_PARAMETERS
        for dataset in (parameter.minus_dataset, parameter.plus_dataset)
    ]


def derivative_column(parameter: str, observable: str) -> str:
    return f"d_{observable}_d{parameter}"


def derivative_columns(observables: Sequence[str]) -> list[str]:
    return [
        derivative_column(parameter.name, observable)
        for parameter in CENTRAL_PARAMETERS
        for observable in observables
    ]


def combined_derivative_columns(
    observables: Sequence[str],
) -> list[str]:
    return derivative_columns(
        [
            combined_observable(sample, observable)
            for sample in COMBINED_ENVIRONMENT_SAMPLES
            for observable in observables
        ]
    )


def combined_column_pairs(
    observables: Sequence[str],
) -> list[tuple[str, str, str]]:
    return [
        (
            sample,
            derivative_column(parameter.name, observable),
            derivative_column(
                parameter.name,
                combined_observable(sample, observable),
            ),
        )
        for sample in COMBINED_ENVIRONMENT_SAMPLES
        for parameter in CENTRAL_PARAMETERS
        for observable in observables
    ]


def scheme_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "finite_difference_scheme": "paired_central",
    }
    for parameter in CENTRAL_PARAMETERS:
        metadata.update(
            {
                f"{parameter.name}_minus_dataset": parameter.minus_dataset,
                f"{parameter.name}_minus_value": f"{parameter.minus_value:.12e}",
                f"{parameter.name}_plus_dataset": parameter.plus_dataset,
                f"{parameter.name}_plus_value": f"{parameter.plus_value:.12e}",
                f"{parameter.name}_denominator": (
                    f"{parameter.denominator:.12e}"
                ),
            }
        )
    return metadata


def compute_products(
    maps: dict[str, dict[ProductKey, Path]],
    keys: Sequence[ProductKey],
    observables: Sequence[str],
    *,
    max_bins: int | None = None,
) -> list[DerivativeProduct]:
    """Form every derivative using plus/minus CSVs with the same key."""

    products: list[DerivativeProduct] = []
    for key in keys:
        spectra = {
            dataset: read_spectrum_csv(
                maps[dataset][key],
                expected_dataset=dataset,
                expected_key=key,
                columns=observables,
                max_bins=max_bins,
            )
            for dataset in _datasets()
        }
        reference = spectra[_datasets()[0]]
        for dataset in _datasets()[1:]:
            validate_compatible_spectra(
                reference,
                spectra[dataset],
                same_realization=True,
            )

        derivatives = {}
        for parameter in CENTRAL_PARAMETERS:
            minus = spectra[parameter.minus_dataset]
            plus = spectra[parameter.plus_dataset]
            for observable in observables:
                name = derivative_column(parameter.name, observable)
                derivatives[name] = (
                    plus.values[observable] - minus.values[observable]
                ) / parameter.denominator
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
    metadata: dict[str, object],
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

    products = compute_products(
        maps,
        keys,
        args.columns,
        max_bins=args.max_bins,
    )
    if args.combine_environments:
        products.extend(
            combine_environment_derivatives(
                products,
                combined_column_pairs(args.columns),
            )
        )
    output_paths = [
        derivative_output_path(args.output_root, SCHEME_NAME, product.key)
        for product in products
    ]
    group_keys: dict[tuple[str, str, int], ProductKey] = {}
    for product in products:
        key = product.key
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

    result_columns = derivative_columns(args.columns)
    combined_columns = combined_derivative_columns(args.columns)
    metadata = scheme_metadata()
    prepared_summaries = []
    for group in group_products_for_summary(products):
        columns = (
            combined_columns
            if group[0].key.category == COMBINED_CATEGORY
            else result_columns
        )
        fields, rows = build_summary_rows(
            group,
            derivative_columns=columns,
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
        columns = (
            combined_columns
            if product.key.category == COMBINED_CATEGORY
            else result_columns
        )
        path = write_product(
            product,
            args.output_root,
            columns=columns,
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
