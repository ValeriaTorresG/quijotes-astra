#!/usr/bin/env python3
"""Validate Mnu finite differences and covariance-inversion stability."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "quijotes-matplotlib-cache"
    )

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from . import fisher_matrix as fm
    from .derivative_utils import (
        COMBINED_CATEGORY,
        COMBINED_SAMPLE,
        DerivativeError,
        ProductKey,
        SpectrumCSV,
        atomic_write_csv,
        check_output_collisions,
        discover_dataset_products,
        path_arg,
        read_spectrum_csv,
    )
else:
    import fisher_matrix as fm  # type: ignore[no-redef]
    from derivative_utils import (  # type: ignore[no-redef]
        COMBINED_CATEGORY,
        COMBINED_SAMPLE,
        DerivativeError,
        ProductKey,
        SpectrumCSV,
        atomic_write_csv,
        check_output_collisions,
        discover_dataset_products,
        path_arg,
        read_spectrum_csv,
    )


PARAMETERS = fm.DEFAULT_PARAMETERS
MNU_ORDERS = (1, 2, 3)
DEFAULT_SAMPLES = ("all", "void", "sheet", "filament", "knot", "combined")
ORDER_COLORS = {1: "#5677A4", 2: "#D4819A", 3: "#E68613"}
DISPLAY_NAMES = {
    "all": "Matter",
    "void": "Void",
    "sheet": "Wall",
    "filament": "Filament",
    "knot": "Cluster",
    "combined": "Combined",
}


class ValidationError(RuntimeError):
    """Raised when stability diagnostics cannot be computed safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Mnu derivative orders, perform leave-one-out covariance "
            "tests, and compare sample-covariance constraints with correlation "
            "shrinkage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pk-root",
        type=path_arg,
        default=Path("pk_data").resolve(),
        help="root containing the original matter/ and env/ spectrum CSVs",
    )
    parser.add_argument(
        "--fisher-root",
        type=path_arg,
        default=Path("pk_data/fisher").resolve(),
        help="root containing covariance and derivative products",
    )
    parser.add_argument(
        "--output-root",
        type=path_arg,
        default=None,
        help="diagnostic output root; defaults to FISHER_ROOT/validation",
    )
    parser.add_argument("--snapnum", type=int, default=3)
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=fm.FISHER_SAMPLES,
        default=list(DEFAULT_SAMPLES),
        help="matter/all, environment, and combined cases to validate",
    )
    parser.add_argument(
        "--mnu-orders",
        nargs="+",
        type=int,
        choices=MNU_ORDERS,
        default=list(MNU_ORDERS),
    )
    parser.add_argument(
        "--precision-correction",
        choices=("hartlap", "none"),
        default="hartlap",
        help="sample-covariance precision correction used for baseline and LOO",
    )
    parser.add_argument(
        "--ratio-floor",
        type=float,
        default=0.05,
        help=(
            "mask derivative ratios where the denominator magnitude is below "
            "this fraction of its maximum within an observable block"
        ),
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def finish_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.snapnum < 0:
        parser.error("--snapnum must be non-negative")
    if not 0.0 <= args.ratio_floor < 1.0:
        parser.error("--ratio-floor must satisfy 0 <= value < 1")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if len(set(args.samples)) != len(args.samples):
        parser.error("--samples contains duplicates")
    if len(set(args.mnu_orders)) != len(args.mnu_orders):
        parser.error("--mnu-orders contains duplicates")
    args.samples = tuple(args.samples)
    args.mnu_orders = tuple(args.mnu_orders)
    args.output_root = (
        path_arg(args.output_root)
        if args.output_root is not None
        else args.fisher_root / "validation"
    )


def case_name(key: fm.FisherKey) -> str:
    return "all" if key.category == "matter" else key.sample


def requested_keys(
    fisher_root: Path,
    snapnum: int,
    samples: Sequence[str],
) -> list[fm.FisherKey]:
    keys = fm.discover_covariance_keys(
        fisher_root,
        snapnum,
        set(samples),
    )
    order = {sample: index for index, sample in enumerate(samples)}
    return sorted(keys, key=lambda key: order[case_name(key)])


def parameter_covariance(
    fisher: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues = np.linalg.eigvalsh(fisher)
    tolerance = (
        np.finfo(np.float64).eps
        * max(fisher.shape)
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if eigenvalues[0] <= tolerance:
        raise ValidationError(
            f"{context}: Fisher matrix is not positive definite; "
            f"minimum eigenvalue={eigenvalues[0]:.6e}"
        )
    inverse = np.linalg.solve(fisher, np.eye(len(fisher)))
    return 0.5 * (inverse + inverse.T)


def sample_covariance(data: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    if data.ndim != 2:
        raise ValidationError("data matrix must be two-dimensional")
    denominator = len(data) - ddof
    if denominator <= 0:
        raise ValidationError("sample covariance has non-positive denominator")
    centered = data - np.mean(data, axis=0)
    covariance = centered.T @ centered / denominator
    return 0.5 * (covariance + covariance.T)


def oas_correlation_shrinkage(
    data: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Shrink the sample correlation toward identity using an OAS intensity."""

    covariance = sample_covariance(data, ddof=1)
    variances = np.diag(covariance)
    if np.any(variances <= 0.0):
        raise ValidationError("cannot shrink a covariance with zero variance")
    standard_deviations = np.sqrt(variances)
    centered = data - np.mean(data, axis=0)
    standardized = centered / standard_deviations
    empirical = standardized.T @ standardized / len(data)
    n_features = empirical.shape[0]
    mu = float(np.trace(empirical)) / n_features
    alpha = float(np.mean(empirical**2))
    denominator = (len(data) + 1.0) * (
        alpha - mu**2 / n_features
    )
    if denominator <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = min((alpha + mu**2) / denominator, 1.0)
    shrunk = (
        (1.0 - shrinkage) * empirical
        + shrinkage * mu * np.eye(n_features)
    )
    shrunk_diagonal = np.sqrt(np.diag(shrunk))
    correlation = shrunk / np.outer(shrunk_diagonal, shrunk_diagonal)
    covariance_shrunk = (
        correlation * np.outer(standard_deviations, standard_deviations)
    )
    return 0.5 * (covariance_shrunk + covariance_shrunk.T), shrinkage


def _split_component_observable(
    key: fm.FisherKey,
    observable: str,
) -> tuple[str, str]:
    if key.category == COMBINED_CATEGORY:
        if "__" not in observable:
            raise ValidationError(
                f"combined observable lacks sample prefix: {observable}"
            )
        return tuple(observable.split("__", 1))  # type: ignore[return-value]
    return key.sample, observable


def build_realization_matrix(
    covariance: fm.CovarianceData,
    products: Mapping[ProductKey, Path],
) -> np.ndarray:
    """Reconstruct the exact realization-by-component matrix used for C."""

    required_by_key: dict[ProductKey, set[str]] = {}
    component_sources: list[tuple[ProductKey, str]] = []
    for component in covariance.components:
        source_sample, observable = _split_component_observable(
            covariance.key,
            component.observable,
        )
        raw_category = "matter" if source_sample == "all" else "env"
        template = ProductKey(
            raw_category,
            source_sample,
            covariance.key.snapshot,
            0,
        )
        component_sources.append((template, observable))
        for simulation_id in covariance.realization_ids:
            source_key = replace(template, simulation_id=simulation_id)
            required_by_key.setdefault(source_key, set()).add(observable)

    spectra: dict[ProductKey, SpectrumCSV] = {}
    for source_key, observables in required_by_key.items():
        try:
            path = products[source_key]
        except KeyError as exc:
            raise ValidationError(
                f"missing fiducial spectrum for {source_key.describe()}"
            ) from exc
        spectra[source_key] = read_spectrum_csv(
            path,
            expected_dataset=fm.DATASET,
            expected_key=source_key,
            columns=sorted(observables),
        )

    data = np.empty(
        (len(covariance.realization_ids), covariance.size),
        dtype=np.float64,
    )
    for realization_position, simulation_id in enumerate(
        covariance.realization_ids
    ):
        for component, (template, observable) in zip(
            covariance.components,
            component_sources,
        ):
            source_key = replace(template, simulation_id=simulation_id)
            spectrum = spectra[source_key]
            index = component.k_index
            if index >= len(spectrum.k):
                raise ValidationError(
                    f"{source_key.describe()}: missing k index {index}"
                )
            if (
                not math.isclose(
                    component.k,
                    float(spectrum.k[index]),
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                )
                or component.nmodes != int(spectrum.nmodes[index])
            ):
                raise ValidationError(
                    f"{source_key.describe()}: covariance component "
                    f"{component.label} is not aligned with the source spectrum"
                )
            data[realization_position, component.position] = (
                spectrum.values[observable][index]
            )
    reconstructed = sample_covariance(data, ddof=covariance.ddof)
    if not np.allclose(
        reconstructed,
        covariance.matrix,
        rtol=2.0e-11,
        atol=1.0e-9 * max(1.0, float(np.max(np.abs(covariance.matrix)))),
    ):
        raise ValidationError(
            f"{covariance.key.category}/{covariance.key.sample}: reconstructed "
            "realization covariance does not match the saved covariance"
        )
    return data


def _safe_ratios(
    numerator: np.ndarray,
    denominator: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = floor * float(np.max(np.abs(denominator)))
    valid = np.abs(denominator) > threshold
    ratios = np.full(denominator.shape, np.nan)
    ratios[valid] = numerator[valid] / denominator[valid]
    return ratios, valid


def derivative_diagnostic_rows(
    covariance: fm.CovarianceData,
    results: Mapping[int, fm.FisherResult],
    ratio_floor: float,
) -> list[dict[str, object]]:
    mnu_index = results[next(iter(results))].parameters.index("Mnu")
    derivatives = {
        order: result.derivative_matrix[:, mnu_index]
        for order, result in results.items()
    }
    ratios_21 = np.full(covariance.size, np.nan)
    ratios_32 = np.full(covariance.size, np.nan)
    valid_21 = np.zeros(covariance.size, dtype=bool)
    valid_32 = np.zeros(covariance.size, dtype=bool)
    blocks: dict[str, list[int]] = {}
    for component in covariance.components:
        blocks.setdefault(component.observable, []).append(component.position)
    if 1 in derivatives and 2 in derivatives:
        for positions in blocks.values():
            index = np.asarray(positions)
            ratios_21[index], valid_21[index] = _safe_ratios(
                derivatives[2][index],
                derivatives[1][index],
                ratio_floor,
            )
    if 2 in derivatives and 3 in derivatives:
        for positions in blocks.values():
            index = np.asarray(positions)
            ratios_32[index], valid_32[index] = _safe_ratios(
                derivatives[3][index],
                derivatives[2][index],
                ratio_floor,
            )

    rows: list[dict[str, object]] = []
    for component in covariance.components:
        position = component.position
        row: dict[str, object] = {
            "data_vector_index": component.label,
            "vector_position": position,
            "observable": component.observable,
            "k_index": component.k_index,
            "k_h_Mpc": f"{component.k:.12e}",
        }
        for order in MNU_ORDERS:
            row[f"dD_dMnu_order{order}"] = (
                ""
                if order not in derivatives
                else f"{derivatives[order][position]:.12e}"
            )
        row.update(
            {
                "R21": (
                    f"{ratios_21[position]:.12e}"
                    if valid_21[position]
                    else ""
                ),
                "R21_masked": int(not valid_21[position]),
                "R32": (
                    f"{ratios_32[position]:.12e}"
                    if valid_32[position]
                    else ""
                ),
                "R32_masked": int(not valid_32[position]),
            }
        )
        rows.append(row)
    return rows


def derivative_plot(
    covariance: fm.CovarianceData,
    results: Mapping[int, fm.FisherResult],
    ratio_floor: float,
) -> plt.Figure:
    mnu_index = results[next(iter(results))].parameters.index("Mnu")
    derivatives = {
        order: result.derivative_matrix[:, mnu_index]
        for order, result in results.items()
    }
    blocks: dict[str, list[fm.DataVectorComponent]] = {}
    for component in covariance.components:
        blocks.setdefault(component.observable, []).append(component)
    figure, axes = plt.subplots(
        len(blocks),
        2,
        figsize=(10.0, 3.3 * len(blocks)),
        squeeze=False,
    )
    for row_index, (observable, components) in enumerate(blocks.items()):
        positions = np.asarray([component.position for component in components])
        k = np.asarray([component.k for component in components])
        derivative_axis, ratio_axis = axes[row_index]
        for order in sorted(derivatives):
            derivative_axis.plot(
                k,
                derivatives[order][positions],
                marker="o",
                linewidth=1.7,
                color=ORDER_COLORS[order],
                label=f"Order {order}",
            )
        if 1 in derivatives and 2 in derivatives:
            ratio, valid = _safe_ratios(
                derivatives[2][positions],
                derivatives[1][positions],
                ratio_floor,
            )
            ratio_axis.plot(
                k[valid],
                ratio[valid],
                marker="o",
                color=ORDER_COLORS[2],
                label=r"$R_{21}$",
            )
        if 2 in derivatives and 3 in derivatives:
            ratio, valid = _safe_ratios(
                derivatives[3][positions],
                derivatives[2][positions],
                ratio_floor,
            )
            ratio_axis.plot(
                k[valid],
                ratio[valid],
                marker="s",
                color=ORDER_COLORS[3],
                label=r"$R_{32}$",
            )
        ratio_axis.axhline(1.0, color="0.35", linewidth=0.8)
        for axis in (derivative_axis, ratio_axis):
            axis.set_xscale("log")
            axis.grid(alpha=0.25)
            axis.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
        derivative_axis.set_ylabel(r"$\partial P/\partial M_\nu$")
        ratio_axis.set_ylabel("Derivative ratio")
        title = observable.split("__", 1)[0] if "__" in observable else observable
        derivative_axis.set_title(DISPLAY_NAMES.get(title, title))
        derivative_axis.legend(frameon=False)
        ratio_axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def _format_float(value: float) -> str:
    return f"{float(value):.12e}"


def fisher_matrix_rows(
    matrix: np.ndarray,
    parameters: Sequence[str],
) -> tuple[list[str], list[dict[str, object]]]:
    fields = ["parameter", *parameters]
    rows = []
    for row_index, parameter in enumerate(parameters):
        row: dict[str, object] = {"parameter": parameter}
        row.update(
            {
                column: _format_float(matrix[row_index, column_index])
                for column_index, column in enumerate(parameters)
            }
        )
        rows.append(row)
    return fields, rows


def save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def heatmap(
    values: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=(1.25 * len(column_labels) + 3.0, 0.55 * len(row_labels) + 2.5)
    )
    image = axis.imshow(values, aspect="auto", cmap="magma")
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if image.norm(value) < 0.55 else "black",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    return figure


def run(args: argparse.Namespace) -> dict[str, Path]:
    keys = requested_keys(args.fisher_root, args.snapnum, args.samples)
    products = discover_dataset_products(
        args.pk_root,
        fm.DATASET,
        args.snapnum,
    )
    outputs: list[Path] = [
        args.output_root / "mnu_order_marginalized_errors.csv",
        args.output_root / "leave_one_out_samples.csv",
        args.output_root / "leave_one_out_summary.csv",
        args.output_root / "shrinkage_comparison.csv",
    ]
    for key in keys:
        name = case_name(key)
        outputs.extend(
            [
                args.output_root / "mnu_derivatives" / f"{name}.csv",
                args.output_root / "mnu_derivatives" / f"{name}.png",
            ]
        )
        for order in args.mnu_orders:
            outputs.append(
                args.output_root
                / "shrinkage_fisher"
                / f"{name}_mnu_order{order}.csv"
            )
    for order in args.mnu_orders:
        outputs.append(
            args.output_root / f"loo_relative_std_order{order}.png"
        )
    outputs.append(args.output_root / "shrinkage_sigma_ratio_order3.png")
    outputs.append(
        args.output_root
        / "shrinkage_sigma_ratio_sample_none_order3.png"
    )
    check_output_collisions(outputs, args.overwrite)

    all_results: dict[
        fm.FisherKey,
        dict[int, fm.FisherResult],
    ] = {}
    data_by_key: dict[fm.FisherKey, np.ndarray] = {}
    derivative_rows_by_case: dict[str, list[dict[str, object]]] = {}
    full_error_rows: list[dict[str, object]] = []
    loo_rows: list[dict[str, object]] = []
    loo_summary_rows: list[dict[str, object]] = []
    shrinkage_rows: list[dict[str, object]] = []

    for key in keys:
        name = case_name(key)
        results = {
            order: fm.compute_result(
                args.fisher_root,
                key,
                PARAMETERS,
                order,
                args.precision_correction,
            )
            for order in args.mnu_orders
        }
        all_results[key] = results
        covariance = results[next(iter(results))].covariance
        data = build_realization_matrix(covariance, products)
        data_by_key[key] = data
        diagnostic_rows = derivative_diagnostic_rows(
            covariance,
            results,
            args.ratio_floor,
        )
        derivative_rows_by_case[name] = diagnostic_rows
        derivative_path = (
            args.output_root / "mnu_derivatives" / f"{name}.csv"
        )
        atomic_write_csv(
            derivative_path,
            list(diagnostic_rows[0]),
            diagnostic_rows,
            overwrite=args.overwrite,
        )
        save_figure(
            derivative_plot(covariance, results, args.ratio_floor),
            args.output_root / "mnu_derivatives" / f"{name}.png",
            dpi=args.dpi,
        )

        shrunk_covariance, shrinkage = oas_correlation_shrinkage(data)
        for order, result in results.items():
            full_parameter_covariance = parameter_covariance(
                result.fisher_matrix,
                context=f"{name}/order{order}/baseline",
            )
            full_sigma = np.sqrt(np.diag(full_parameter_covariance))
            for parameter_index, parameter in enumerate(PARAMETERS):
                full_error_rows.append(
                    {
                        "sample": name,
                        "mnu_order": order,
                        "parameter": parameter,
                        "precision_correction": args.precision_correction,
                        "precision_factor": _format_float(
                            result.precision_factor
                        ),
                        "sigma_marginalized": _format_float(
                            full_sigma[parameter_index]
                        ),
                    }
                )

            loo_sigmas = np.empty(
                (len(data), len(PARAMETERS)),
                dtype=np.float64,
            )
            loo_precision_factors = np.empty(len(data), dtype=np.float64)
            for removed_position, removed_id in enumerate(
                covariance.realization_ids
            ):
                keep = np.arange(len(data)) != removed_position
                loo_matrix = sample_covariance(
                    data[keep],
                    ddof=covariance.ddof,
                )
                loo_covariance = replace(
                    covariance,
                    matrix=loo_matrix,
                    n_realizations=int(np.sum(keep)),
                    realization_ids=[
                        simulation_id
                        for position, simulation_id in enumerate(
                            covariance.realization_ids
                        )
                        if position != removed_position
                    ],
                )
                loo_calculation = fm.calculate_fisher(
                    loo_covariance,
                    result.derivative_matrix,
                    precision_correction=args.precision_correction,
                )
                loo_fisher = loo_calculation[0]
                loo_precision_factors[removed_position] = loo_calculation[1]
                loo_sigmas[removed_position] = np.sqrt(
                    np.diag(
                        parameter_covariance(
                            loo_fisher,
                            context=(
                                f"{name}/order{order}/remove{removed_id}"
                            ),
                        )
                    )
                )
                for parameter_index, parameter in enumerate(PARAMETERS):
                    sigma = loo_sigmas[removed_position, parameter_index]
                    loo_rows.append(
                        {
                            "sample": name,
                            "mnu_order": order,
                            "removed_simulation_id": removed_id,
                            "parameter": parameter,
                            "sigma_marginalized": _format_float(sigma),
                            "full_sigma": _format_float(
                                full_sigma[parameter_index]
                            ),
                            "full_precision_factor": _format_float(
                                result.precision_factor
                            ),
                            "loo_precision_factor": _format_float(
                                loo_precision_factors[removed_position]
                            ),
                            "relative_to_full": _format_float(
                                sigma / full_sigma[parameter_index]
                            ),
                        }
                    )
            for parameter_index, parameter in enumerate(PARAMETERS):
                values = loo_sigmas[:, parameter_index]
                relative = values / full_sigma[parameter_index]
                mean_value = float(np.mean(values))
                relative_to_mean = values / mean_value
                loo_summary_rows.append(
                    {
                        "sample": name,
                        "mnu_order": order,
                        "parameter": parameter,
                        "n_leave_one_out": len(values),
                        "full_sigma": _format_float(
                            full_sigma[parameter_index]
                        ),
                        "full_precision_factor": _format_float(
                            result.precision_factor
                        ),
                        "loo_precision_factor": _format_float(
                            loo_precision_factors[0]
                        ),
                        "loo_mean_sigma": _format_float(mean_value),
                        "loo_std_sigma": _format_float(
                            np.std(values, ddof=1)
                        ),
                        "loo_coefficient_of_variation": _format_float(
                            np.std(values, ddof=1) / mean_value
                        ),
                        "relative_std_to_full": _format_float(
                            np.std(values, ddof=1)
                            / full_sigma[parameter_index]
                        ),
                        "relative_min": _format_float(np.min(relative)),
                        "relative_max": _format_float(np.max(relative)),
                        "max_abs_fractional_change": _format_float(
                            np.max(np.abs(relative - 1.0))
                        ),
                        "relative_min_to_loo_mean": _format_float(
                            np.min(relative_to_mean)
                        ),
                        "relative_max_to_loo_mean": _format_float(
                            np.max(relative_to_mean)
                        ),
                        "max_abs_fractional_change_from_loo_mean": (
                            _format_float(
                                np.max(np.abs(relative_to_mean - 1.0))
                            )
                        ),
                    }
                )

            sample_none_fisher = fm.calculate_fisher(
                covariance,
                result.derivative_matrix,
                precision_correction="none",
            )[0]
            shrunk_data = replace(covariance, matrix=shrunk_covariance)
            shrinkage_fisher = fm.calculate_fisher(
                shrunk_data,
                result.derivative_matrix,
                precision_correction="none",
            )[0]
            sigma_sample_none = np.sqrt(
                np.diag(
                    parameter_covariance(
                        sample_none_fisher,
                        context=f"{name}/order{order}/sample-none",
                    )
                )
            )
            sigma_shrinkage = np.sqrt(
                np.diag(
                    parameter_covariance(
                        shrinkage_fisher,
                        context=f"{name}/order{order}/shrinkage",
                    )
                )
            )
            for parameter_index, parameter in enumerate(PARAMETERS):
                shrinkage_rows.append(
                    {
                        "sample": name,
                        "mnu_order": order,
                        "parameter": parameter,
                        "oas_correlation_shrinkage": _format_float(shrinkage),
                        "baseline_precision_correction": (
                            args.precision_correction
                        ),
                        "sigma_sample_baseline": _format_float(
                            full_sigma[parameter_index]
                        ),
                        "sigma_sample_none": _format_float(
                            sigma_sample_none[parameter_index]
                        ),
                        "sigma_shrinkage_none": _format_float(
                            sigma_shrinkage[parameter_index]
                        ),
                        "shrinkage_over_sample_none": _format_float(
                            sigma_shrinkage[parameter_index]
                            / sigma_sample_none[parameter_index]
                        ),
                        "shrinkage_over_sample_baseline": _format_float(
                            sigma_shrinkage[parameter_index]
                            / full_sigma[parameter_index]
                        ),
                    }
                )
            matrix_path = (
                args.output_root
                / "shrinkage_fisher"
                / f"{name}_mnu_order{order}.csv"
            )
            fields, rows = fisher_matrix_rows(
                shrinkage_fisher,
                PARAMETERS,
            )
            atomic_write_csv(
                matrix_path,
                fields,
                rows,
                overwrite=args.overwrite,
            )

    table_specs = [
        (
            args.output_root / "mnu_order_marginalized_errors.csv",
            full_error_rows,
        ),
        (
            args.output_root / "leave_one_out_samples.csv",
            loo_rows,
        ),
        (
            args.output_root / "leave_one_out_summary.csv",
            loo_summary_rows,
        ),
        (
            args.output_root / "shrinkage_comparison.csv",
            shrinkage_rows,
        ),
    ]
    for path, rows in table_specs:
        atomic_write_csv(
            path,
            list(rows[0]),
            rows,
            overwrite=args.overwrite,
        )

    sample_names = [case_name(key) for key in keys]
    for order in args.mnu_orders:
        values = np.empty((len(keys), len(PARAMETERS)))
        for sample_index, name in enumerate(sample_names):
            for parameter_index, parameter in enumerate(PARAMETERS):
                row = next(
                    row
                    for row in loo_summary_rows
                    if row["sample"] == name
                    and row["mnu_order"] == order
                    and row["parameter"] == parameter
                )
                values[sample_index, parameter_index] = float(
                    row["relative_std_to_full"]
                )
        save_figure(
            heatmap(
                values,
                [DISPLAY_NAMES[name] for name in sample_names],
                PARAMETERS,
                title=f"Leave-one-out instability: Mnu order {order}",
                colorbar_label=r"$\mathrm{std}(\sigma_{\rm LOO})/\sigma_{\rm full}$",
            ),
            args.output_root / f"loo_relative_std_order{order}.png",
            dpi=args.dpi,
        )

    comparison_order = 3 if 3 in args.mnu_orders else args.mnu_orders[-1]
    shrinkage_values = np.empty((len(keys), len(PARAMETERS)))
    for sample_index, name in enumerate(sample_names):
        for parameter_index, parameter in enumerate(PARAMETERS):
            row = next(
                row
                for row in shrinkage_rows
                if row["sample"] == name
                and row["mnu_order"] == comparison_order
                and row["parameter"] == parameter
            )
            shrinkage_values[sample_index, parameter_index] = float(
                row["shrinkage_over_sample_baseline"]
            )
    save_figure(
        heatmap(
            shrinkage_values,
            [DISPLAY_NAMES[name] for name in sample_names],
            PARAMETERS,
            title=f"Shrinkage vs Hartlap: Mnu order {comparison_order}",
            colorbar_label=r"$\sigma_{\rm shrink}/\sigma_{\rm Hartlap}$",
        ),
        args.output_root / "shrinkage_sigma_ratio_order3.png",
        dpi=args.dpi,
    )
    shrinkage_none_values = np.empty((len(keys), len(PARAMETERS)))
    for sample_index, name in enumerate(sample_names):
        for parameter_index, parameter in enumerate(PARAMETERS):
            row = next(
                row
                for row in shrinkage_rows
                if row["sample"] == name
                and row["mnu_order"] == comparison_order
                and row["parameter"] == parameter
            )
            shrinkage_none_values[sample_index, parameter_index] = float(
                row["shrinkage_over_sample_none"]
            )
    save_figure(
        heatmap(
            shrinkage_none_values,
            [DISPLAY_NAMES[name] for name in sample_names],
            PARAMETERS,
            title=(
                "Correlation shrinkage vs uncorrected sample covariance: "
                f"Mnu order {comparison_order}"
            ),
            colorbar_label=r"$\sigma_{\rm shrink}/\sigma_{\rm sample}$",
        ),
        args.output_root
        / "shrinkage_sigma_ratio_sample_none_order3.png",
        dpi=args.dpi,
    )
    return {
        "order_errors": args.output_root
        / "mnu_order_marginalized_errors.csv",
        "loo_summary": args.output_root / "leave_one_out_summary.csv",
        "shrinkage": args.output_root / "shrinkage_comparison.csv",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finish_args(args, parser)
    try:
        outputs = run(args)
    except (
        DerivativeError,
        ValidationError,
        OSError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    for label, path in outputs.items():
        print(f"[write] {label}: {path}")
    print("[done] completed Mnu-order, leave-one-out, and shrinkage validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
