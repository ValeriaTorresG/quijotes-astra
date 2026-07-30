#!/usr/bin/env python3
"""Plot marginalized Fisher constraints for the Quijote parameters."""

from __future__ import annotations

import argparse
import csv
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
matplotlib.rcParams['text.usetex'] = True
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Patch


PARAMETERS = ("Om", "h", "ns", "s8", "Mnu")
SAMPLES = ("matter", "void", "sheet", "filament", "knot", "combined")
FIDUCIAL_VALUES = {
    "Om": 0.3175,
    "h": 0.6711,
    "ns": 0.9624,
    "s8": 0.834,
    "Mnu": 0.0,
}
PARAMETER_LABELS = {
    "Om": r"$\Omega_m$",
    "h": r"$h$",
    "ns": r"$n_s$",
    "s8": r"$\sigma_8$",
    "Mnu": r"$M_\nu\,[\mathrm{eV}]$",
}
SAMPLE_LABELS = {
    "matter": "Matter (all halos)",
    "void": "Void",
    "sheet": "Sheet",
    "filament": "Filament",
    "knot": "Knot",
    "combined": "Combined",
}
SAMPLE_COLORS = {
    "matter": "#7F7F7F",
    "void": "#5677A4",
    "sheet": "#9761B0",
    "filament": "#85B5B2",
    "knot": "#D4819A",
    "combined": "#E68613",
}
CONFIDENCE_LEVELS = (0.6827, 0.9545)
INNER_TONE_WHITE_MIX = 0.12
OUTER_TONE_WHITE_MIX = 0.58


class PlotError(RuntimeError):
    """Raised when a Fisher product cannot be plotted safely."""


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read labelled Fisher CSVs, invert the complete Fisher matrix, "
            "and plot marginalized 1D constraints and 2D confidence ellipses."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fisher-root",
        type=path_arg,
        default=Path("pk_data/fisher").resolve(),
        help="root containing matrices/env/",
    )
    parser.add_argument(
        "--snapnum",
        type=int,
        default=3,
        help="snapshot used in the Fisher filenames",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=SAMPLES,
        default=list(SAMPLES),
        help="matter, environment, and/or combined constraints to overlay",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        choices=PARAMETERS,
        default=list(PARAMETERS),
        help="ordered parameters shown in the triangle plot",
    )
    parser.add_argument(
        "--mnu-order",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Mnu forward-difference order in the Fisher filename",
    )
    parser.add_argument(
        "--precision-correction",
        choices=("none", "hartlap"),
        default="none",
        help="precision correction in the Fisher filename",
    )
    parser.add_argument(
        "--fiducial",
        nargs="*",
        metavar="PARAM=VALUE",
        default=[],
        help="override one or more fiducial parameter values",
    )
    parser.add_argument(
        "--range-sigma",
        type=float,
        default=2.5,
        help="axis half-width in units of the largest marginalized error",
    )
    parser.add_argument(
        "--output",
        type=path_arg,
        default=None,
        help="plot path; defaults to FISHER_ROOT/matrices/fisher_ellipses.png",
    )
    parser.add_argument(
        "--errors-output",
        type=path_arg,
        default=None,
        help="marginalized-error CSV path; defaults beside the plot",
    )
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing plot and marginalized-error CSV",
    )
    return parser


def finish_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.snapnum < 0:
        parser.error("--snapnum must be non-negative")
    if args.range_sigma <= 0.0 or not math.isfinite(args.range_sigma):
        parser.error("--range-sigma must be finite and positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if len(set(args.samples)) != len(args.samples):
        parser.error("--samples contains duplicates")
    if len(set(args.params)) != len(args.params):
        parser.error("--params contains duplicates")
    args.samples = tuple(args.samples)
    args.params = tuple(args.params)
    args.output = (
        args.output
        if args.output is not None
        else args.fisher_root / "matrices" / "fisher_ellipses.png"
    )
    args.errors_output = (
        args.errors_output
        if args.errors_output is not None
        else args.output.with_name(f"{args.output.stem}_marginalized_errors.csv")
    )


def parse_fiducials(
    overrides: Sequence[str],
) -> dict[str, float]:
    values = dict(FIDUCIAL_VALUES)
    for item in overrides:
        if "=" not in item:
            raise PlotError(
                f"invalid --fiducial value {item!r}; expected PARAM=VALUE"
            )
        parameter, raw_value = item.split("=", 1)
        parameter = parameter.strip()
        if parameter not in PARAMETERS:
            raise PlotError(
                f"unknown fiducial parameter {parameter!r}; "
                f"choose from {', '.join(PARAMETERS)}"
            )
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise PlotError(
                f"invalid fiducial value for {parameter}: {raw_value!r}"
            ) from exc
        if not math.isfinite(value):
            raise PlotError(f"fiducial value for {parameter} is not finite")
        values[parameter] = value
    return values


def fisher_path(
    fisher_root: Path,
    sample: str,
    snapnum: int,
    mnu_order: int,
    precision_correction: str,
) -> Path:
    parameter_suffix = "-".join(PARAMETERS)
    if sample == "matter":
        category = "matter"
        sample_suffix = ""
    elif sample == "combined":
        category = "combined"
        sample_suffix = "_combined"
    else:
        category = "env"
        sample_suffix = f"_{sample}"
    name = (
        f"fiducial_snap{snapnum:03d}{sample_suffix}_params_{parameter_suffix}"
        f"_mnu_order{mnu_order}_precision_{precision_correction}_fisher.csv"
    )
    return fisher_root / "matrices" / category / name


def read_fisher_csv(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or reader.fieldnames[:1] != ["parameter"]:
                raise PlotError(f"{path}: expected first column 'parameter'")
            parameters = tuple(reader.fieldnames[1:])
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PlotError(f"cannot read {path}: {exc}") from exc
    if not parameters or len(rows) != len(parameters):
        raise PlotError(f"{path}: Fisher matrix is not square")
    row_parameters = tuple(row.get("parameter", "").strip() for row in rows)
    if row_parameters != parameters:
        raise PlotError(
            f"{path}: row parameter order does not match the CSV header"
        )
    try:
        matrix = np.asarray(
            [
                [float(row[parameter]) for parameter in parameters]
                for row in rows
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlotError(f"{path}: invalid Fisher matrix value") from exc
    if not np.all(np.isfinite(matrix)):
        raise PlotError(f"{path}: Fisher matrix contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12 * scale):
        raise PlotError(f"{path}: Fisher matrix is not symmetric")
    return parameters, 0.5 * (matrix + matrix.T)


def marginalized_covariance(
    fisher: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(fisher)
    tolerance = (
        np.finfo(np.float64).eps
        * max(fisher.shape)
        * max(1.0, float(np.max(np.abs(eigenvalues))))
    )
    if eigenvalues[0] <= tolerance:
        raise PlotError(
            f"{context}: Fisher matrix is not positive definite "
            f"(minimum eigenvalue={eigenvalues[0]:.6e})"
        )
    try:
        covariance = np.linalg.solve(fisher, np.eye(len(fisher)))
    except np.linalg.LinAlgError as exc:
        raise PlotError(f"{context}: Fisher matrix cannot be inverted") from exc
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) <= 0.0):
        raise PlotError(
            f"{context}: marginalized covariance has non-positive variance"
        )
    return covariance


def load_covariances(
    fisher_root: Path,
    samples: Sequence[str],
    snapnum: int,
    mnu_order: int,
    precision_correction: str,
) -> dict[str, np.ndarray]:
    covariances: dict[str, np.ndarray] = {}
    for sample in samples:
        path = fisher_path(
            fisher_root,
            sample,
            snapnum,
            mnu_order,
            precision_correction,
        )
        if not path.is_file():
            raise PlotError(f"missing Fisher matrix: {path}")
        parameters, fisher = read_fisher_csv(path)
        if parameters != PARAMETERS:
            raise PlotError(
                f"{path}: expected parameters {PARAMETERS}, found {parameters}"
            )
        covariances[sample] = marginalized_covariance(
            fisher,
            context=f"{sample} ({path})",
        )
    return covariances


def confidence_scale_2d(level: float) -> float:
    return math.sqrt(-2.0 * math.log(1.0 - level))


def lighter_tone(color: str, white_mix: float) -> tuple[float, float, float]:
    """Blend a base color toward white while preserving its hue family."""

    rgb = np.asarray(to_rgb(color), dtype=np.float64)
    return tuple(rgb * (1.0 - white_mix) + white_mix)


def add_confidence_ellipse(
    ax: plt.Axes,
    covariance: np.ndarray,
    center: np.ndarray,
    *,
    level: float,
    facecolor: str | tuple[float, float, float],
    edgecolor: str,
    linewidth: float,
    alpha: float,
    zorder: float,
) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0.0:
        raise PlotError("a 2D marginalized covariance is not positive definite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    scale = confidence_scale_2d(level)
    width, height = 2.0 * scale * np.sqrt(eigenvalues)
    angle = math.degrees(
        math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )
    fill_rgba = (*to_rgb(facecolor), alpha)
    ax.add_patch(
        Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=fill_rgba,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def plot_triangle(
    covariances: Mapping[str, np.ndarray],
    fiducials: Mapping[str, float],
    parameters: Sequence[str],
    samples: Sequence[str],
    *,
    range_sigma: float,
) -> plt.Figure:
    full_indices = [PARAMETERS.index(parameter) for parameter in parameters]
    centers = np.asarray([fiducials[parameter] for parameter in parameters])
    selected = {
        sample: covariance[np.ix_(full_indices, full_indices)]
        for sample, covariance in covariances.items()
    }
    errors = {
        sample: np.sqrt(np.diag(covariance))
        for sample, covariance in selected.items()
    }
    limit_errors = np.max(np.stack(list(errors.values())), axis=0)
    n_parameters = len(parameters)
    fig, axes = plt.subplots(
        n_parameters,
        n_parameters,
        figsize=(3.0 * n_parameters, 3.0 * n_parameters),
        squeeze=False,
    )

    for row in range(n_parameters):
        for column in range(n_parameters):
            ax = axes[row, column]
            if column > row:
                ax.axis("off")
                continue
            if row == column:
                center = centers[row]
                half_width = range_sigma * limit_errors[row]
                x = np.linspace(center - half_width, center + half_width, 500)
                for sample in samples:
                    sigma = errors[sample][row]
                    density = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                    ax.fill_between(
                        x,
                        0.0,
                        density,
                        facecolor=lighter_tone(
                            SAMPLE_COLORS[sample],
                            INNER_TONE_WHITE_MIX,
                        ),
                        edgecolor="none",
                        alpha=0.38,
                    )
                    ax.plot(
                        x,
                        density,
                        color=SAMPLE_COLORS[sample],
                        linewidth=1.6,
                        zorder=3.0,
                    )
                ax.axvline(center, color="0.25", linewidth=0.8)
                ax.set_xlim(center - half_width, center + half_width)
                ax.set_ylim(0.0, 1.08)
                ax.set_yticks([])
            else:
                x_center = centers[column]
                y_center = centers[row]
                for sample in samples:
                    covariance_2d = selected[sample][
                        np.ix_([column, row], [column, row])
                    ]
                    add_confidence_ellipse(
                        ax,
                        covariance_2d,
                        np.asarray([x_center, y_center]),
                        level=CONFIDENCE_LEVELS[1],
                        facecolor=lighter_tone(
                            SAMPLE_COLORS[sample],
                            OUTER_TONE_WHITE_MIX,
                        ),
                        edgecolor=SAMPLE_COLORS[sample],
                        linewidth=1.0,
                        alpha=0.40,
                        zorder=1.0,
                    )
                    add_confidence_ellipse(
                        ax,
                        covariance_2d,
                        np.asarray([x_center, y_center]),
                        level=CONFIDENCE_LEVELS[0],
                        facecolor=lighter_tone(
                            SAMPLE_COLORS[sample],
                            INNER_TONE_WHITE_MIX,
                        ),
                        edgecolor=SAMPLE_COLORS[sample],
                        linewidth=1.8,
                        alpha=0.58,
                        zorder=2.0,
                    )
                ax.scatter(
                    x_center,
                    y_center,
                    marker="+",
                    color="black",
                    s=28,
                    linewidths=1.0,
                    zorder=5,
                )
                ax.set_xlim(
                    x_center - range_sigma * limit_errors[column],
                    x_center + range_sigma * limit_errors[column],
                )
                ax.set_ylim(
                    y_center - range_sigma * limit_errors[row],
                    y_center + range_sigma * limit_errors[row],
                )

            ax.grid(linewidth=0.4)
            ax.tick_params(labelsize=9)
            if row < n_parameters - 1:
                ax.tick_params(labelbottom=False)
            if column > 0 and row != column:
                ax.tick_params(labelleft=False)
            if row == n_parameters - 1:
                ax.set_xlabel(PARAMETER_LABELS[parameters[column]], fontsize=16)
            if column == 0 and row > 0:
                ax.set_ylabel(PARAMETER_LABELS[parameters[row]], fontsize=16)

    sample_handles = [
        Patch(
            facecolor=SAMPLE_COLORS[sample],
            edgecolor="none",
            label=SAMPLE_LABELS[sample],
        )
        for sample in samples
    ]
    level_handles = [
        Patch(
            facecolor=lighter_tone(
                "#555555",
                INNER_TONE_WHITE_MIX
                if index == 0
                else OUTER_TONE_WHITE_MIX,
            ),
            edgecolor="none",
            label=f"{100.0 * level:.1f}% CL",
        )
        for index, level in enumerate(CONFIDENCE_LEVELS)
    ]
    fig.legend(
        handles=[*sample_handles],#, *level_handles],
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        frameon=False,
        fontsize=18,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.98,
        wspace=0.0,
        hspace=0.0,
    )
    return fig


def marginalized_error_rows(
    covariances: Mapping[str, np.ndarray],
    fiducials: Mapping[str, float],
    samples: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for sample in samples:
        sigma = np.sqrt(np.diag(covariances[sample]))
        for index, parameter in enumerate(PARAMETERS):
            rows.append(
                {
                    "sample": sample,
                    "parameter": parameter,
                    "fiducial": f"{fiducials[parameter]:.12e}",
                    "sigma_marginalized": f"{sigma[index]:.12e}",
                }
            )
    return rows


def check_outputs(
    paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise PlotError(
            "output already exists: "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace it"
        )


def write_errors_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "parameter",
                "fiducial",
                "sigma_marginalized",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    suffix = path.suffix or ".png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=suffix,
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    fiducials = parse_fiducials(args.fiducial)
    check_outputs(
        [args.output, args.errors_output],
        overwrite=args.overwrite,
    )
    covariances = load_covariances(
        args.fisher_root,
        args.samples,
        args.snapnum,
        args.mnu_order,
        args.precision_correction,
    )
    figure = plot_triangle(
        covariances,
        fiducials,
        args.params,
        args.samples,
        range_sigma=args.range_sigma,
    )
    try:
        save_figure(figure, args.output, dpi=args.dpi)
    finally:
        plt.close(figure)
    write_errors_csv(
        args.errors_output,
        marginalized_error_rows(
            covariances,
            fiducials,
            args.samples,
        ),
    )
    return args.output, args.errors_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    finish_args(args, parser)
    try:
        plot_path, errors_path = run(args)
    except (PlotError, OSError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(f"[write] {plot_path}")
    print(f"[write] {errors_path}")
    print("[done] plotted marginalized constraints from the full Fisher inverse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
