import csv
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher import central_differences as central
from fisher import forward_differences as forward
from fisher.derivative_utils import DEFAULT_OBSERVABLE_COLUMNS


K = np.array([0.01, 0.02], dtype=np.float64)
FIELDNAMES = [
    "dataset",
    "simulation_id",
    "snapshot",
    "redshift",
    "sample",
    "tracer",
    "n_k_shells",
    "grid",
    "mass_assignment",
    "los_axis",
    "threads",
    "k_min_h_Mpc",
    "k_max_h_Mpc",
    "k_nyquist_h_Mpc",
    "k_h_Mpc",
    "Nmodes",
    "binning_mode",
    "k_fundamental_h_Mpc",
    "k_bin_width_h_Mpc",
    "k_bin_index",
    "k_bin_min_h_Mpc",
    "k_bin_max_h_Mpc",
    *DEFAULT_OBSERVABLE_COLUMNS,
]


def _write_spectrum(
    pk_root,
    dataset,
    simulation_id,
    values,
    *,
    k=K,
    sample="all",
    binned=False,
):
    if sample == "all":
        path = (
            pk_root
            / "matter"
            / f"{dataset}_sim{simulation_id:03d}_snap003_pk.csv"
        )
        tracer = "FoF_halo"
    else:
        path = (
            pk_root
            / "env"
            / (
                f"{dataset}_sim{simulation_id:03d}_snap003_"
                f"{sample}_pk.csv"
            )
        )
        tracer = "uniform_random" if sample == "random_void" else "FoF_halo"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for index, k_value in enumerate(k):
            row = {
                "dataset": dataset,
                "simulation_id": simulation_id,
                "snapshot": 3,
                "redshift": "5.000000000000e-01",
                "sample": sample,
                "tracer": tracer,
                "n_k_shells": len(k),
                "grid": 512,
                "mass_assignment": "CIC",
                "los_axis": 0,
                "threads": 1,
                "k_min_h_Mpc": "8.000000000000e-03",
                "k_max_h_Mpc": "5.000000000000e-01",
                "k_nyquist_h_Mpc": "1.608495438638e+00",
                "k_h_Mpc": f"{k_value:.12e}",
                "Nmodes": (index + 1) * 10,
                "binning_mode": (
                    "fixed_nmodes_weighted"
                    if binned
                    else "native_pylians"
                ),
                "k_fundamental_h_Mpc": "6.283185307180e-03",
                "k_bin_width_h_Mpc": (
                    "1.000000000000e-02" if binned else ""
                ),
                "k_bin_index": index if binned else "",
                "k_bin_min_h_Mpc": (
                    f"{index * 0.01:.12e}" if binned else ""
                ),
                "k_bin_max_h_Mpc": (
                    f"{(index + 1) * 0.01:.12e}" if binned else ""
                ),
            }
            for column_index, column in enumerate(
                DEFAULT_OBSERVABLE_COLUMNS,
                start=1,
            ):
                row[column] = f"{values[index] * column_index:.12e}"
            writer.writerow(row)
    return path


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _central_dataset_values():
    return {
        parameter.minus_dataset: parameter.minus_value
        for parameter in central.CENTRAL_PARAMETERS
    } | {
        parameter.plus_dataset: parameter.plus_value
        for parameter in central.CENTRAL_PARAMETERS
    }


def test_central_derivatives_are_paired_before_mean_and_dispersion(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    for simulation_id, slope in ((0, 1.0), (1, 3.0)):
        baseline = simulation_id * 10_000.0 + np.array([10.0, 20.0])
        for dataset, parameter_value in _central_dataset_values().items():
            values = baseline + slope * parameter_value
            _write_spectrum(pk_root, dataset, simulation_id, values)
            _write_spectrum(
                pk_root,
                dataset,
                simulation_id,
                values + 100.0,
                sample="void",
            )

    status = central.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(output_root),
        ]
    )
    assert status == 0

    sim1 = _read_rows(
        output_root
        / "central"
        / "matter"
        / "sim001_snap003_derivatives.csv"
    )
    for parameter in ("Om", "h", "ns", "s8"):
        for observable_index, observable in enumerate(
            DEFAULT_OBSERVABLE_COLUMNS,
            start=1,
        ):
            column = f"d_{observable}_d{parameter}"
            np.testing.assert_allclose(
                [float(row[column]) for row in sim1],
                3.0 * observable_index,
                rtol=1.0e-7,
            )

    summary = _read_rows(
        output_root
        / "central"
        / "matter"
        / "snap003_derivatives_mean_std.csv"
    )
    derivative = "d_Pk0_raw_Mpc3_h3_dOm"
    assert {row["n_realizations"] for row in summary} == {"2"}
    assert {row["realization_ids"] for row in summary} == {"0,1"}
    np.testing.assert_allclose(
        [float(row[f"mean_{derivative}"]) for row in summary],
        2.0,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        [float(row[f"std_{derivative}"]) for row in summary],
        np.sqrt(2.0),
        rtol=1.0e-7,
    )
    environment = _read_rows(
        output_root
        / "central"
        / "env"
        / "sim001_snap003_void_derivatives.csv"
    )
    assert {row["sample"] for row in environment} == {"void"}
    np.testing.assert_allclose(
        [float(row[derivative]) for row in environment],
        3.0,
        rtol=1.0e-7,
    )


def test_forward_stencils_use_0_h_2h_4h_and_positive_last_weight(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    masses = dict(forward.MNU_DATASETS)
    for simulation_id, slope in ((0, 1.0), (1, 2.0)):
        baseline = simulation_id * 10_000.0 + np.array([7.0, 11.0])
        for dataset, mass in masses.items():
            polynomial = 2.0 * mass + 3.0 * mass**2 + 4.0 * mass**3
            values = baseline + slope * polynomial
            _write_spectrum(pk_root, dataset, simulation_id, values)

    status = forward.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(output_root),
        ]
    )
    assert status == 0

    sim0 = _read_rows(
        output_root
        / "forward"
        / "matter"
        / "sim000_snap003_derivatives.csv"
    )
    expected = {1: 2.34, 2: 1.92, 3: 2.0}
    for order, value in expected.items():
        column = (
            "d_Pk0_raw_Mpc3_h3_dMnu_"
            f"forward_order{order}"
        )
        np.testing.assert_allclose(
            [float(row[column]) for row in sim0],
            value,
            rtol=1.0e-9,
        )

    constant = np.full(3, 42.0)
    estimates = forward.forward_stencils(
        constant,
        constant,
        constant,
        constant,
    )
    for estimate in estimates:
        np.testing.assert_array_equal(estimate, 0.0)


def test_missing_realization_counterpart_fails_in_strict_mode(
    tmp_path,
    capsys,
):
    pk_root = tmp_path / "pk"
    for dataset, parameter_value in _central_dataset_values().items():
        _write_spectrum(
            pk_root,
            dataset,
            0,
            np.array([1.0, 2.0]) + parameter_value,
        )
    _write_spectrum(pk_root, "Om_m", 1, np.array([3.0, 4.0]))

    status = central.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert status == 1
    assert "do not have identical paired products" in capsys.readouterr().err


def test_k_grid_mismatch_is_rejected_without_interpolation(tmp_path, capsys):
    pk_root = tmp_path / "pk"
    for dataset, mass in forward.MNU_DATASETS:
        k = np.array([0.01, 0.03]) if dataset == "Mnu_ppp" else K
        _write_spectrum(
            pk_root,
            dataset,
            0,
            np.array([1.0, 2.0]) + mass,
            k=k,
        )

    status = forward.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert status == 1
    error = capsys.readouterr().err
    assert "k grid mismatch" in error
    assert "never interpolated" in error


def test_single_realization_has_undefined_sample_dispersion(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    for dataset, mass in forward.MNU_DATASETS:
        _write_spectrum(
            pk_root,
            dataset,
            4,
            np.array([1.0, 2.0]) + mass,
        )

    assert (
        forward.main(
            [
                "--pk-root",
                str(pk_root),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    summary = _read_rows(
        output_root
        / "forward"
        / "matter"
        / "snap003_derivatives_mean_std.csv"
    )
    assert summary[0]["n_realizations"] == "1"
    assert summary[0]["dispersion_ddof"] == "1"
    std_columns = [
        column for column in summary[0] if column.startswith("std_")
    ]
    assert std_columns
    assert all(row[column] == "" for row in summary for column in std_columns)


def test_fixed_bin_metadata_is_validated_and_preserved(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    for dataset, mass in forward.MNU_DATASETS:
        _write_spectrum(
            pk_root,
            dataset,
            2,
            np.array([5.0, 7.0]) + 4.0 * mass,
            binned=True,
        )

    assert (
        forward.main(
            [
                "--pk-root",
                str(pk_root),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    rows = _read_rows(
        output_root
        / "forward"
        / "matter"
        / "sim002_snap003_derivatives.csv"
    )
    assert {row["binning_mode"] for row in rows} == {
        "fixed_nmodes_weighted"
    }
    assert [row["k_bin_index"] for row in rows] == ["0", "1"]
    assert [float(row["k_bin_min_h_Mpc"]) for row in rows] == [0.0, 0.01]
    derivative = "d_Pk0_raw_Mpc3_h3_dMnu_forward_order3"
    np.testing.assert_allclose(
        [float(row[derivative]) for row in rows],
        4.0,
        rtol=1.0e-10,
    )
