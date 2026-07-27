import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher import fisher_matrix


OBSERVABLE = "Pk0_shot_subtracted_Mpc3_h3"
K = np.array([0.01, 0.02])
NMODES = np.array([10, 20])
PARAMETERS = ("Om", "h", "ns", "s8")


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_covariance_products(root, matrix, *, n_realizations=10, k=K):
    size = len(k)
    labels = [f"v{index:06d}" for index in range(size)]
    base = root / "covariance" / "matter" / "fiducial_snap003"
    mapping_fields = [
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
    ]
    mapping_rows = [
        {
            "data_vector_index": label,
            "vector_position": index,
            "dataset": "fiducial",
            "n_realizations": n_realizations,
            "snapshot": 3,
            "category": "matter",
            "sample": "all",
            "tracer": "FoF_halo",
            "data_vector_order": "observable_then_k",
            "observable": OBSERVABLE,
            "k_index": index,
            "k_h_Mpc": f"{k[index]:.17e}",
            "Nmodes": int(NMODES[index]),
            "binning_mode": "native_pylians",
            "k_bin_width_h_Mpc": "",
            "k_bin_index": "",
            "k_bin_min_h_Mpc": "",
            "k_bin_max_h_Mpc": "",
            "covariance_ddof": 1,
            "data_vector_size": size,
        }
        for index, label in enumerate(labels)
    ]
    _write_csv(
        Path(f"{base}_data_vector.csv"),
        mapping_fields,
        mapping_rows,
    )

    matrix_fields = [
        "data_vector_index",
        "n_realizations",
        "covariance_ddof",
        *labels,
    ]
    matrix_rows = []
    for index, label in enumerate(labels):
        row = {
            "data_vector_index": label,
            "n_realizations": n_realizations,
            "covariance_ddof": 1,
        }
        row.update(
            {
                column_label: f"{matrix[index, column]:.17e}"
                for column, column_label in enumerate(labels)
            }
        )
        matrix_rows.append(row)
    _write_csv(
        Path(f"{base}_covariance.csv"),
        matrix_fields,
        matrix_rows,
    )

    realization_fields = [
        "realization_order",
        "simulation_id",
        "dataset",
        "snapshot",
        "category",
        "sample",
    ]
    _write_csv(
        Path(f"{base}_realizations.csv"),
        realization_fields,
        [
            {
                "realization_order": index,
                "simulation_id": index,
                "dataset": "fiducial",
                "snapshot": 3,
                "category": "matter",
                "sample": "all",
            }
            for index in range(n_realizations)
        ],
    )


def _write_derivative_summaries(root, central_values, mnu_values, *, k=K):
    base_fields = [
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
    ]
    central_columns = [
        f"mean_d_{OBSERVABLE}_d{parameter}"
        for parameter in PARAMETERS
    ]
    central_rows = []
    for k_index, k_value in enumerate(k):
        row = {
            "snapshot": 3,
            "sample": "all",
            "tracer": "FoF_halo",
            "n_realizations": 500,
            "k_h_Mpc": f"{k_value:.17e}",
            "Nmodes": int(NMODES[k_index]),
            "binning_mode": "native_pylians",
            "k_bin_width_h_Mpc": "",
            "k_bin_index": "",
            "k_bin_min_h_Mpc": "",
            "k_bin_max_h_Mpc": "",
        }
        row.update(
            {
                column: f"{central_values[k_index, index]:.17e}"
                for index, column in enumerate(central_columns)
            }
        )
        central_rows.append(row)
    _write_csv(
        root
        / "central"
        / "matter"
        / "snap003_derivatives_mean_std.csv",
        [*base_fields, *central_columns],
        central_rows,
    )

    mnu_columns = [
        f"mean_d_{OBSERVABLE}_dMnu_forward_order{order}"
        for order in (1, 2, 3)
    ]
    forward_rows = []
    for k_index, k_value in enumerate(k):
        row = {
            "snapshot": 3,
            "sample": "all",
            "tracer": "FoF_halo",
            "n_realizations": 500,
            "k_h_Mpc": f"{k_value:.17e}",
            "Nmodes": int(NMODES[k_index]),
            "binning_mode": "native_pylians",
            "k_bin_width_h_Mpc": "",
            "k_bin_index": "",
            "k_bin_min_h_Mpc": "",
            "k_bin_max_h_Mpc": "",
        }
        row.update(
            {
                column: f"{mnu_values[order - 1, k_index]:.17e}"
                for order, column in enumerate(mnu_columns, start=1)
            }
        )
        forward_rows.append(row)
    _write_csv(
        root
        / "forward"
        / "matter"
        / "snap003_derivatives_mean_std.csv",
        [*base_fields, *mnu_columns],
        forward_rows,
    )


def _prepare_inputs(root, matrix, *, n_realizations=10, covariance_k=K):
    central = np.array(
        [
            [2.0, 0.0, 1.0, 4.0],
            [0.0, 3.0, 1.0, -3.0],
        ]
    )
    mnu = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    _write_covariance_products(
        root,
        matrix,
        n_realizations=n_realizations,
        k=covariance_k,
    )
    _write_derivative_summaries(root, central, mnu)
    return central, mnu


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_fisher_matches_labelled_hartlap_corrected_matrix(tmp_path):
    root = tmp_path / "fisher"
    output = tmp_path / "output"
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    central, mnu = _prepare_inputs(root, covariance)

    assert (
        fisher_matrix.main(
            [
                "--fisher-root",
                str(root),
                "--output-root",
                str(output),
                "--precision-correction",
                "hartlap",
            ]
        )
        == 0
    )
    derivatives = np.column_stack((central, mnu[2]))
    hartlap = (10.0 - 2.0 - 2.0) / (10.0 - 1.0)
    expected = hartlap * (
        derivatives.T @ np.linalg.solve(covariance, derivatives)
    )

    rows = _read_rows(
        output
        / "matter"
        / (
            "fiducial_snap003_params_Om-h-ns-s8-Mnu"
            "_mnu_order3_precision_hartlap_fisher.csv"
        )
    )
    actual = np.asarray(
        [
            [float(row[parameter]) for parameter in fisher_matrix.DEFAULT_PARAMETERS]
            for row in rows
        ]
    )
    np.testing.assert_allclose(actual, expected)
    assert [row["parameter"] for row in rows] == list(
        fisher_matrix.DEFAULT_PARAMETERS
    )
    diagnostics = _read_rows(
        output
        / "matter"
        / (
            "fiducial_snap003_params_Om-h-ns-s8-Mnu"
            "_mnu_order3_precision_hartlap_diagnostics.csv"
        )
    )[0]
    assert float(diagnostics["precision_factor"]) == hartlap
    assert diagnostics["mnu_forward_order"] == "3"


def test_mnu_order_and_parameter_subset_select_exact_derivatives(tmp_path):
    root = tmp_path / "fisher"
    covariance = np.array([[2.0, 0.25], [0.25, 3.0]])
    central, mnu = _prepare_inputs(root, covariance)
    output = tmp_path / "order1"

    assert (
        fisher_matrix.main(
            [
                "--fisher-root",
                str(root),
                "--output-root",
                str(output),
                "--parameters",
                "Om",
                "Mnu",
                "--mnu-order",
                "1",
            ]
        )
        == 0
    )
    derivatives = np.column_stack((central[:, 0], mnu[0]))
    expected = derivatives.T @ np.linalg.solve(covariance, derivatives)
    rows = _read_rows(
        output
        / "matter"
        / (
            "fiducial_snap003_params_Om-Mnu"
            "_mnu_order1_precision_none_fisher.csv"
        )
    )
    actual = np.asarray(
        [[float(row["Om"]), float(row["Mnu"])] for row in rows]
    )
    np.testing.assert_allclose(actual, expected)


def test_output_names_distinguish_parameters_and_precision_correction(tmp_path):
    root = tmp_path / "fisher"
    output = tmp_path / "output"
    covariance = np.array([[2.0, 0.25], [0.25, 3.0]])
    _prepare_inputs(root, covariance)

    assert (
        fisher_matrix.main(
            [
                "--fisher-root",
                str(root),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert (
        fisher_matrix.main(
            [
                "--fisher-root",
                str(root),
                "--output-root",
                str(output),
                "--parameters",
                "Om",
                "Mnu",
                "--precision-correction",
                "hartlap",
            ]
        )
        == 0
    )

    fisher_outputs = sorted((output / "matter").glob("*_fisher.csv"))
    assert [path.name for path in fisher_outputs] == [
        (
            "fiducial_snap003_params_Om-Mnu"
            "_mnu_order3_precision_hartlap_fisher.csv"
        ),
        (
            "fiducial_snap003_params_Om-h-ns-s8-Mnu"
            "_mnu_order3_precision_none_fisher.csv"
        ),
    ]


def test_hartlap_requires_more_than_p_plus_two_realizations(tmp_path, capsys):
    root = tmp_path / "fisher"
    covariance = np.array([[2.0, 0.1], [0.1, 1.0]])
    _prepare_inputs(root, covariance, n_realizations=4)

    status = fisher_matrix.main(
        [
            "--fisher-root",
            str(root),
            "--output-root",
            str(tmp_path / "output"),
            "--precision-correction",
            "hartlap",
        ]
    )
    assert status == 1
    assert "requires N_realizations > data_vector_size + 2" in (
        capsys.readouterr().err
    )
    assert fisher_matrix._precision_factor("hartlap", 10, 2, 0) == 0.6


def test_fisher_rejects_derivative_k_mismatch(tmp_path, capsys):
    root = tmp_path / "fisher"
    covariance = np.array([[2.0, 0.1], [0.1, 1.0]])
    _prepare_inputs(
        root,
        covariance,
        covariance_k=np.array([0.01, 0.03]),
    )

    status = fisher_matrix.main(
        [
            "--fisher-root",
            str(root),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )
    assert status == 1
    assert "k mismatch" in capsys.readouterr().err


def test_fisher_refuses_singular_covariance_without_pseudoinverse(
    tmp_path,
    capsys,
):
    root = tmp_path / "fisher"
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    _prepare_inputs(root, covariance)

    status = fisher_matrix.main(
        [
            "--fisher-root",
            str(root),
            "--output-root",
            str(tmp_path / "output"),
            "--precision-correction",
            "none",
        ]
    )
    assert status == 1
    assert "refusing to use a pseudoinverse" in capsys.readouterr().err
