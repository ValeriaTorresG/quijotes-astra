import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher import fiducial_covariance as covariance


K = np.array([0.01, 0.02])
COLUMNS = (
    "Pk0_raw_Mpc3_h3",
    "Pk0_shot_subtracted_Mpc3_h3",
    "Pk2_Mpc3_h3",
    "Pk4_Mpc3_h3",
)
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
    *COLUMNS,
]


def _write_spectrum(
    pk_root,
    dataset,
    simulation_id,
    values,
    *,
    k=K,
    sample="all",
):
    directory = "matter" if sample == "all" else "env"
    suffix = "" if sample == "all" else f"_{sample}"
    path = (
        pk_root
        / directory
        / (
            f"{dataset}_sim{simulation_id:03d}_snap003"
            f"{suffix}_pk.csv"
        )
    )
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
                "tracer": "FoF_halo",
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
                "binning_mode": "native_pylians",
                "k_fundamental_h_Mpc": "6.283185307180e-03",
                "k_bin_width_h_Mpc": "",
                "k_bin_index": "",
                "k_bin_min_h_Mpc": "",
                "k_bin_max_h_Mpc": "",
            }
            for column in COLUMNS:
                row[column] = f"{values[column][index]:.12e}"
            writer.writerow(row)
    return path


def _values(shot_subtracted, *, pk2=None):
    shot_subtracted = np.asarray(shot_subtracted, dtype=np.float64)
    if pk2 is None:
        pk2 = np.zeros_like(shot_subtracted)
    return {
        "Pk0_raw_Mpc3_h3": shot_subtracted + 10.0,
        "Pk0_shot_subtracted_Mpc3_h3": shot_subtracted,
        "Pk2_Mpc3_h3": np.asarray(pk2, dtype=np.float64),
        "Pk4_Mpc3_h3": np.zeros_like(shot_subtracted),
    }


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_sample_covariance_uses_only_standard_fiducial_realizations(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    for simulation_id, vector in enumerate(vectors):
        _write_spectrum(
            pk_root,
            "fiducial",
            simulation_id,
            _values(vector),
        )
        _write_spectrum(
            pk_root,
            "fiducial_ZA",
            simulation_id,
            _values(vector + 1000.0),
        )

    assert (
        covariance.main(
            [
                "--pk-root",
                str(pk_root),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    mapping = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_data_vector.csv"
    )
    matrix = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_covariance.csv"
    )
    realizations = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_realizations.csv"
    )
    expected = np.cov(vectors, rowvar=False, ddof=1)
    np.testing.assert_allclose(
        [float(row["mean"]) for row in mapping],
        np.mean(vectors, axis=0),
    )
    np.testing.assert_allclose(
        [
            [float(row["v000000"]), float(row["v000001"])]
            for row in matrix
        ],
        expected,
    )
    assert {row["dataset"] for row in mapping} == {"fiducial"}
    assert {row["n_realizations"] for row in mapping} == {"3"}
    assert {row["covariance_ddof"] for row in mapping} == {"1"}
    assert [row["simulation_id"] for row in realizations] == ["0", "1", "2"]


def test_multiple_observables_keep_block_order_and_cross_covariance(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    p0 = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 7.0], [8.0, 9.0]])
    p2 = np.array([[5.0, 1.0], [4.0, 3.0], [2.0, 6.0], [1.0, 8.0]])
    for simulation_id in range(len(p0)):
        _write_spectrum(
            pk_root,
            "fiducial",
            simulation_id,
            _values(p0[simulation_id], pk2=p2[simulation_id]),
        )

    assert (
        covariance.main(
            [
                "--pk-root",
                str(pk_root),
                "--output-root",
                str(output_root),
                "--columns",
                "Pk0_shot_subtracted_Mpc3_h3",
                "Pk2_Mpc3_h3",
            ]
        )
        == 0
    )
    mapping = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_data_vector.csv"
    )
    matrix_rows = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_covariance.csv"
    )
    expected_vectors = np.concatenate((p0, p2), axis=1)
    expected_covariance = np.cov(
        expected_vectors,
        rowvar=False,
        ddof=1,
    )
    labels = [f"v{index:06d}" for index in range(4)]
    actual_covariance = np.asarray(
        [
            [float(row[label]) for label in labels]
            for row in matrix_rows
        ]
    )
    np.testing.assert_allclose(actual_covariance, expected_covariance)
    assert [row["observable"] for row in mapping] == [
        "Pk0_shot_subtracted_Mpc3_h3",
        "Pk0_shot_subtracted_Mpc3_h3",
        "Pk2_Mpc3_h3",
        "Pk2_Mpc3_h3",
    ]
    assert [row["k_index"] for row in mapping] == ["0", "1", "0", "1"]


def test_covariance_rejects_misaligned_k_grids(tmp_path, capsys):
    pk_root = tmp_path / "pk"
    _write_spectrum(
        pk_root,
        "fiducial",
        0,
        _values([1.0, 2.0]),
    )
    _write_spectrum(
        pk_root,
        "fiducial",
        1,
        _values([3.0, 4.0]),
        k=np.array([0.01, 0.03]),
    )

    status = covariance.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(tmp_path / "results"),
        ]
    )
    assert status == 1
    assert "k grid mismatch" in capsys.readouterr().err


def test_max_bins_reduces_each_covariance_observable_block(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    k = np.array([0.01, 0.02, 0.03])
    for simulation_id in range(4):
        vector = np.array(
            [simulation_id + 1.0, 2.0 * simulation_id + 1.0, 7.0]
        )
        _write_spectrum(
            pk_root,
            "fiducial",
            simulation_id,
            _values(vector),
            k=k,
        )

    status = covariance.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(output_root),
            "--max-bins",
            "2",
        ]
    )
    assert status == 0
    mapping = _read_rows(
        output_root
        / "covariance"
        / "matter"
        / "fiducial_snap003_data_vector.csv"
    )
    assert len(mapping) == 2
    assert [float(row["k_h_Mpc"]) for row in mapping] == [0.01, 0.02]
    assert {row["data_vector_size"] for row in mapping} == {"2"}


def test_combined_environment_covariance_keeps_cross_blocks(tmp_path):
    pk_root = tmp_path / "pk"
    output_root = tmp_path / "results"
    samples = ("void", "sheet", "filament", "knot")
    vectors = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [2.0, 1.0, 5.0, 7.0],
            [4.0, 3.0, 1.0, 6.0],
            [7.0, 5.0, 2.0, 1.0],
            [9.0, 8.0, 7.0, 3.0],
        ]
    )
    k = np.array([0.01])
    for simulation_id, vector in enumerate(vectors):
        for sample, value in zip(samples, vector):
            _write_spectrum(
                pk_root,
                "fiducial",
                simulation_id,
                _values([value]),
                sample=sample,
                k=k,
            )

    status = covariance.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(output_root),
            "--samples",
            *samples,
            "--combine-environments",
        ]
    )
    assert status == 0
    mapping = _read_rows(
        output_root
        / "covariance"
        / "combined"
        / "fiducial_snap003_combined_data_vector.csv"
    )
    assert [row["observable"] for row in mapping] == [
        f"{sample}__Pk0_shot_subtracted_Mpc3_h3"
        for sample in samples
    ]
    matrix_rows = _read_rows(
        output_root
        / "covariance"
        / "combined"
        / "fiducial_snap003_combined_covariance.csv"
    )
    labels = [f"v{index:06d}" for index in range(4)]
    actual = np.asarray(
        [[float(row[label]) for label in labels] for row in matrix_rows]
    )
    np.testing.assert_allclose(actual, np.cov(vectors, rowvar=False, ddof=1))
    assert actual[0, 1] != 0.0


def test_explicit_ids_must_exist_for_every_selected_sample(tmp_path, capsys):
    pk_root = tmp_path / "pk"
    for simulation_id in (0, 1):
        _write_spectrum(
            pk_root,
            "fiducial",
            simulation_id,
            _values([1.0 + simulation_id, 2.0]),
        )
    _write_spectrum(
        pk_root,
        "fiducial",
        0,
        _values([3.0, 4.0]),
        sample="void",
    )

    status = covariance.main(
        [
            "--pk-root",
            str(pk_root),
            "--output-root",
            str(tmp_path / "results"),
            "--simulation-ids",
            "0",
            "1",
            "--samples",
            "all",
            "void",
        ]
    )
    assert status == 1
    assert "void missing IDs 1" in capsys.readouterr().err
