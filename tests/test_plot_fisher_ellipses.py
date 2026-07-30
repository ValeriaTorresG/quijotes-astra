import csv
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher import plot_fisher_ellipses as plotter


def _write_fisher(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["parameter", *plotter.PARAMETERS],
        )
        writer.writeheader()
        for index, parameter in enumerate(plotter.PARAMETERS):
            row = {"parameter": parameter}
            row.update(
                {
                    column: matrix[index, column_index]
                    for column_index, column in enumerate(plotter.PARAMETERS)
                }
            )
            writer.writerow(row)


def test_marginalized_covariance_inverts_full_fisher():
    fisher = np.asarray(
        [
            [4.0, 1.0],
            [1.0, 2.0],
        ]
    )
    actual = plotter.marginalized_covariance(fisher, context="test")
    np.testing.assert_allclose(actual, np.linalg.inv(fisher))
    assert np.sqrt(actual[0, 0]) > 1.0 / np.sqrt(fisher[0, 0])


def test_cli_writes_triangle_plot_and_marginalized_errors(tmp_path):
    fisher_root = tmp_path / "fisher"
    matrix = np.diag([400.0, 225.0, 100.0, 625.0, 25.0])
    matrix[0, 1] = matrix[1, 0] = 20.0
    for sample in plotter.SAMPLES:
        _write_fisher(
            plotter.fisher_path(fisher_root, sample, 3, 3, "none"),
            matrix,
        )
    output = tmp_path / "ellipses.png"
    errors = tmp_path / "errors.csv"

    status = plotter.main(
        [
            "--fisher-root",
            str(fisher_root),
            "--params",
            "Om",
            "h",
            "Mnu",
            "--output",
            str(output),
            "--errors-output",
            str(errors),
        ]
    )

    assert status == 0
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with errors.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(plotter.SAMPLES) * len(plotter.PARAMETERS)
    assert {row["sample"] for row in rows} == set(plotter.SAMPLES)
