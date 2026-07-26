import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pk import compute_power_spectra as cps


DATASET = "fiducial"
SIMULATION_ID = 7
SNAPSHOT = 3
K_VALUES = np.array([0.005, 0.01, 0.02, 0.6])
MATTER_PK0 = np.array([100.0, 100.0, 200.0, 100.0])
RATIOS = {
    "void": 2.0,
    "sheet": 3.0,
    "filament": 4.0,
    "knot": 5.0,
    cps.RANDOM_VOID_SAMPLE: 6.0,
}


def _write_plot_csv(
    path,
    sample,
    pk0,
    *,
    k=K_VALUES,
    metadata_overrides=None,
):
    fieldnames = list(cps.PLOT_METADATA_COLUMNS) + [
        "sample",
        "n_k_shells",
        "k_h_Mpc",
        "Pk0_raw_Mpc3_h3",
        "Pk0_shot_subtracted_Mpc3_h3",
    ]
    common = {
        "dataset": DATASET,
        "simulation_id": str(SIMULATION_ID),
        "snapshot": str(SNAPSHOT),
        "redshift": "0.5",
        "fof_catalog_sha256": "f" * 64,
        "box_size_Mpc_h": "1000",
        "grid": "512",
        "mass_assignment": "CIC",
        "los_axis": "0",
        "threads": "1",
        "k_min_h_Mpc": "0.008",
        "k_max_h_Mpc": "0.5",
        "k_nyquist_h_Mpc": "1.6084954386379742",
        "sample": sample,
        "n_k_shells": str(len(k)),
    }
    if sample == "all":
        common.update(
            {column: "" for column in cps.PLOT_ENVIRONMENT_METADATA_COLUMNS}
        )
        common["astra_probability_sha256"] = ""
    else:
        common.update(
            {
                "n_astra_iterations": "45",
                "astra_release": "QUIJOTES",
                "astra_random_seed": "42",
                "astra_r_lower": "-0.25",
                "astra_r_med": "0.25",
                "astra_r_upper": "0.65",
                "astra_periodic": "True",
                "astra_box_min_Mpc_h": "0",
                "astra_box_max_Mpc_h": "1000",
                "astra_probability_sha256": (
                    "" if sample == cps.RANDOM_VOID_SAMPLE else "a" * 64
                ),
            }
        )
    if metadata_overrides:
        common.update(metadata_overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k_value, pk0_value in zip(k, pk0):
            row = dict(common)
            row.update(
                {
                    "k_h_Mpc": repr(float(k_value)),
                    "Pk0_raw_Mpc3_h3": repr(float(pk0_value)),
                    # Deliberately unrelated values: the normalized plot must
                    # use the raw monopole, exactly as the notebook does.
                    "Pk0_shot_subtracted_Mpc3_h3": "-999",
                }
            )
            writer.writerow(row)


@pytest.fixture
def plot_csvs(tmp_path):
    matter = tmp_path / "matter" / "all.csv"
    _write_plot_csv(matter, "all", MATTER_PK0)
    environments = {}
    for sample, ratio in RATIOS.items():
        path = tmp_path / "env" / f"{sample}.csv"
        _write_plot_csv(path, sample, MATTER_PK0 * ratio)
        environments[sample] = path
    return matter, environments


def test_normalized_curves_use_raw_monopole_and_common_notebook_mask(plot_csvs):
    matter, environments = plot_csvs
    spectra = cps.build_normalized_power_spectra(
        matter,
        environments,
        dataset=DATASET,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        kmin=0.008,
        kmax=0.5,
    )

    np.testing.assert_array_equal(spectra.k, [0.01, 0.02])
    for sample, expected_ratio in RATIOS.items():
        np.testing.assert_array_equal(
            spectra.ratios[sample],
            [expected_ratio, expected_ratio],
        )


def test_normalized_plot_has_notebook_style_and_is_atomic(
    plot_csvs,
    tmp_path,
    monkeypatch,
):
    matter, environments = plot_csvs
    mpl_config = tmp_path / "mplconfig"
    mpl_config.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_config))
    spectra = cps.build_normalized_power_spectra(
        matter,
        environments,
        dataset=DATASET,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        kmin=0.008,
        kmax=0.5,
    )

    fig, ax = cps.draw_normalized_power_spectra(spectra)
    try:
        assert tuple(fig.get_size_inches()) == (8.0, 5.0)
        assert fig.dpi == pytest.approx(360.0)
        assert fig.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
        assert ax.get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert [line.get_label() for line in ax.lines] == [
            "Voids",
            "Sheets",
            "Filaments",
            "Knots",
            "Random voids",
            "All",
        ]
        assert [line.get_color() for line in ax.lines[:5]] == [
            "#17becf",
            "orange",
            "limegreen",
            "magenta",
            "#ffd166",
        ]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    output = tmp_path / "env" / "normalized.png"
    counters = cps.Counters()
    args = SimpleNamespace(kmin=0.008, kmax=0.5, overwrite=False)
    cps.maybe_write_normalized_plot(
        matter_path=matter,
        environment_paths=environments,
        output_path=output,
        dataset=DATASET,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        args=args,
        counters=counters,
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert counters.plots_written == 1
    assert not list(output.parent.glob(".*.png"))

    cps.maybe_write_normalized_plot(
        matter_path=matter,
        environment_paths=environments,
        output_path=output,
        dataset=DATASET,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        args=args,
        counters=counters,
    )
    assert counters.plots_skipped == 1

    output.write_bytes(b"not a PNG")
    cps.maybe_write_normalized_plot(
        matter_path=matter,
        environment_paths=environments,
        output_path=output,
        dataset=DATASET,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        args=args,
        counters=counters,
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert counters.plots_written == 2


def test_normalized_plot_rejects_mismatched_k_shells(plot_csvs):
    matter, environments = plot_csvs
    _write_plot_csv(
        environments["sheet"],
        "sheet",
        MATTER_PK0 * RATIOS["sheet"],
        k=np.array([0.005, 0.01, 0.021, 0.6]),
    )
    with pytest.raises(cps.PowerSpectrumError, match="k shells do not match"):
        cps.build_normalized_power_spectra(
            matter,
            environments,
            dataset=DATASET,
            simulation_id=SIMULATION_ID,
            snapnum=SNAPSHOT,
            kmin=0.008,
            kmax=0.5,
        )


def test_normalized_plot_rejects_mixed_astra_runs(plot_csvs):
    matter, environments = plot_csvs
    _write_plot_csv(
        environments[cps.RANDOM_VOID_SAMPLE],
        cps.RANDOM_VOID_SAMPLE,
        MATTER_PK0 * RATIOS[cps.RANDOM_VOID_SAMPLE],
        metadata_overrides={"astra_random_seed": "31415"},
    )
    with pytest.raises(cps.PowerSpectrumError, match="astra_random_seed"):
        cps.build_normalized_power_spectra(
            matter,
            environments,
            dataset=DATASET,
            simulation_id=SIMULATION_ID,
            snapnum=SNAPSHOT,
            kmin=0.008,
            kmax=0.5,
        )
