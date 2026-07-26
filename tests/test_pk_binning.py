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


BOX_SIZE = 1000.0
GRID = 512
N_OBJECTS = 100
SHOT_NOISE = BOX_SIZE**3 / N_OBJECTS
K_FUNDAMENTAL = 2.0 * np.pi / BOX_SIZE
AUTO_BIN_WIDTH = 2.0 * K_FUNDAMENTAL
CUSTOM_BIN_WIDTH = 3.0 * K_FUNDAMENTAL


def _native_spectrum():
    k = np.array([0.006, 0.009, 0.013, 0.018, 0.025])
    pk0 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    nmodes = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    return cps.Spectrum(
        k=k,
        pk0_raw=pk0,
        pk0_shot_subtracted=pk0 - SHOT_NOISE,
        pk2=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        pk4=np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        sigma_pk0=pk0 * np.sqrt(2.0 / nmodes),
        nmodes=nmodes,
        n_objects=N_OBJECTS,
        number_density=N_OBJECTS / BOX_SIZE**3,
        shot_noise=SHOT_NOISE,
    )


def test_auto_k_bin_edges_follow_example_rule():
    assert cps.resolve_k_bin_width(BOX_SIZE, None) is None
    edges, width = cps.make_k_bin_edges(
        BOX_SIZE,
        GRID,
        0.5,
        0.0,
    )
    expected_width = AUTO_BIN_WIDTH
    assert width == pytest.approx(expected_width)
    np.testing.assert_allclose(np.diff(edges), expected_width)
    assert edges[0] == 0.0
    assert edges[-2] < 0.5 <= edges[-1]

    with pytest.raises(cps.PowerSpectrumError, match="native Pylians resolution"):
        cps.resolve_k_bin_width(BOX_SIZE, 0.001)
    with pytest.raises(cps.PowerSpectrumError, match="integer multiple"):
        cps.resolve_k_bin_width(BOX_SIZE, 0.01)


def test_rebin_spectrum_uses_nmodes_weights_and_conserves_modes():
    native = _native_spectrum()
    edges = np.array([0.0, 0.01, 0.02, 0.03])
    binned = cps.rebin_spectrum(native, edges, 0.01)

    expected_modes = np.array([3, 7, 5])
    expected_k = np.array(
        [
            (0.006 * 1 + 0.009 * 2) / 3,
            (0.013 * 3 + 0.018 * 4) / 7,
            0.025,
        ]
    )
    expected_pk0 = np.array(
        [
            (10.0 * 1 + 20.0 * 2) / 3,
            (30.0 * 3 + 40.0 * 4) / 7,
            50.0,
        ]
    )
    np.testing.assert_allclose(binned.k, expected_k)
    np.testing.assert_allclose(binned.pk0_raw, expected_pk0)
    np.testing.assert_allclose(
        binned.pk2,
        [(1.0 + 4.0) / 3, (9.0 + 16.0) / 7, 5.0],
    )
    np.testing.assert_allclose(
        binned.pk4,
        [(2.0 + 8.0) / 3, (18.0 + 32.0) / 7, 10.0],
    )
    np.testing.assert_array_equal(binned.nmodes, expected_modes)
    assert int(np.sum(binned.nmodes)) == int(np.sum(native.nmodes))
    np.testing.assert_allclose(
        binned.pk0_shot_subtracted,
        expected_pk0 - SHOT_NOISE,
    )
    np.testing.assert_allclose(
        binned.sigma_pk0,
        expected_pk0 * np.sqrt(2.0 / expected_modes),
    )
    np.testing.assert_array_equal(binned.k_bin_index, [0, 1, 2])
    np.testing.assert_allclose(binned.k_bin_min, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(binned.k_bin_max, [0.01, 0.02, 0.03])


def test_binned_csv_records_edges_and_shot_noise_and_validates(tmp_path):
    edges, width = cps.make_k_bin_edges(
        BOX_SIZE,
        GRID,
        0.03,
        CUSTOM_BIN_WIDTH,
    )
    spectrum = cps.rebin_spectrum(_native_spectrum(), edges, width)
    output = tmp_path / "binned_pk.csv"
    cps.write_spectrum_csv(
        output,
        spectrum,
        dataset="fiducial",
        simulation_id=7,
        snapnum=3,
        sample="all",
        tracer="FoF_halo",
        fof_catalog_sha256="f" * 64,
        astra_provenance=None,
        box_size=BOX_SIZE,
        grid=GRID,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.005,
        kmax=0.03,
        overwrite=False,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["binning_mode"] == "fixed_nmodes_weighted"
    assert float(rows[0]["k_bin_width_h_Mpc"]) == pytest.approx(
        CUSTOM_BIN_WIDTH
    )
    assert [int(row["k_bin_index"]) for row in rows] == [0, 1]
    assert all(
        float(row["shot_noise_Mpc3_h3"]) == pytest.approx(SHOT_NOISE)
        for row in rows
    )

    args = SimpleNamespace(
        box_size=BOX_SIZE,
        grid=GRID,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.005,
        kmax=0.03,
        k_bin_width=CUSTOM_BIN_WIDTH,
    )
    cps.validate_existing_spectrum_csv(
        output,
        dataset="fiducial",
        simulation_id=7,
        snapnum=3,
        sample="all",
        tracer="FoF_halo",
        n_objects=N_OBJECTS,
        fof_catalog_sha256="f" * 64,
        astra_provenance=None,
        args=args,
    )

    incompatible = SimpleNamespace(**vars(args))
    incompatible.k_bin_width = AUTO_BIN_WIDTH
    with pytest.raises(cps.PowerSpectrumError):
        cps.validate_existing_spectrum_csv(
            output,
            dataset="fiducial",
            simulation_id=7,
            snapnum=3,
            sample="all",
            tracer="FoF_halo",
            n_objects=N_OBJECTS,
            fof_catalog_sha256="f" * 64,
            astra_provenance=None,
            args=incompatible,
        )

    corrupted = tmp_path / "binned_pk_bad_shot_noise.csv"
    bad_rows = [dict(row) for row in rows]
    bad_rows[0]["shot_noise_Mpc3_h3"] = repr(SHOT_NOISE + 1.0)
    with corrupted.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cps.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(bad_rows)
    with pytest.raises(cps.PowerSpectrumError, match="shot noise"):
        cps.validate_existing_spectrum_csv(
            corrupted,
            dataset="fiducial",
            simulation_id=7,
            snapnum=3,
            sample="all",
            tracer="FoF_halo",
            n_objects=N_OBJECTS,
            fof_catalog_sha256="f" * 64,
            astra_provenance=None,
            args=args,
        )


def test_pre_binning_csv_is_accepted_only_for_native_mode(tmp_path):
    current = tmp_path / "native_current.csv"
    cps.write_spectrum_csv(
        current,
        _native_spectrum(),
        dataset="fiducial",
        simulation_id=7,
        snapnum=3,
        sample="all",
        tracer="FoF_halo",
        fof_catalog_sha256="f" * 64,
        astra_provenance=None,
        box_size=BOX_SIZE,
        grid=GRID,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.005,
        kmax=0.03,
        overwrite=False,
    )
    with current.open("r", encoding="utf-8", newline="") as handle:
        current_rows = list(csv.DictReader(handle))
    old = tmp_path / "native_pre_binning.csv"
    with old.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=cps.PRE_BINNING_CSV_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(
            {
                column: row[column]
                for column in cps.PRE_BINNING_CSV_COLUMNS
            }
            for row in current_rows
        )

    args = SimpleNamespace(
        box_size=BOX_SIZE,
        grid=GRID,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.005,
        kmax=0.03,
        k_bin_width=None,
    )
    common = dict(
        path=old,
        dataset="fiducial",
        simulation_id=7,
        snapnum=3,
        sample="all",
        tracer="FoF_halo",
        n_objects=N_OBJECTS,
        fof_catalog_sha256="f" * 64,
        astra_provenance=None,
    )
    cps.validate_existing_spectrum_csv(args=args, **common)

    args.k_bin_width = 0.0
    with pytest.raises(cps.PowerSpectrumError, match="pre-binning CSV"):
        cps.validate_existing_spectrum_csv(args=args, **common)


def test_k_bin_width_cli_default_auto_and_custom():
    parser = cps.build_parser()
    assert parser.parse_args([]).k_bin_width is None
    assert parser.parse_args(["--k-bin-width"]).k_bin_width == 0.0
    assert parser.parse_args(["--k-bin-width", "0"]).k_bin_width == 0.0
    custom = parser.parse_args(
        ["--k-bin-width", format(CUSTOM_BIN_WIDTH, ".17g")]
    )
    assert custom.k_bin_width == CUSTOM_BIN_WIDTH
    cps.validate_args(custom, parser)

    too_narrow = parser.parse_args(["--k-bin-width", "0.001"])
    with pytest.raises(SystemExit):
        cps.validate_args(too_narrow, parser)

    misaligned = parser.parse_args(["--k-bin-width", "0.01"])
    with pytest.raises(SystemExit):
        cps.validate_args(misaligned, parser)
