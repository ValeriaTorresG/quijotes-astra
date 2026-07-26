from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pk import compute_power_spectra as cps


N_HALOS = 4
N_ITERATIONS = 2
SIMULATION_ID = 7
SNAPSHOT = 3
BOX_SIZE = 1000.0


def _classification_meta():
    return {
        "ZONE": "00",
        "RELEASE": "QUIJOTES",
        "RLOWER": -0.25,
        "RMED": 0.25,
        "RUPPER": 0.65,
        "NITER": N_ITERATIONS,
        "SNAPNUM": SNAPSHOT,
        "SIMID": SIMULATION_ID,
        "REDSHFT": 0.5,
        "RNGSEED": 42,
        "PERIODIC": True,
        "BOXXMIN": 0.0,
        "BOXYMIN": 0.0,
        "BOXZMIN": 0.0,
        "BOXXMAX": BOX_SIZE,
        "BOXYMAX": BOX_SIZE,
        "BOXZMAX": BOX_SIZE,
    }


@pytest.fixture
def astra_random_products(tmp_path):
    run = tmp_path / "astra" / f"fiducial_{SIMULATION_ID}"
    class_dir = (
        run
        / "astra"
        / "classification"
        / f"sim{SIMULATION_ID:03d}"
        / "00"
    )
    class_dir.mkdir(parents=True)

    random_counts = (
        # Ratios: -1, 0, 1, 0. The zero-denominator row is not a void.
        (np.array([0, 1, 2, 0]), np.array([2, 1, 0, 0])),
        # Ratios: -1, -1, 0, 1.
        (np.array([0, 0, 1, 3]), np.array([1, 3, 1, 0])),
    )
    classification_paths = []
    for iteration, (ndata_random, nrand_random) in enumerate(random_counts):
        random_start = N_HALOS * (iteration + 1) + 1
        table = Table()
        table["TARGETID"] = np.concatenate(
            [
                np.arange(1, N_HALOS + 1),
                np.arange(random_start, random_start + N_HALOS),
            ]
        ).astype(np.int64)
        table["RANDITER"] = np.full(2 * N_HALOS, iteration, dtype=np.int32)
        table["ISDATA"] = np.concatenate(
            [
                np.ones(N_HALOS, dtype=bool),
                np.zeros(N_HALOS, dtype=bool),
            ]
        )
        table["NDATA"] = np.concatenate(
            [np.ones(N_HALOS, dtype=np.int32), ndata_random]
        )
        table["NRAND"] = np.concatenate(
            [np.ones(N_HALOS, dtype=np.int32), nrand_random]
        )
        table["TRACERTYPE"] = np.full(2 * N_HALOS, "HALO")
        table.meta.update(_classification_meta())
        path = class_dir / (
            f"zone_00_sim{SIMULATION_ID:03d}_snap{SNAPSHOT:03d}_"
            f"iter{iteration:03d}.fits.gz"
        )
        table.write(path)
        classification_paths.append(path)

    total_rows = N_HALOS * (N_ITERATIONS + 1)
    coordinates = (
        np.arange(total_rows * 3, dtype=np.float64).reshape(total_rows, 3)
        + 0.125
    )
    raw = Table()
    raw["TARGETID"] = np.arange(1, total_rows + 1, dtype=np.int64)
    raw["RANDITER"] = np.concatenate(
        [
            np.full(N_HALOS, -1, dtype=np.int32),
            np.repeat(np.arange(N_ITERATIONS, dtype=np.int32), N_HALOS),
        ]
    )
    raw["XCART"] = coordinates[:, 0]
    raw["YCART"] = coordinates[:, 1]
    raw["ZCART"] = coordinates[:, 2]
    raw.meta.update(
        {
            "RELEASE": "QUIJOTES",
            "SNAPNUM": SNAPSHOT,
            "REDSHFT": 0.5,
            "SIMID": SIMULATION_ID,
            "NREAL": N_HALOS,
            "NITER": N_ITERATIONS,
            "NRANDPT": N_HALOS,
            "RNGSEED": 42,
            "PERIODIC": True,
            "BOXLOX": 0.0,
            "BOXLOY": 0.0,
            "BOXLOZ": 0.0,
            "BOXHIX": BOX_SIZE,
            "BOXHIY": BOX_SIZE,
            "BOXHIZ": BOX_SIZE,
        }
    )
    raw_dir = run / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / (
        f"zone_00_sim{SIMULATION_ID:03d}_snap{SNAPSHOT:03d}.fits.gz"
    )
    raw.write(raw_path)
    return {
        "astra_root": tmp_path / "astra",
        "classification_paths": tuple(classification_paths),
        "raw_path": raw_path,
        "coordinates": coordinates,
    }


def test_random_void_ids_are_selected_then_crossmatched_to_raw(
    astra_random_products,
    tmp_path,
):
    astra_root = astra_random_products["astra_root"]
    classification_paths = cps.resolve_classification_paths(
        astra_root,
        "fiducial",
        SIMULATION_ID,
        SNAPSHOT,
    )
    assert classification_paths == tuple(
        path.resolve()
        for path in astra_random_products["classification_paths"]
    )
    assert cps.resolve_classification_paths(
        astra_root.parent,
        "fiducial",
        SIMULATION_ID,
        SNAPSHOT,
    ) == classification_paths
    raw_path = cps.resolve_raw_path(
        astra_root,
        "fiducial",
        SIMULATION_ID,
        SNAPSHOT,
    )
    assert raw_path == astra_random_products["raw_path"].resolve()

    selection = cps.read_random_void_selection(
        classification_paths,
        N_HALOS,
        SIMULATION_ID,
        SNAPSHOT,
        BOX_SIZE,
    )
    np.testing.assert_array_equal(selection.target_ids, [5, 9, 10])
    assert selection.n_objects == 3
    assert selection.provenance.classification_sha256

    temp_parent = tmp_path / "raw_tmp"
    positions, provenance = cps.read_random_void_positions_from_raw(
        raw_path,
        selection,
        n_halos=N_HALOS,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        box_size=BOX_SIZE,
        temp_parent=temp_parent,
    )
    expected = astra_random_products["coordinates"][
        selection.target_ids - 1
    ].astype(np.float32)
    np.testing.assert_array_equal(positions, expected)
    assert provenance.raw_path == raw_path
    assert provenance.raw_sha256
    assert temp_parent.is_dir()
    assert list(temp_parent.iterdir()) == []


def test_missing_classification_iteration_is_rejected(astra_random_products):
    astra_random_products["classification_paths"][1].unlink()
    paths = cps.resolve_classification_paths(
        astra_random_products["astra_root"],
        "fiducial",
        SIMULATION_ID,
        SNAPSHOT,
    )
    with pytest.raises(cps.AstraProductsIncompleteError, match="NITER"):
        cps.read_random_void_selection(
            paths,
            N_HALOS,
            SIMULATION_ID,
            SNAPSHOT,
            BOX_SIZE,
        )


def test_random_void_csv_records_and_validates_source_provenance(
    astra_random_products,
    tmp_path,
):
    paths = cps.resolve_classification_paths(
        astra_random_products["astra_root"],
        "fiducial",
        SIMULATION_ID,
        SNAPSHOT,
    )
    selection = cps.read_random_void_selection(
        paths,
        N_HALOS,
        SIMULATION_ID,
        SNAPSHOT,
        BOX_SIZE,
    )
    _, provenance = cps.read_random_void_positions_from_raw(
        astra_random_products["raw_path"],
        selection,
        n_halos=N_HALOS,
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        box_size=BOX_SIZE,
        temp_parent=tmp_path / "raw_tmp",
    )
    nmodes = np.array([13, 33], dtype=np.int64)
    pk0 = np.array([100.0, 200.0])
    shot_noise = BOX_SIZE**3 / selection.n_objects
    spectrum = cps.Spectrum(
        k=np.array([0.01, 0.02]),
        pk0_raw=pk0,
        pk0_shot_subtracted=pk0 - shot_noise,
        pk2=np.array([1.0, 2.0]),
        pk4=np.array([3.0, 4.0]),
        sigma_pk0=pk0 * np.sqrt(2.0 / nmodes),
        nmodes=nmodes,
        n_objects=selection.n_objects,
        number_density=selection.n_objects / BOX_SIZE**3,
        shot_noise=shot_noise,
    )
    output = tmp_path / "random_void_pk.csv"
    cps.write_spectrum_csv(
        output,
        spectrum,
        dataset="fiducial",
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        sample=cps.RANDOM_VOID_SAMPLE,
        tracer=cps.RANDOM_VOID_TRACER,
        fof_catalog_sha256="f" * 64,
        astra_provenance=provenance,
        box_size=BOX_SIZE,
        grid=512,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.008,
        kmax=0.5,
        overwrite=False,
    )
    args = SimpleNamespace(
        box_size=BOX_SIZE,
        grid=512,
        mas="CIC",
        axis=0,
        threads=1,
        kmin=0.008,
        kmax=0.5,
    )
    cps.validate_existing_spectrum_csv(
        output,
        dataset="fiducial",
        simulation_id=SIMULATION_ID,
        snapnum=SNAPSHOT,
        sample=cps.RANDOM_VOID_SAMPLE,
        tracer=cps.RANDOM_VOID_TRACER,
        n_objects=selection.n_objects,
        fof_catalog_sha256="f" * 64,
        astra_provenance=provenance,
        args=args,
    )
    first = Table.read(output, format="ascii.csv")[0]
    assert first["astra_classification_sha256"] == (
        provenance.classification_sha256
    )
    assert first["astra_raw_sha256"] == provenance.raw_sha256
    assert first["selection_rule"] == cps.RANDOM_VOID_SELECTION_RULE
