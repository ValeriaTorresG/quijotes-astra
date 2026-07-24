from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
ASTRA_ROOT = REPO_ROOT / "astra"
if str(ASTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ASTRA_ROOT))

from releases import quijotes  # noqa: E402


class FakeFoFCatalog:
    """Small readfof stand-in that also inspects the staged file."""

    calls = []

    def __init__(self, basedir, snapnum, **kwargs):
        staged = (
            Path(basedir)
            / f"groups_{snapnum:03d}"
            / f"group_tab_{snapnum:03d}.0"
        )
        assert staged.is_file()
        type(self).calls.append((staged.resolve(), snapnum, kwargs))

        self.GroupPos = np.array(
            [[1000.0, 2000.0, 3000.0], [999000.0, 998000.0, 997000.0]],
            dtype=np.float32,
        )
        self.GroupMass = np.array([2.0, 3.5], dtype=np.float32)
        self.GroupVel = np.array(
            [[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]],
            dtype=np.float32,
        )
        self.GroupLen = np.array([20, 35], dtype=np.int32)
        self.Nfiles = 1


@pytest.fixture
def flattened_catalog(tmp_path):
    group_dir = tmp_path / "groups_003"
    group_dir.mkdir()
    for simulation_id in (1, 7):
        (group_dir / f"group_tab_003_{simulation_id}.0").write_bytes(
            f"simulation-{simulation_id}".encode()
        )
    return tmp_path


def test_discovers_and_stages_flattened_realization(flattened_catalog):
    found = quijotes.discover_realizations(flattened_catalog, snapshot=3)
    assert list(found) == [1, 7]
    assert found[7][0].name == "group_tab_003_7.0"

    FakeFoFCatalog.calls.clear()
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    fof = quijotes.load_fof_catalog(
        flattened_catalog,
        snapshot=3,
        simulation_id=7,
        readfof_module=fake_readfof,
    )

    source, snapnum, kwargs = FakeFoFCatalog.calls[-1]
    assert source == (
        flattened_catalog / "groups_003" / "group_tab_003_7.0"
    ).resolve()
    assert snapnum == 3
    assert kwargs == {
        "long_ids": False,
        "swap": False,
        "SFR": False,
        "read_IDs": False,
    }
    assert fof._quijotes_source_files == found[7]


def test_staging_parent_is_created(flattened_catalog, tmp_path):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    staging_dir = tmp_path / "new" / "spill"

    quijotes.load_fof_catalog(
        flattened_catalog,
        snapshot=3,
        simulation_id=1,
        staging_dir=staging_dir,
        readfof_module=fake_readfof,
    )

    assert staging_dir.is_dir()


def test_load_converts_notebook_units_and_validates_bounds(flattened_catalog):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    catalog = quijotes.load_halo_catalog(
        flattened_catalog,
        snapshot=3,
        simulation_id=1,
        readfof_module=fake_readfof,
    )

    np.testing.assert_allclose(
        catalog.positions,
        [[1.0, 2.0, 3.0], [999.0, 998.0, 997.0]],
    )
    np.testing.assert_allclose(catalog.masses, [2.0e10, 3.5e10])
    np.testing.assert_allclose(
        catalog.velocities,
        [[15.0, 30.0, 45.0], [-15.0, -30.0, -45.0]],
    )
    np.testing.assert_array_equal(catalog.npart, [20, 35])
    assert catalog.redshift == 0.5
    assert catalog.simulation_id == 1

    with pytest.raises(ValueError, match="outside"):
        quijotes.validate_positions(
            np.array([[1000.0, 0.0, 0.0]]),
            box_min=0.0,
            box_max=1000.0,
        )


def test_raw_table_is_reproducible_unique_and_preserves_real_fields(
    flattened_catalog,
):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    catalog = quijotes.load_halo_catalog(
        flattened_catalog,
        snapshot=3,
        simulation_id=1,
        readfof_module=fake_readfof,
    )

    table = quijotes.build_raw_table(
        catalog,
        n_random=2,
        seed=42,
        box_min=0.0,
        box_max=1000.0,
    )
    assert len(table) == 6
    np.testing.assert_array_equal(table["TARGETID"], np.arange(1, 7))
    assert len(np.unique(table["TARGETID"])) == len(table)
    np.testing.assert_array_equal(table["TRACER_ID"], 0)
    np.testing.assert_array_equal(table["RANDITER"], [-1, -1, 0, 0, 1, 1])
    np.testing.assert_allclose(table["MASS"][:2], catalog.masses)
    np.testing.assert_allclose(table["VEL_X"][:2], catalog.velocities[:, 0])
    np.testing.assert_array_equal(table["NPART"][:2], catalog.npart)
    assert np.all(np.isnan(table["MASS"][2:]))
    assert np.all(np.isnan(table["VEL_X"][2:]))
    np.testing.assert_array_equal(table["NPART"][2:], -1)

    expected_rng = np.random.default_rng(seed=42)
    expected = expected_rng.uniform(
        low=0.0, high=1000.0, size=(4, 3)
    )
    actual = np.column_stack(
        (table["XCART"][2:], table["YCART"][2:], table["ZCART"][2:])
    )
    np.testing.assert_allclose(actual, expected)

    repeated = quijotes.build_raw_table(catalog, n_random=2, seed=42)
    np.testing.assert_array_equal(table["XCART"], repeated["XCART"])
    assert table.meta["PERIODIC"] is True
    assert table.meta["COORDUNT"] == "Mpc/h"
    assert table.meta["NRAND"] == 2


def test_atomic_fits_write_round_trip(tmp_path, flattened_catalog):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    catalog = quijotes.load_halo_catalog(
        flattened_catalog,
        simulation_id=1,
        readfof_module=fake_readfof,
    )
    table = quijotes.build_raw_table(catalog, n_random=1)
    output = tmp_path / "output" / "zone_00.fits.gz"

    result = quijotes.atomic_write_fits(table, output)
    assert result == output.resolve()
    assert output.read_bytes()[:2] == b"\x1f\x8b"
    restored = Table.read(output)
    assert len(restored) == len(table)
    np.testing.assert_array_equal(restored["TARGETID"], table["TARGETID"])
    assert restored.meta["SIMID"] == 1
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_preload_uses_cli_catalog_path_and_optional_halo_limit(
    flattened_catalog,
):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    args = Namespace(
        cat_dir=str(flattened_catalog),
        snapnum=3,
        simulation_id=7,
        redshift=None,
        box_min=[0.0],
        box_max=[1000.0],
        pylians_library="/unused/with/mock",
        spill_dir=None,
        max_halos=1,
    )
    real_tables, random_tables = quijotes.preload_quijotes(
        args, ["HALO"], readfof_module=fake_readfof
    )
    assert random_tables == {}
    assert len(real_tables["HALO"]) == 1
    assert real_tables["HALO"].simulation_id == 7


def test_release_writer_uses_automatic_simulation_snapshot_tag(
    tmp_path, flattened_catalog
):
    fake_readfof = SimpleNamespace(FoF_catalog=FakeFoFCatalog)
    catalog = quijotes.load_halo_catalog(
        flattened_catalog,
        snapshot=3,
        simulation_id=7,
        readfof_module=fake_readfof,
    )
    args = Namespace(
        n_random=1,
        seed=42,
        box_min=[0.0],
        box_max=[1000.0],
        periodic=True,
        max_halos=None,
        raw_out=str(tmp_path),
        out_tag=None,
    )
    quijotes.build_raw_release_table(
        0, {"HALO": catalog}, {}, ["HALO"], args, "QUIJOTES"
    )
    assert (tmp_path / "zone_00_sim007_snap003.fits.gz").is_file()


def test_create_config_supports_single_box():
    args = Namespace(
        zone=None,
        zones=None,
        box_min=[0.0],
        box_max=[1000.0],
        periodic=True,
        snapnum=3,
        simulation_id=0,
        redshift=None,
        seed=42,
        max_halos=8,
        out_tag=None,
    )
    config = quijotes.create_config(args)
    assert config.name == "QUIJOTES"
    assert config.tracers == ["HALO"]
    assert config.zones == [0]
    assert config.combine_outputs is False
    np.testing.assert_array_equal(config.periodic_box[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        config.periodic_box[1], [1000.0, 1000.0, 1000.0]
    )
    assert config.supports_plot is False
    assert config.supports_groups is False
    assert config.product_meta["MAXHALO"] == 8
    assert args.out_tag == "sim000_snap003"
    if hasattr(config, "preload"):
        assert callable(config.preload)

    with pytest.raises(RuntimeError, match="only zone 0"):
        quijotes.create_config(Namespace(zone=1, zones=None))
