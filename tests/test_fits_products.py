from pathlib import Path
import sys

import numpy as np
from astropy.table import Table


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.desiproc.implement_astra import (
    CLASS_ROW_DTYPE,
    PAIR_ROW_DTYPE,
    save_classification_fits,
    save_pairs_fits,
)


def test_chunked_fits_writer_honours_gzip_extension(tmp_path):
    rows = np.empty(2, dtype=PAIR_ROW_DTYPE)
    rows["TARGETID1"] = [1, 2]
    rows["TARGETID2"] = [3, 4]
    rows["RANDITER"] = [0, 1]
    output = tmp_path / "pairs.fits.gz"

    save_pairs_fits(rows, output, meta={"NITER": 2})

    assert output.read_bytes()[:2] == b"\x1f\x8b"
    restored = Table.read(output)
    np.testing.assert_array_equal(restored["TARGETID1"], [1, 2])
    assert restored.meta["NITER"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_classification_writer_splits_rows_by_iteration(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_CLASS_SPLIT_ITER", "1")
    monkeypatch.delenv("ASTRA_CLASS_SKIP_COMBINED", raising=False)

    rows = np.empty(3, dtype=CLASS_ROW_DTYPE)
    rows["TARGETID"] = [1, 2, 3]
    rows["RANDITER"] = [-1, 0, 1]
    rows["ISDATA"] = [True, False, False]
    rows["NDATA"] = [1, 0, 0]
    rows["NRAND"] = [0, 1, 1]
    rows["TRACER_ID"] = 0
    rows["TRACERTYPE"] = b"HALO"
    output = tmp_path / "zone_0_classified.fits"

    save_classification_fits(rows, output)

    np.testing.assert_array_equal(Table.read(output)["RANDITER"], [-1, 0, 1])
    np.testing.assert_array_equal(Table.read(tmp_path / "zone_0_iterm001.fits")["RANDITER"], [-1])
    np.testing.assert_array_equal(Table.read(tmp_path / "zone_0_iter000.fits")["RANDITER"], [0])
    np.testing.assert_array_equal(Table.read(tmp_path / "zone_0_iter001.fits")["RANDITER"], [1])
    assert not list(tmp_path.glob("*.tmp"))
