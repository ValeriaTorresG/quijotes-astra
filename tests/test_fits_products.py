from pathlib import Path
import sys

import numpy as np
from astropy.table import Table


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.desiproc.implement_astra import PAIR_ROW_DTYPE, save_pairs_fits


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
