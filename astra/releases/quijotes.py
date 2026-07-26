"""Quijote FoF catalogue support for the ASTRA pipeline.

The downloaded catalogues used by this project are flattened: files from
different realizations live together and encode the realization in names such
as ``group_tab_003_105.0``.  Pylians' :class:`readfof.FoF_catalog`, however,
expects the usual Gadget layout::

    <base>/groups_003/group_tab_003.0

This module resolves both layouts and stages read-only links in a temporary
directory before invoking ``readfof``.  Source catalogues are never renamed or
modified.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory, mkstemp
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.table import Table

from desiproc.implement_astra import register_tracer_mapping

from .base import ReleaseConfig


TRACER = "HALO"
TRACERS = [TRACER]
TRACER_ALIAS = {"halo": TRACER, "quijotes": TRACER}

DEFAULT_CATALOG_DIR = (
    Path.home() / "Desktop" / "Quijotes" / "data" / "fiducial"
)
DEFAULT_PYLIANS_LIBRARY = Path.home() / "Pylians3" / "library"
DEFAULT_SNAPSHOT = 3
DEFAULT_SIMULATION_ID = 0
DEFAULT_BOX_MIN = 0.0
DEFAULT_BOX_MAX = 1000.0
DEFAULT_RANDOM_SEED = 42

# Mapping documented by Pylians for the Quijote halo catalogues.
SNAPSHOT_REDSHIFTS = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.5, 4: 0.0}


@dataclass(frozen=True)
class QuijoteHaloCatalog:
    """Converted physical fields from one Quijote FoF realization."""

    positions: np.ndarray
    masses: np.ndarray
    velocities: np.ndarray
    npart: np.ndarray
    snapshot: int
    redshift: float
    simulation_id: int
    source_files: tuple[Path, ...]

    def __len__(self) -> int:
        return int(self.positions.shape[0])


def _as_path(path: str | os.PathLike[str]) -> Path:
    """Return an expanded, absolute path without requiring that it exist."""

    return Path(path).expanduser().resolve()


def _part_number(path: Path) -> int:
    """Return the numeric file-part suffix from a FoF tab filename."""

    try:
        return int(path.suffix[1:])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid FoF part filename: {path}") from exc


def _ordered_parts(paths: Sequence[Path]) -> tuple[Path, ...]:
    """Sort FoF parts and require a contiguous sequence beginning at zero."""

    ordered = tuple(sorted((_as_path(path) for path in paths), key=_part_number))
    if not ordered:
        return ordered
    numbers = [_part_number(path) for path in ordered]
    expected = list(range(len(ordered)))
    if numbers != expected:
        raise ValueError(
            f"FoF parts must be contiguous from 0; found {numbers}, "
            f"expected {expected}"
        )
    return ordered


def discover_realizations(
    catalog_dir: str | os.PathLike[str],
    snapshot: int = DEFAULT_SNAPSHOT,
) -> dict[int, tuple[Path, ...]]:
    """Discover available realization IDs for one snapshot.

    The resolver recognizes:

    - flattened files below ``groups_NNN``;
    - flattened files directly below ``catalog_dir``;
    - standard ``catalog_dir/<realization>/groups_NNN`` directories;
    - a canonical catalogue directly below ``catalog_dir/groups_NNN``
      (treated as realization zero when no explicit zero file is present).
    """

    root = _as_path(catalog_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Quijote catalogue directory not found: {root}")

    tag = f"{int(snapshot):03d}"
    flattened_re = re.compile(
        rf"^group_tab_{re.escape(tag)}_(\d+)\.(\d+)$"
    )
    canonical_re = re.compile(rf"^group_tab_{re.escape(tag)}\.(\d+)$")
    found: dict[int, list[Path]] = {}

    for directory in (root / f"groups_{tag}", root):
        if not directory.is_dir():
            continue
        directory_found: dict[int, list[Path]] = {}
        for path in directory.iterdir():
            match = flattened_re.match(path.name)
            if match and path.is_file():
                directory_found.setdefault(int(match.group(1)), []).append(path)
        for simulation_id, paths in directory_found.items():
            # Prefer the conventional groups_NNN directory when duplicate
            # flattened catalogues also exist at the root.
            found.setdefault(simulation_id, paths)

    for child in root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        directory = child / f"groups_{tag}"
        if not directory.is_dir():
            continue
        parts = [
            path
            for path in directory.iterdir()
            if path.is_file() and canonical_re.match(path.name)
        ]
        if parts:
            found[int(child.name)] = parts

    canonical_dir = root / f"groups_{tag}"
    if 0 not in found and canonical_dir.is_dir():
        parts = [
            path
            for path in canonical_dir.iterdir()
            if path.is_file() and canonical_re.match(path.name)
        ]
        if parts:
            found[0] = parts

    return {
        simulation_id: _ordered_parts(parts)
        for simulation_id, parts in sorted(found.items())
    }


def resolve_realization_parts(
    catalog_dir: str | os.PathLike[str],
    snapshot: int = DEFAULT_SNAPSHOT,
    simulation_id: int = DEFAULT_SIMULATION_ID,
) -> tuple[Path, ...]:
    """Resolve all FoF tab parts for one requested realization."""

    available = discover_realizations(catalog_dir, snapshot)
    try:
        return available[int(simulation_id)]
    except KeyError as exc:
        ids = ", ".join(str(value) for value in available) or "none"
        raise FileNotFoundError(
            f"Quijote realization {simulation_id} for snapshot "
            f"{int(snapshot):03d} not found in {_as_path(catalog_dir)}; "
            f"available realizations: {ids}"
        ) from exc


def import_readfof(
    pylians_library: str | os.PathLike[str] = DEFAULT_PYLIANS_LIBRARY,
) -> Any:
    """Import Pylians' standalone ``readfof.py`` module."""

    try:
        import readfof  # type: ignore[import]
        return readfof
    except ImportError:
        library = _as_path(pylians_library)
        module_path = library / "readfof.py"
        if not module_path.is_file():
            raise FileNotFoundError(f"Pylians readfof module not found: {module_path}")
        library_text = str(library)
        if library_text not in sys.path:
            sys.path.insert(0, library_text)
        return importlib.import_module("readfof")


def _stage_part(source: Path, destination: Path) -> None:
    """Create a read-only staging link, copying only when links are unavailable."""

    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def load_fof_catalog(
    catalog_dir: str | os.PathLike[str],
    snapshot: int = DEFAULT_SNAPSHOT,
    simulation_id: int = DEFAULT_SIMULATION_ID,
    *,
    pylians_library: str | os.PathLike[str] = DEFAULT_PYLIANS_LIBRARY,
    staging_dir: str | os.PathLike[str] | None = None,
    readfof_module: Any | None = None,
) -> Any:
    """Load one realization with ``readfof.FoF_catalog``.

    ``read_IDs`` is deliberately false because the downloaded data contain
    ``group_tab`` files only.
    """

    snapshot = int(snapshot)
    parts = resolve_realization_parts(catalog_dir, snapshot, simulation_id)
    readfof = (
        readfof_module
        if readfof_module is not None
        else import_readfof(pylians_library)
    )
    tag = f"{snapshot:03d}"
    if staging_dir is None:
        temp_parent = None
    else:
        staging_parent = _as_path(staging_dir)
        staging_parent.mkdir(parents=True, exist_ok=True)
        temp_parent = str(staging_parent)

    with TemporaryDirectory(prefix="quijotes_fof_", dir=temp_parent) as temp:
        base = Path(temp)
        group_dir = base / f"groups_{tag}"
        group_dir.mkdir()
        for part, source in enumerate(parts):
            _stage_part(source, group_dir / f"group_tab_{tag}.{part}")

        fof = readfof.FoF_catalog(
            str(base),
            snapshot,
            long_ids=False,
            swap=False,
            SFR=False,
            read_IDs=False,
        )

    nfiles = int(getattr(fof, "Nfiles", len(parts)))
    if nfiles != len(parts):
        raise ValueError(
            f"FoF header declares {nfiles} files, but {len(parts)} were resolved"
        )
    # Useful for diagnostics while keeping the public return value compatible
    # with direct use of readfof.
    fof._quijotes_source_files = parts
    return fof


def snapshot_redshift(snapshot: int, redshift: float | None = None) -> float:
    """Return an explicit redshift or the documented Quijote snapshot value."""

    if redshift is not None:
        value = float(redshift)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid redshift: {redshift}")
        return value
    try:
        return SNAPSHOT_REDSHIFTS[int(snapshot)]
    except KeyError as exc:
        raise ValueError(
            f"No default redshift is known for snapshot {snapshot}; "
            "provide redshift explicitly"
        ) from exc


def validate_box_bounds(
    box_min: float | Sequence[float],
    box_max: float | Sequence[float],
    n_coord: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and broadcast lower/upper box bounds."""

    try:
        low = np.broadcast_to(
            np.asarray(box_min, dtype=np.float64), (int(n_coord),)
        ).copy()
        high = np.broadcast_to(
            np.asarray(box_max, dtype=np.float64), (int(n_coord),)
        ).copy()
    except ValueError as exc:
        raise ValueError(
            f"Box bounds must be scalar or length {int(n_coord)}"
        ) from exc

    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Box bounds must be finite")
    if np.any(high <= low):
        raise ValueError(
            f"Every upper box bound must exceed its lower bound: "
            f"low={low}, high={high}"
        )
    return low, high


def validate_positions(
    positions: np.ndarray,
    box_min: float | Sequence[float] = DEFAULT_BOX_MIN,
    box_max: float | Sequence[float] = DEFAULT_BOX_MAX,
) -> np.ndarray:
    """Validate a non-empty ``(N, 3)`` catalogue inside ``[min, max)``."""

    values = np.asarray(positions)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"Positions must have shape (N, 3), got {values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError("Quijote halo catalogue is empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("Positions contain non-finite values")

    low, high = validate_box_bounds(box_min, box_max, 3)
    outside = np.any((values < low) | (values >= high), axis=1)
    if np.any(outside):
        mins = np.min(values, axis=0)
        maxs = np.max(values, axis=0)
        raise ValueError(
            f"{int(outside.sum())} positions fall outside [box_min, box_max); "
            f"catalogue min={mins}, max={maxs}, low={low}, high={high}"
        )
    return values


def halo_catalog_from_fof(
    fof: Any,
    *,
    snapshot: int = DEFAULT_SNAPSHOT,
    simulation_id: int = DEFAULT_SIMULATION_ID,
    redshift: float | None = None,
    box_min: float | Sequence[float] = DEFAULT_BOX_MIN,
    box_max: float | Sequence[float] = DEFAULT_BOX_MAX,
    source_files: Sequence[Path] | None = None,
) -> QuijoteHaloCatalog:
    """Convert the fields and units used in ``read_data.ipynb``."""

    z = snapshot_redshift(snapshot, redshift)
    positions = np.asarray(fof.GroupPos, dtype=np.float64) / 1.0e3
    masses = np.asarray(fof.GroupMass, dtype=np.float64) * 1.0e10
    velocities = np.asarray(fof.GroupVel, dtype=np.float64) * (1.0 + z)
    npart = np.asarray(fof.GroupLen, dtype=np.int32)

    validate_positions(positions, box_min, box_max)
    n_halos = positions.shape[0]
    expected_vectors = (n_halos, 3)
    if velocities.shape != expected_vectors:
        raise ValueError(
            f"GroupVel must have shape {expected_vectors}, "
            f"got {velocities.shape}"
        )
    if masses.shape != (n_halos,):
        raise ValueError(
            f"GroupMass must have shape {(n_halos,)}, got {masses.shape}"
        )
    if npart.shape != (n_halos,):
        raise ValueError(
            f"GroupLen must have shape {(n_halos,)}, got {npart.shape}"
        )
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("GroupMass contains invalid values")
    if not np.all(np.isfinite(velocities)):
        raise ValueError("GroupVel contains non-finite values")
    if np.any(npart < 0):
        raise ValueError("GroupLen contains negative particle counts")

    sources = source_files
    if sources is None:
        sources = getattr(fof, "_quijotes_source_files", ())
    return QuijoteHaloCatalog(
        positions=positions,
        masses=masses,
        velocities=velocities,
        npart=npart,
        snapshot=int(snapshot),
        redshift=z,
        simulation_id=int(simulation_id),
        source_files=tuple(_as_path(path) for path in sources),
    )


def load_halo_catalog(
    catalog_dir: str | os.PathLike[str],
    snapshot: int = DEFAULT_SNAPSHOT,
    simulation_id: int = DEFAULT_SIMULATION_ID,
    *,
    redshift: float | None = None,
    box_min: float | Sequence[float] = DEFAULT_BOX_MIN,
    box_max: float | Sequence[float] = DEFAULT_BOX_MAX,
    pylians_library: str | os.PathLike[str] = DEFAULT_PYLIANS_LIBRARY,
    staging_dir: str | os.PathLike[str] | None = None,
    readfof_module: Any | None = None,
) -> QuijoteHaloCatalog:
    """Resolve, read, convert, and validate one Quijote realization."""

    parts = resolve_realization_parts(catalog_dir, snapshot, simulation_id)
    fof = load_fof_catalog(
        catalog_dir,
        snapshot,
        simulation_id,
        pylians_library=pylians_library,
        staging_dir=staging_dir,
        readfof_module=readfof_module,
    )
    return halo_catalog_from_fof(
        fof,
        snapshot=snapshot,
        simulation_id=simulation_id,
        redshift=redshift,
        box_min=box_min,
        box_max=box_max,
        source_files=parts,
    )


def build_raw_table(
    catalog: QuijoteHaloCatalog,
    n_random: int = 100,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    box_min: float | Sequence[float] = DEFAULT_BOX_MIN,
    box_max: float | Sequence[float] = DEFAULT_BOX_MAX,
    zone: int = 0,
    tracer: str = TRACER,
    release_tag: str = "QUIJOTES",
    periodic: bool = True,
) -> Table:
    """Build real rows plus one uniform random catalogue per iteration.

    Every random iteration contains exactly as many objects as the real halo
    catalogue.  A single NumPy generator is seeded once, matching the notebook
    convention and making all iterations reproducible.
    """

    n_random = int(n_random)
    if n_random <= 0:
        raise ValueError(f"n_random must be positive, got {n_random}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    n_real = len(catalog)
    low, high = validate_box_bounds(box_min, box_max, 3)
    validate_positions(catalog.positions, low, high)

    total = n_real * (n_random + 1)
    if total > np.iinfo(np.int64).max - 1:
        raise OverflowError("Raw catalogue is too large for int64 TARGETIDs")

    coords = np.empty((total, 3), dtype=np.float64)
    coords[:n_real] = catalog.positions
    randiter = np.empty(total, dtype=np.int32)
    randiter[:n_real] = -1

    rng = np.random.default_rng(seed=seed)
    for iteration in range(n_random):
        start = n_real * (iteration + 1)
        stop = start + n_real
        coords[start:stop] = rng.uniform(
            low=low, high=high, size=(n_real, 3)
        )
        randiter[start:stop] = iteration

    masses = np.full(total, np.nan, dtype=np.float64)
    masses[:n_real] = catalog.masses
    velocities = np.full((total, 3), np.nan, dtype=np.float64)
    velocities[:n_real] = catalog.velocities
    npart = np.full(total, -1, dtype=np.int32)
    npart[:n_real] = catalog.npart

    tracer_width = max(1, len(str(tracer)))
    table = Table()
    table["TARGETID"] = np.arange(1, total + 1, dtype=np.int64)
    table["TRACERTYPE"] = np.full(
        total, str(tracer), dtype=f"U{tracer_width}"
    )
    table["TRACER_ID"] = np.zeros(total, dtype=np.uint8)
    table["RANDITER"] = randiter
    table["ZONE"] = np.full(total, int(zone), dtype=np.int32)
    table["XCART"] = coords[:, 0]
    table["YCART"] = coords[:, 1]
    table["ZCART"] = coords[:, 2]
    table["MASS"] = masses
    table["VEL_X"] = velocities[:, 0]
    table["VEL_Y"] = velocities[:, 1]
    table["VEL_Z"] = velocities[:, 2]
    table["NPART"] = npart

    table.meta.update(
        {
            "RELEASE": str(release_tag),
            "SNAPNUM": catalog.snapshot,
            "REDSHFT": catalog.redshift,
            "SIMID": catalog.simulation_id,
            "NREAL": n_real,
            "NRAND": n_random,
            "NITER": n_random,
            "NRANDPT": n_real,
            "RNGSEED": seed,
            "PERIODIC": bool(periodic),
            "COORDUNT": "Mpc/h",
            "MASSUNIT": "Msun/h",
            "VELUNIT": "km/s",
            "NULLNPAR": -1,
        }
    )
    for axis, name in enumerate(("X", "Y", "Z")):
        table.meta[f"BOXLO{name}"] = float(low[axis])
        table.meta[f"BOXHI{name}"] = float(high[axis])
    if catalog.source_files:
        table.meta["CATFILE"] = ",".join(
            path.name for path in catalog.source_files
        )
    return table


def atomic_write_fits(
    table: Table,
    output_path: str | os.PathLike[str],
    *,
    overwrite: bool = True,
) -> Path:
    """Write a FITS table atomically in its destination directory."""

    destination = _as_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    suffix = ".fits.gz" if destination.name.endswith(".fits.gz") else ".fits"
    descriptor, temp_name = mkstemp(
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        table.write(temp_path, format="fits", overwrite=True)
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def _safe_tag(value: Any | None) -> str:
    """Return the filename suffix convention used by the other releases."""

    if value is None:
        return ""
    safe = re.sub(r"[^A-Za-z0-9_\-.\+]+", "_", str(value))
    return f"_{safe}" if safe else ""


def default_output_tag(simulation_id: int, snapshot: int) -> str:
    """Return a collision-resistant tag for one simulation and snapshot."""

    return f"sim{int(simulation_id):03d}_snap{int(snapshot):03d}"


def _zone_tag(zone: Any) -> str:
    """Return a zero-padded numeric zone or preserve a string label."""

    try:
        return f"{int(zone):02d}"
    except (TypeError, ValueError):
        return str(zone)


def _arg(args: Namespace, *names: str, default: Any = None) -> Any:
    """Read the first non-``None`` argument from a namespace."""

    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return value
    return default


def preload_quijotes(
    args: Namespace,
    tracers: Sequence[str] | None = None,
    *,
    readfof_module: Any | None = None,
) -> tuple[dict[str, QuijoteHaloCatalog], dict[str, Any]]:
    """Release-specific preload hook used by the main pipeline."""

    selected = TRACERS if tracers is None else list(tracers)
    unknown = [tracer for tracer in selected if tracer != TRACER]
    if unknown:
        raise ValueError(f"Unsupported Quijote tracers: {unknown}")

    catalog_dir = _arg(
        args,
        "cat_dir",
        "quijotes_catalog_dir",
        "catalog_dir",
        "base_dir",
        default=DEFAULT_CATALOG_DIR,
    )
    snapshot = int(
        _arg(args, "quijotes_snapshot", "snapshot", "snapnum",
             default=DEFAULT_SNAPSHOT)
    )
    simulation_id = int(
        _arg(
            args,
            "quijotes_simulation_id",
            "simulation_id",
            "sim_id",
            default=DEFAULT_SIMULATION_ID,
        )
    )
    redshift = _arg(args, "quijotes_redshift", "redshift", default=None)
    box_min = _arg(
        args, "quijotes_box_min", "box_min", default=DEFAULT_BOX_MIN
    )
    box_max = _arg(
        args, "quijotes_box_max", "box_max", default=DEFAULT_BOX_MAX
    )
    pylians_library = _arg(
        args,
        "pylians_library",
        default=DEFAULT_PYLIANS_LIBRARY,
    )
    staging_dir = _arg(args, "staging_dir", "spill_dir", default=None)

    catalog = load_halo_catalog(
        catalog_dir,
        snapshot,
        simulation_id,
        redshift=redshift,
        box_min=box_min,
        box_max=box_max,
        pylians_library=pylians_library,
        staging_dir=staging_dir,
        readfof_module=readfof_module,
    )
    max_halos = _arg(args, "max_halos", default=None)
    if max_halos is not None:
        max_halos = int(max_halos)
        if max_halos <= 0:
            raise ValueError(f"max_halos must be positive, got {max_halos}")
        if max_halos < len(catalog):
            catalog = QuijoteHaloCatalog(
                positions=catalog.positions[:max_halos].copy(),
                masses=catalog.masses[:max_halos].copy(),
                velocities=catalog.velocities[:max_halos].copy(),
                npart=catalog.npart[:max_halos].copy(),
                snapshot=catalog.snapshot,
                redshift=catalog.redshift,
                simulation_id=catalog.simulation_id,
                source_files=catalog.source_files,
            )
    return {TRACER: catalog}, {}


def build_raw_release_table(
    zone: Any,
    real_tables: Mapping[str, QuijoteHaloCatalog],
    random_tables: Mapping[str, Any],
    tracers: Sequence[str],
    args: Namespace,
    release_tag: str,
) -> Table:
    """Build and atomically persist the raw Quijote table for one box."""

    del random_tables
    selected = list(tracers)
    if selected != [TRACER]:
        raise ValueError(
            f"Quijote requires exactly tracer {TRACER!r}, got {selected}"
        )
    try:
        catalog = real_tables[TRACER]
    except KeyError as exc:
        raise KeyError("Quijote halo catalogue was not preloaded") from exc

    n_random = int(_arg(args, "n_random", default=100))
    seed = int(
        _arg(
            args,
            "quijotes_random_seed",
            "random_seed",
            "seed",
            default=DEFAULT_RANDOM_SEED,
        )
    )
    box_min = _arg(
        args, "quijotes_box_min", "box_min", default=DEFAULT_BOX_MIN
    )
    box_max = _arg(
        args, "quijotes_box_max", "box_max", default=DEFAULT_BOX_MAX
    )
    table = build_raw_table(
        catalog,
        n_random,
        seed=seed,
        box_min=box_min,
        box_max=box_max,
        zone=int(zone),
        tracer=TRACER,
        release_tag=release_tag,
        periodic=bool(_arg(args, "periodic", default=True)),
    )
    max_halos = _arg(args, "max_halos", default=None)
    if max_halos is not None:
        table.meta["MAXHALO"] = int(max_halos)

    output_dir = _as_path(_arg(args, "raw_out"))
    out_tag = _arg(args, "out_tag", default=None)
    if out_tag is None:
        out_tag = default_output_tag(
            catalog.simulation_id, catalog.snapshot
        )
    filename = f"zone_{_zone_tag(zone)}{_safe_tag(out_tag)}.fits.gz"
    atomic_write_fits(table, output_dir / filename)
    return table


def create_config(args: Namespace) -> ReleaseConfig:
    """Create a Quijote release configuration."""

    zone = _arg(args, "zone", default=None)
    if zone is not None and int(zone) != 0:
        raise RuntimeError("Quijote contains one periodic box; only zone 0 is valid")
    zones_arg = _arg(args, "zones", default=None)
    if zones_arg is not None:
        parsed_zones = [int(value) for value in zones_arg]
        if parsed_zones != [0]:
            raise RuntimeError(
                "Quijote contains one periodic box; --zones must be exactly 0"
            )

    box_min = _arg(args, "box_min", default=DEFAULT_BOX_MIN)
    box_max = _arg(args, "box_max", default=DEFAULT_BOX_MAX)
    low, high = validate_box_bounds(box_min, box_max)
    periodic = bool(_arg(args, "periodic", default=True))
    snapshot = int(_arg(args, "snapnum", "snapshot", default=DEFAULT_SNAPSHOT))
    simulation_id = int(
        _arg(args, "simulation_id", "sim_id", default=DEFAULT_SIMULATION_ID)
    )
    redshift = snapshot_redshift(
        snapshot, _arg(args, "redshift", default=None)
    )
    seed = int(_arg(args, "seed", "random_seed", default=DEFAULT_RANDOM_SEED))
    if _arg(args, "out_tag", default=None) is None:
        # ``main`` uses this same namespace for every downstream product, so
        # assigning the default here prevents collisions in raw, pairs,
        # classification, and probability files alike.
        args.out_tag = default_output_tag(simulation_id, snapshot)

    register_tracer_mapping({TRACER: 0})

    product_meta = {
        "SNAPNUM": snapshot,
        "SIMID": simulation_id,
        "REDSHFT": redshift,
        "RNGSEED": seed,
    }
    max_halos = _arg(args, "max_halos", default=None)
    if max_halos is not None:
        product_meta["MAXHALO"] = int(max_halos)

    def _preload(
        parsed_args: Namespace,
        selected_tracers: Sequence[str] | None = None,
    ) -> tuple[dict[str, QuijoteHaloCatalog], dict[str, Any]]:
        return preload_quijotes(parsed_args, selected_tracers)

    def _build(
        current_zone: Any,
        real_tables: Mapping[str, QuijoteHaloCatalog],
        random_tables: Mapping[str, Any],
        selected_tracers: Sequence[str],
        parsed_args: Namespace,
        release_tag: str,
    ) -> Table:
        return build_raw_release_table(
            current_zone,
            real_tables,
            random_tables,
            selected_tracers,
            parsed_args,
            release_tag,
        )

    kwargs: dict[str, Any] = {
        "name": "QUIJOTES",
        "release_tag": "QUIJOTES",
        "tracers": TRACERS,
        "tracer_alias": TRACER_ALIAS,
        "real_suffix": None,
        "random_suffix": None,
        "n_random_files": 0,
        "real_columns": (),
        "random_columns": (),
        "use_dr2_preload": False,
        "preload_kwargs": {},
        "zones": [0],
        "build_raw": _build,
        "combine_outputs": False,
        "periodic_box": (low, high) if periodic else None,
        "product_meta": product_meta,
        "supports_plot": False,
        "supports_groups": False,
    }
    # The main agent adds this optional hook to ReleaseConfig.  Keeping the
    # guard makes this module importable while that coordinated change lands.
    if "preload" in getattr(ReleaseConfig, "__dataclass_fields__", {}):
        kwargs["preload"] = _preload
    return ReleaseConfig(**kwargs)


__all__ = [
    "QuijoteHaloCatalog",
    "SNAPSHOT_REDSHIFTS",
    "atomic_write_fits",
    "build_raw_release_table",
    "build_raw_table",
    "create_config",
    "default_output_tag",
    "discover_realizations",
    "halo_catalog_from_fof",
    "import_readfof",
    "load_fof_catalog",
    "load_halo_catalog",
    "preload_quijotes",
    "resolve_realization_parts",
    "snapshot_redshift",
    "validate_box_bounds",
    "validate_positions",
]
