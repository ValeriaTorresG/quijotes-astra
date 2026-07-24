from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from argparse import Namespace
from astropy.table import Table


@dataclass(frozen=True)
class ReleaseConfig:
    """
    Container with release-specific configuration needed by the pipeline.
    """
    name: str
    release_tag: str
    tracers: List[str]
    tracer_alias: Dict[str, str]
    real_suffix: Dict[str, str] | None
    random_suffix: Dict[str, str] | None
    n_random_files: int
    real_columns: Sequence[str]
    random_columns: Sequence[str]
    use_dr2_preload: bool
    preload_kwargs: Dict[str, Any]
    zones: List[Any]
    build_raw: Callable[[Any, Dict[str, Any], Dict[str, Any], List[str], Namespace, str], Table]
    combine_outputs: bool = True
    preload: Optional[Callable[[Namespace, List[str]], Any]] = None
    periodic_box: Any = None
    product_meta: Dict[str, Any] = field(default_factory=dict)
    supports_plot: bool = True
    supports_groups: bool = True
