"""Hanner-Hydra: Hanner-polytope ternary kernel dictionaries for Hydra-style time series
classification.
"""

import importlib.util
from typing import TYPE_CHECKING

from hanner_hydra.control import (
    RandomTernaryGroup,
    matched_control_group,
    symmetric_control_group,
)
from hanner_hydra.cotree import Cotree, sample_cotree
from hanner_hydra.transform import HannerHydraTransform, make_classifier

if TYPE_CHECKING:
    from hanner_hydra.estimators import HannerHydraClassifier, HannerHydraTransformer

__version__ = "0.0.1"

_AEON_ESTIMATORS = ("HannerHydraClassifier", "HannerHydraTransformer")


def _has_aeon() -> bool:
    return importlib.util.find_spec("aeon") is not None


def __getattr__(name: str) -> object:
    """The aeon estimators load on first use, so the core package does not need aeon."""
    if name in _AEON_ESTIMATORS:
        if not _has_aeon():
            raise ImportError(
                f"{name} needs aeon: pip install 'hanner-hydra[aeon]' (or uv sync --extra aeon)"
            )
        # aeon is installed: any other ImportError (e.g. a moved aeon module) propagates.
        from hanner_hydra import estimators

        return getattr(estimators, name)
    raise AttributeError(f"module 'hanner_hydra' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *(_AEON_ESTIMATORS if _has_aeon() else ())])


__all__ = [
    "Cotree",
    "HannerHydraTransform",
    "RandomTernaryGroup",
    "__version__",
    "make_classifier",
    "matched_control_group",
    "sample_cotree",
    "symmetric_control_group",
]
# Only with aeon, so that ``from hanner_hydra import *`` works on the core install.
if _has_aeon():
    __all__ += ["HannerHydraClassifier", "HannerHydraTransformer"]
