"""The core package imports and works without aeon (the aeon estimators load lazily)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_without_aeon(code: str) -> None:
    """Run ``code`` in a fresh interpreter in which ``import aeon`` fails."""
    prelude = "import sys; sys.modules['aeon'] = None  # blocks the import, find_spec -> None\n"
    subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], check=True, timeout=120)


def test_star_import_without_aeon():
    _run_without_aeon(
        """
        from hanner_hydra import *
        assert "HannerHydraTransform" in dir() and "make_classifier" in dir()
        assert "HannerHydraClassifier" not in dir()
        """
    )


def test_estimators_without_aeon_give_a_clear_error():
    _run_without_aeon(
        """
        import hanner_hydra
        assert "HannerHydraClassifier" not in hanner_hydra.__all__
        assert "HannerHydraClassifier" not in dir(hanner_hydra)
        for name in ("HannerHydraClassifier", "HannerHydraTransformer"):
            try:
                getattr(hanner_hydra, name)
            except ImportError as e:
                assert "hanner-hydra[aeon]" in str(e), e
            else:
                raise AssertionError(f"{name} loaded without aeon")
        try:
            from hanner_hydra import HannerHydraClassifier
        except ImportError as e:
            assert "hanner-hydra[aeon]" in str(e), e
        else:
            raise AssertionError("imported without aeon")
        """
    )


def test_unknown_attribute_is_attribute_error():
    import hanner_hydra

    assert not hasattr(hanner_hydra, "no_such_name")
