"""experiments/release.py on a small fake results tree: determinism, verify, checksums."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("release", ROOT / "experiments" / "release.py")
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

GIT = {
    ("rev-parse", "HEAD"): "a" * 40,
    ("status", "--porcelain", "--untracked-files=no"): "",
    ("log", "-1", "--format=%ct"): "1762600000",
}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A results tree with one file, one directory (with a nested README) and one
    third-party file; git is stubbed (CI checks out without tags)."""
    results = tmp_path / "results"
    (results / "m2" / "predictions" / "A").mkdir(parents=True)
    (results / "m2" / "results.csv").write_text("dataset,accuracy\nA,0.9\n")
    (results / "m2" / "predictions" / "A" / "hanner.csv").write_text("y_true,y_pred\n1,1\n")
    (results / "m2" / "predictions" / "README.md").write_text("nested, must be listed\n")
    (results / "m2" / "predictions" / ".DS_Store").write_text("junk")
    (results / "published.csv").write_text("third party\n")
    readme = tmp_path / "README.md"
    readme.write_text("# bundle\n")
    monkeypatch.setattr(release, "INCLUDE", ["m2/results.csv", "m2/predictions"])
    monkeypatch.setattr(release, "THIRD_PARTY", ["published.csv"])
    monkeypatch.setattr(release, "git", lambda *args: GIT[args])
    return results, readme, tmp_path / "dist"


def _build(tree, **kwargs) -> Path:
    results, readme, dist = tree
    return release.build("t", results=results, readme=readme, dist=dist, **kwargs)


def _members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers()}


def _rewrite(archive: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_build_is_deterministic_and_verifies(tree, capsys):
    a = _build(tree).read_bytes()
    b = _build(tree)
    assert b.read_bytes() == a
    assert a[4:8] == b"\0\0\0\0"  # no timestamp in the gzip header
    members = _members(b)
    assert sorted(members) == [
        "results/MANIFEST.sha256",
        "results/README.md",
        "results/RELEASE.json",
        "results/m2/predictions/A/hanner.csv",
        "results/m2/predictions/README.md",
        "results/m2/results.csv",
    ]
    meta = json.loads(members["results/RELEASE.json"])
    assert meta["n_files"] == 3
    assert meta["license"] == "CC-BY-4.0"
    assert meta["third_party_sha256"]["published.csv"] == release.sha256(b"third party\n")
    release.verify(b)
    assert "OK: 3 files" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: m.update({"results/m2/results.csv": b"tampered\n"}), "1 checksum mismatches"),
        (lambda m: m.pop("results/m2/results.csv"), "1 missing"),
        (lambda m: m.update({"results/extra.csv": b"x"}), "1 unlisted"),
        (lambda m: m.update({"results/m2/README.md": b"x"}), "1 unlisted"),  # name, not path
    ],
)
def test_verify_reports_damage(tree, change, message):
    archive = _build(tree)
    members = _members(archive)
    change(members)
    _rewrite(archive, members)
    with pytest.raises(SystemExit, match=message):
        release.verify(archive)


def test_build_refuses_dirty_tree(tree, monkeypatch):
    monkeypatch.setattr(
        release, "git", lambda *args: " M x.py" if args[0] == "status" else GIT[args]
    )
    with pytest.raises(SystemExit, match="uncommitted"):
        _build(tree)
    assert json.loads(_members(_build(tree, allow_dirty=True))["results/RELEASE.json"])["dirty"]


def test_build_refuses_missing_third_party(tree):
    (tree[0] / "published.csv").unlink()
    with pytest.raises(SystemExit, match="third-party"):
        _build(tree)
    meta = json.loads(
        _members(_build(tree, allow_missing_third_party=True))["results/RELEASE.json"]
    )
    assert meta["third_party_sha256"] == {"published.csv": None}


def test_check_third_party(tree, tmp_path, capsys):
    with tarfile.open(_build(tree)) as tar:
        tar.extractall(tmp_path / "x", filter="data")
    extracted = tmp_path / "x" / "results"
    release.check_third_party(extracted)  # not downloaded yet: reported, not an error
    assert "not downloaded yet" in capsys.readouterr().out
    (extracted / "published.csv").write_text("third party\n")
    release.check_third_party(extracted)
    assert "OK: published.csv" in capsys.readouterr().out
    (extracted / "published.csv").write_text("changed upstream\n")
    with pytest.raises(SystemExit, match="1 third-party files differ"):
        release.check_third_party(extracted)
