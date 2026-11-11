"""Build the results bundle as a reproducible tarball.

The bundle holds every result file the analyses read, under the same paths as in the
repository (``results/...``): extract it in a checkout of the release commit and
``analyze.py``, ``secondary.py`` and the ``--summary`` modes of ``timing.py`` and
``memory.py`` run on it unchanged. ``docs/results-release.md`` becomes ``results/README.md``
in the bundle; it lists the files and their columns.

Third-party results (aeon's published results and the Hydra paper's) are not
redistributed; the scripts download them again when they are missing. ``RELEASE.json``
records the SHA-256 of the copies the analyses used, and ``--check-third-party`` compares
a new download with them.

The archive is deterministic: sorted entries, the commit time as every file's mtime, no
owner names, gzip without a timestamp. Building twice from the same commit and results
with the same Python (and zlib) gives the same bytes; the manifest checksums do not depend
on the build environment.

Usage:
    uv run python experiments/release.py [--name results-v1]
    uv run python experiments/release.py --verify dist/hanner-hydra-results-v1.tar.gz
    uv run python experiments/release.py --check-third-party   # after extracting the bundle
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "docs" / "results-release.md"
DIST = ROOT / "dist"

# Directories are included recursively; every path is relative to results/.
INCLUDE = [
    "benchmark/m2/config.json",
    "benchmark/m2/results.csv",
    "benchmark/m2/dataset_means.csv",
    "benchmark/m2/dataset_means_ucr112.csv",
    "benchmark/m2/dataset_means_uea26.csv",
    "benchmark/m2/dataset_means_fresh.csv",
    "benchmark/m2/predictions",
    "benchmark/m2/secondary",
    "benchmark/ablations/config.json",
    "benchmark/ablations/results.csv",
    "benchmark/ablations/predictions",
    "timing/endpoint",
    "memory/endpoint/memory.csv",
    "reproduce_hydra/hydra_ucr112.csv",
]
# Downloaded by the scripts; only their checksums are recorded.
THIRD_PARTY = [
    "benchmark/published_hydra_ucr112_30.csv",
    "benchmark/published_WEASEL-2.0_ucr112_30.csv",
    "benchmark/published_MiniROCKET_ucr112_30.csv",
    "benchmark/published_MultiROCKET_ucr112_30.csv",
    "benchmark/published_HC2_ucr112_30.csv",
    "reproduce_hydra/reference_hydra_ucr112.csv",
    "reproduce_hydra/paper_hydra_ucr112.csv",
]
# The protocol was frozen in the private development repository before any M2 run
# (docs/preregistration.md, "Status"); its history is not public, so the freeze is
# recorded as data rather than looked up as a git tag.
PREREGISTRATION = {
    "frozen": "2026-11-04",
    "commit": "3b0cd46c56dfc6fae2a6f2e27af3fe0bdc3dbe5a",
    "document_sha256": "33213ced23ed9558c6a296483e0e9c3960f2e2aadf6ea9d2a4b60572bdeb9363",
}
# SPDX identifier of the results licence (docs/results-release.md, "Licence").
RESULTS_LICENSE = "CC-BY-4.0"
# Bundle files at the top of results/ that are not in the manifest.
BUNDLE_META = {"MANIFEST.sha256", "RELEASE.json", "README.md"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def collect(results: Path = RESULTS) -> list[str]:
    files: list[str] = []
    for entry in INCLUDE:
        path = results / entry
        if path.is_dir():
            files += [
                p.relative_to(results).as_posix()
                for p in path.rglob("*")
                if p.is_file() and p.name != ".DS_Store"
            ]
        elif path.is_file():
            files.append(entry)
        else:
            sys.exit(f"missing: results/{entry}")
    return sorted(files)


def add(tar: tarfile.TarFile, name: str, data: bytes, mtime: int) -> None:
    info = tarfile.TarInfo(f"results/{name}")
    info.size, info.mtime, info.mode = len(data), mtime, 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def build(
    name: str,
    *,
    results: Path = RESULTS,
    readme: Path = README,
    dist: Path = DIST,
    allow_dirty: bool = False,
    allow_missing_third_party: bool = False,
) -> Path:
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not allow_dirty:
        sys.exit("uncommitted changes: build from a clean checkout (or pass --allow-dirty)")
    missing = [f for f in THIRD_PARTY if not (results / f).is_file()]
    if missing and not allow_missing_third_party:
        sys.exit(
            "third-party results missing, so their checksums cannot be recorded "
            f"(run the analyses first, or pass --allow-missing-third-party): {missing}"
        )
    mtime = int(git("log", "-1", "--format=%ct"))
    files = collect(results)
    contents = {f: (results / f).read_bytes() for f in files}
    meta = {
        "name": name,
        "commit": commit,
        "dirty": dirty,
        "preregistration": PREREGISTRATION,
        "license": RESULTS_LICENSE,  # the results; the code is BSD-3-Clause
        "n_files": len(files),
        "third_party_sha256": {
            f: None if f in missing else sha256((results / f).read_bytes()) for f in THIRD_PARTY
        },
    }
    extra = {
        "README.md": readme.read_bytes(),
        "RELEASE.json": (json.dumps(meta, indent=2) + "\n").encode(),
        "MANIFEST.sha256": "".join(f"{sha256(contents[f])}  {f}\n" for f in files).encode(),
    }
    assert set(extra) == BUNDLE_META

    dist.mkdir(parents=True, exist_ok=True)
    out = dist / f"hanner-hydra-{name}.tar.gz"
    with (
        out.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for f in sorted(extra):
            add(tar, f, extra[f], mtime)
        for f in files:
            add(tar, f, contents[f], mtime)
    digest = sha256(out.read_bytes())
    print(f"{out}: {len(files)} files, {out.stat().st_size / 2**20:.1f} MB")
    print(f"sha256 {digest}  (put this in the release notes)")
    print(f"commit {commit}{' (DIRTY: rebuild from a clean checkout to release)' if dirty else ''}")
    if missing:
        print(f"WARNING: no checksum recorded for {len(missing)} third-party files: {missing}")
    return out


def verify(archive: Path) -> None:
    """Check an archive against its manifest: no mismatched, missing or unlisted files."""
    with tempfile.TemporaryDirectory() as tmp, tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")
        base = Path(tmp) / "results"
        lines = (base / "MANIFEST.sha256").read_text().splitlines()
        listed = {path: digest for digest, path in (line.split("  ", 1) for line in lines)}
        present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}
        present -= BUNDLE_META  # top-level bundle files only, not same-named files deeper
        missing = sorted(set(listed) - present)
        unlisted = sorted(present - set(listed))
        bad = sorted(
            f for f in set(listed) & present if sha256((base / f).read_bytes()) != listed[f]
        )
        if bad or missing or unlisted:
            for label, group in (("mismatch", bad), ("missing", missing), ("unlisted", unlisted)):
                for f in group[:20]:
                    print(f"{label}: {f}")
            sys.exit(
                f"FAILED: {len(bad)} checksum mismatches, {len(missing)} missing, "
                f"{len(unlisted)} unlisted"
            )
        meta = json.loads((base / "RELEASE.json").read_text())
        print(f"OK: {len(listed)} files match the manifest (commit {meta['commit'][:7]})")


def check_third_party(results: Path = RESULTS) -> None:
    """Compare the downloaded third-party results with the checksums in RELEASE.json."""
    meta_path = results / "RELEASE.json"
    if not meta_path.is_file():
        sys.exit(f"{meta_path} not found: extract the bundle first")
    recorded: dict[str, str | None] = json.loads(meta_path.read_text())["third_party_sha256"]
    changed = []
    for f, expected in sorted(recorded.items()):
        path = results / f
        if not path.is_file():
            status = "not downloaded yet (run the analyses that use it)"
        elif expected is None:
            status = "no checksum recorded in this release"
        elif sha256(path.read_bytes()) == expected:
            status = "OK"
        else:
            status = "CHANGED upstream"
            changed.append(f)
        print(f"{status}: {f}")
    if changed:
        sys.exit(f"{len(changed)} third-party files differ from the ones the analyses used")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="results-v1")
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE")
    parser.add_argument("--check-third-party", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="not for a release")
    parser.add_argument("--allow-missing-third-party", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.check_third_party:
        check_third_party()
    else:
        build(
            args.name,
            allow_dirty=args.allow_dirty,
            allow_missing_third_party=args.allow_missing_third_party,
        )


if __name__ == "__main__":
    main()
