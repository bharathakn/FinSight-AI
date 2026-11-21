"""Storage abstraction (Plan §8.0): one function to fetch a file from anywhere.

    local path / file://      → copy
    http(s)://                → urllib download (GitHub Release assets; supports a GITHUB_TOKEN for private repos)
    s3://, gs://, …           → fsspec (install s3fs / gcsfs; e.g. Supabase Storage or Cloudflare R2 via s3://)

    python -m finsight.storage fetch <uri> <dest_dir>   # download + extract an artifact bundle

The services call `ensure_bundle()` at startup: if FINSIGHT_ARTIFACTS_DIR is missing and FINSIGHT_ARTIFACTS_URI
is set, the bundle is downloaded and extracted once. Docker images normally bake the bundle in at build time
instead (immutable images), so this path is for platforms with a writable disk and for local experiments.
"""
from __future__ import annotations

import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def fetch(uri: str, dest: Path) -> Path:
    """Copy/download `uri` to the file `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if uri.startswith(("http://", "https://")):
        req = urllib.request.Request(uri, headers={"Accept": "application/octet-stream"})
        token = os.environ.get("GITHUB_TOKEN")
        if token and "github" in uri:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:   # noqa: S310 — URL from config
            shutil.copyfileobj(r, f)
    elif "://" in uri and not uri.startswith("file://"):
        import fsspec

        with fsspec.open(uri, "rb") as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copy2(uri.removeprefix("file://"), dest)
    return dest


def extract_bundle(tarball: Path, dest_dir: Path) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(dest_dir, filter="data")          # 'data' filter blocks path traversal / special files
    return dest_dir


def ensure_bundle(artifacts_dir: Path, artifacts_uri: str | None) -> Path:
    artifacts_dir = Path(artifacts_dir)
    if (artifacts_dir / "manifest.json").exists():
        return artifacts_dir
    if not artifacts_uri:
        raise FileNotFoundError(f"No artifact bundle at {artifacts_dir} and FINSIGHT_ARTIFACTS_URI is not set "
                                "(run `make bundle` or set the URI).")
    with tempfile.TemporaryDirectory() as tmp:
        tb = fetch(artifacts_uri, Path(tmp) / "bundle.tar.gz")
        extract_bundle(tb, artifacts_dir)
    return artifacts_dir


def main(argv=None):
    a = argv or sys.argv[1:]
    if len(a) != 3 or a[0] != "fetch":
        sys.exit("usage: python -m finsight.storage fetch <uri> <dest_dir>")
    print(ensure_bundle(Path(a[2]), a[1]))


if __name__ == "__main__":
    main()
