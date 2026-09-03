#!/usr/bin/env python3
"""Download the casting datasets from Kaggle into ``data/``.

Credentials are read from the environment (or a local ``.env``) — never
hard-coded. Create a token at https://www.kaggle.com/settings -> API.

    export KAGGLE_USERNAME=...
    export KAGGLE_KEY=...
    python scripts/download_dataset.py --which primary

Datasets
--------
primary   simmoshaikh/casting-defect-detection
          12-class SYNTHETIC augmented casting defect images -> data/raw/
external  ravirajsinh45/real-life-industrial-dataset-of-casting-product
          7,348 REAL photographs, binary ok/defective -> data/external/
          OPT-IN ONLY. Not part of the assigned experiment: it is neither
          trained on nor used for any headline metric. Available strictly for
          the Phase-12 qualitative generalization check, because the primary
          dataset's defect labels are synthetic. See PROJECT_DECISIONS.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, resolve_path  # noqa: E402

LOG = get_logger("download")

DATASETS = {
    "primary": ("simmoshaikh/casting-defect-detection", "data/raw"),
    "external": (
        "ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        "data/external",
    ),
}


def _load_dotenv() -> None:
    """Populate os.environ from a project-root .env if present."""
    env_file = resolve_path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _check_credentials() -> None:
    """Fail early with an actionable message if Kaggle creds are absent."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if kaggle_json.exists():
        return
    LOG.error(
        "No Kaggle credentials found.\n"
        "  Option A (env vars):\n"
        "      export KAGGLE_USERNAME=your_username\n"
        "      export KAGGLE_KEY=your_api_key\n"
        "  Option B (token file):\n"
        "      mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
        "      chmod 600 ~/.kaggle/kaggle.json\n"
        "  Get a token at https://www.kaggle.com/settings -> API -> Create New Token"
    )
    raise SystemExit(2)


def download(slug: str, dest: str, force: bool = False) -> Path:
    """Download and unzip one Kaggle dataset into ``dest``."""
    import kaggle  # imported late: it authenticates at import time

    out_dir = resolve_path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in out_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"]
    if existing and not force:
        LOG.info("%s already populated (%d files) — skipping. Use --force to redownload.",
                 out_dir, len(existing))
        return out_dir

    LOG.info("Downloading %s -> %s", slug, out_dir)
    kaggle.api.dataset_download_files(slug, path=str(out_dir), unzip=True, quiet=False)
    n = sum(1 for p in out_dir.rglob("*") if p.is_file())
    LOG.info("Done: %d files in %s", n, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", choices=[*DATASETS, "all"], default="primary",
                        help="Which dataset to fetch (default: primary — the assigned dataset)")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if the target directory is populated")
    args = parser.parse_args()

    _load_dotenv()
    _check_credentials()

    targets = list(DATASETS) if args.which == "all" else [args.which]
    for name in targets:
        slug, dest = DATASETS[name]
        download(slug, dest, force=args.force)


if __name__ == "__main__":
    main()
