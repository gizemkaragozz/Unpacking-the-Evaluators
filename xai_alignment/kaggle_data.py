"""Anonymous Kaggle acquisition helpers used by the medical dataset pipelines."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def download_kaggle_dataset(
    dataset_slug: str,
    target_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Download and unzip a public Kaggle dataset without embedding credentials.

    Authentication is handled by the Kaggle client through ``KAGGLE_USERNAME`` /
    ``KAGGLE_KEY`` or the user's standard Kaggle configuration file. Credential
    values are never accepted as function arguments, printed, or stored here.
    """
    destination = Path(target_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if any(destination.iterdir()) and not force:
        return destination

    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError(
            "The Kaggle command-line client is required. Install the study extra "
            "and configure Kaggle authentication before downloading the data."
        )

    command = [
        executable,
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "--unzip",
        "-p",
        str(destination),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed for dataset {dataset_slug!r}. "
            "Check the Kaggle client authentication and dataset access."
        )
    return destination
