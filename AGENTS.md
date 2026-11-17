# Project guide

Use the repository root for installs, tests, and experiment commands. `code/src/cs2_pipeline/` contains the Python package. `configs/` contains the data manifests and evaluation plans. `scripts/` contains import, analysis, and audit commands.

Use Python 3.11+ and the versions in `uv.lock`.

```sh
uv sync --frozen --extra dev --extra ml --extra context
uv run --frozen python -m pytest -q
uv run --frozen ruff check code/src scripts tests
```

The integration tests use the local public fixture and, where indicated, archived data. Follow the [data guide](data/README.md) to prepare them.

Keep replay files, generated datasets, trained models, and run logs in their ignored directories. Treat frozen manifests as inputs to the existing audit checks. Write each new run to a fresh output path and record the data split and evaluation scope with the results.
