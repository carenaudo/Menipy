# Contributing to Menipy

Thanks for your interest in contributing! This file covers the basic steps to run the code-quality checks we added and how to run them locally on Windows and Unix-like systems.

## Quick setup
- Create and activate a virtual environment (Windows PowerShell):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -e .
  pip install --upgrade black ruff mypy pre-commit
  ```

- Install pre-commit hooks (optional):
  ```powershell
  pre-commit install
  pre-commit run --all-files
  ```

## Running individual checks
- Black (formatting):
  ```powershell
  black --check .
  ```

- Ruff (lint/auto-fix):
  ```powershell
  # Save output to a file because direct terminal reading may block in some shells
  ruff check . > ruff_report.txt 2>&1

  # Inspect the report (PowerShell):
  Get-Content ruff_report.txt -Tail 200
  ```
  To auto-fix trivial issues:
  ```powershell
  ruff check --fix .
  ```

- Mypy (type checks):
  ```powershell
  mypy src --ignore-missing-imports > mypy_report.txt 2>&1
  Get-Content mypy_report.txt -Tail 200
  ```

Notes:
- We save `ruff`/`mypy` outputs to files and read them (instead of piping directly to the terminal) to avoid cases where the interactive terminal appears to freeze on long outputs.
- If you see lots of type/import errors in `mypy`, start by adding `# type: ignore` to third-party imports or by incrementally enabling checks in `mypy.ini`.

## Creating a fix PR
1. Branch from `main` (e.g., `chore/fix-ruff-issues`).
2. Run `ruff check --fix .` and `black .` locally, run the tests `pytest`.
3. Commit only formatting/auto-fix changes in a separate commit and push.
4. Open a PR and link any failing checks from CI in the description.

If you want help triaging the reports or preparing a fix PR, open an issue or ping a maintainer — I'm happy to help.
