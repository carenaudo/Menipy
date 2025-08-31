# Runs the command-line interface:  python -m adsa
from .menipy.cli import main as cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())