from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hida_baa.trainers import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train HiDA-TargetModel from a HiDA-SourceModel checkpoint."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/target.yaml",
        help="Path to YAML config."
    )
    args = parser.parse_args()

    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
