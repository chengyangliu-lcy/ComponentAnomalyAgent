from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from tools.dataset_parser import DatasetParser
from tools.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset and local image mapping.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    result = DatasetParser(config.dataset_path, config.image_root).validate()
    output = Path(args.output) if args.output else config.outputs_dir / "data_validation.json"
    write_json(output, result)
    print(result)
    print(f"[validate] wrote {output}")


if __name__ == "__main__":
    main()

