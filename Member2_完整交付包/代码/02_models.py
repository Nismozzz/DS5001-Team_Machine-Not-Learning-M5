#!/usr/bin/env python3
"""Member 2 baselines and LightGBM. See MEMBER2_README.md for commands."""
import argparse
from pathlib import Path

from modeling.common import write_json
from modeling.contracts import verify_run
from modeling.pipeline import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'validate', 'backtest', 'test', 'forecast', 'calibrate', 'verify'])
    parser.add_argument('--data', type=Path, default=Path('data/processed'))
    parser.add_argument('--scope', choices=['development', 'final'], default='development')
    parser.add_argument('--config', type=Path, default=Path('configs/model_config.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.mode == 'verify':
        print(verify_run(args.output), flush=True)
    else:
        execute(args)


if __name__ == '__main__':
    main()
