#!/usr/bin/env python3
"""Merge checked local batches using the LeRobot 0.6.1 aggregation API."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-root', required=True)
    p.add_argument('--repo-prefix', required=True)
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repo-id', required=True)
    a = p.parse_args()
    if a.batches < 1 or a.output.exists():
        p.error('batches must be positive; output must not exist')
    roots = [Path(f'{a.base_root}_batch_{i:03d}') for i in range(1, a.batches + 1)]
    infos = [json.loads((r / 'meta/info.json').read_text()) for r in roots]
    from lerobot.datasets.aggregate import aggregate_datasets
    aggregate_datasets(
        repo_ids=[f'{a.repo_prefix}_batch_{i:03d}' for i in range(1, a.batches + 1)],
        roots=roots, aggr_repo_id=a.repo_id, aggr_root=a.output,
        concatenate_videos=False, concatenate_data=False,
    )
    got = json.loads((a.output / 'meta/info.json').read_text())
    for key in ('total_episodes', 'total_frames'):
        expected = sum(x[key] for x in infos)
        if got[key] != expected:
            raise RuntimeError(f'{key}: {got[key]} != {expected}')
        print(f'{key}: {got[key]}')
    print('Counts passed; still run decoded-image and semantic checks.')


if __name__ == '__main__':
    main()
