#!/usr/bin/env python3
"""Export displayed LeRobot log metrics without inventing exact rounded step counts."""
import argparse
import csv
import re
from pathlib import Path

FIELDS = ['step', 'step_display', 'step_is_approximate', 'loss', 'grad_norm',
          'learning_rate', 'update_s', 'data_s', 'samples_per_s', 'memory_gb']
KEYS = {'loss': 'loss', 'grdn': 'grad_norm', 'lr': 'learning_rate',
        'updt_s': 'update_s', 'data_s': 'data_s', 'smp/s': 'samples_per_s', 'mem_gb': 'memory_gb'}


def parse(text):
    rows = []
    for line in text.replace('\r', '\n').splitlines():
        step = re.search(r'\bstep:([0-9.]+[KMG]?)\b', line)
        if not step or 'loss:' not in line:
            continue
        displayed = step.group(1)
        suffix = displayed[-1] if displayed[-1] in 'KMG' else ''
        value = float(displayed[:-1] if suffix else displayed) * {'': 1, 'K': 1e3, 'M': 1e6, 'G': 1e9}[suffix]
        row = {'step': int(value), 'step_display': displayed,
               'step_is_approximate': bool(suffix)}
        for key, name in KEYS.items():
            match = re.search(re.escape(key) + r':\s*(\S+)', line)
            row[name] = match.group(1) if match else ''
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('csv', type=Path)
    a = p.parse_args()
    rows = parse(a.log.read_text(errors='replace'))
    if not rows:
        raise SystemExit('No metric records found; inspect log format/training state')
    a.csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.csv.with_suffix(a.csv.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(a.csv)
    print(f'{len(rows)} rows; abbreviated step counters remain approximate: {a.csv}')


if __name__ == '__main__':
    main()
