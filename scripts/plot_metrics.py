#!/usr/bin/env python3
"""Plot saved metrics on Mac/Linux. Raw values and rolling mean are both shown."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = list(csv.DictReader(a.csv.open()))
    if not rows:
        raise ValueError('CSV is empty')
    x = np.array([float(r['step']) for r in rows])
    series = [('loss', 'Loss'), ('learning_rate', 'Learning rate'),
              ('grad_norm', 'Gradient norm'), ('samples_per_s', 'Samples / second'),
              ('memory_gb', 'Logger GPU memory (GB)')]
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True, layout='constrained')
    for ax, (key, title) in zip(axes, series):
        y = np.array([float(r[key]) if r.get(key) else float('nan') for r in rows])
        ax.plot(x, y, alpha=.25, linewidth=.7, label='raw')
        avg = np.array([np.nanmean(y[max(0, i-25):i+26]) for i in range(len(y))])
        ax.plot(x, avg, linewidth=1.5, label='up to 51-point mean')
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
    approximate = any(r.get('step_is_approximate', '').lower() == 'true' for r in rows)
    axes[-1].set_xlabel('Displayed step (rounded at large values)' if approximate else 'Step (see CSV provenance)')
    axes[0].legend()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150)
    print(a.output)


if __name__ == '__main__':
    main()
