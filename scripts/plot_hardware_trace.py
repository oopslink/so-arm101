#!/usr/bin/env python3
"""Plot historical trace files only. Does not import robot code or connect devices."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--max-step', type=int, required=True,
                   help='Last sampled policy step; exclude teardown using the matching log')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in a.trace.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['step'] <= a.max_step]
    if not rows:
        raise ValueError('No policy trace rows selected')
    t = [r['timestamp'] - rows[0]['timestamp'] for r in rows]
    joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained', sharex=True)
    fig.suptitle('Historical 60 s rollout — sparse policy-phase samples only', fontsize=16)
    for ax, joint in zip(axes.flat, joints):
        k = joint + '.pos'
        ax.plot(t, [r['requested'][k] for r in rows], label='requested target', color='#2766bd')
        ax.plot(t, [r['actual'][k] for r in rows], label='last observed position', color='#dc7b29')
        ax.set_title(joint)
        ax.set_ylabel('0–100 scale' if joint == 'gripper' else 'degrees')
        ax.grid(alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds since first saved action sample')
    axes[0, 0].legend(fontsize=9)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=150)
    print(f'Saved {a.output}; {len(rows)} sparse rows; not a full-rate trajectory or success metric.')


if __name__ == '__main__':
    main()
