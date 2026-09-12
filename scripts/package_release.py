#!/usr/bin/env python3
"""Package an existing final checkpoint; never replaces trained processors with base templates."""
import argparse
import json
import re
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; use a new release directory')
    required = ['config.json', 'model.safetensors', 'train_config.json',
                'policy_preprocessor.json', 'policy_postprocessor.json']
    for name in required:
        if not (a.checkpoint / name).is_file():
            raise FileNotFoundError(a.checkpoint / name)
    for log in (a.log, a.metrics):
        if re.search(r'(?:hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_-]{20,})', log.read_text()):
            raise ValueError(f'Possible credential in {log}; redact a copy before packaging')
    a.output.mkdir(parents=True)
    files = set(required)
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in json.loads((a.checkpoint / name).read_text())['steps']:
            if 'state_file' in step:
                files.add(step['state_file'])
    for name in files:
        src = (a.checkpoint / name).resolve()
        if not src.is_relative_to(a.checkpoint.resolve()):
            raise ValueError('state_file escapes checkpoint')
        dest = a.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.tokenizer.resolve()), local_files_only=True)
    tokenizer.save_pretrained(a.output / 'tokenizer')
    # Change only the tokenizer locator. Preserve all trained normalization state references.
    pre_path = a.output / 'policy_preprocessor.json'
    pre = json.loads(pre_path.read_text())
    for step in pre['steps']:
        if step['registry_name'] == 'tokenizer_processor':
            step['config']['tokenizer_name'] = './tokenizer'
    pre_path.write_text(json.dumps(pre, indent=2) + '\n')
    shutil.copy2(a.log, a.output / 'training.log')
    shutil.copy2(a.metrics, a.output / 'metrics.csv')
    print('Packaged. Must run verify_release.py before upload:', a.output)


if __name__ == '__main__':
    main()
