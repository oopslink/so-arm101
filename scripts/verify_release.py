#!/usr/bin/env python3
"""Offline PI0 release checks. Never connects to a robot. MEAN_STD absolute actions only."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--load-weights', action='store_true', help='Needs sufficient CPU RAM')
    p.add_argument('--write-manifest', action='store_true')
    a = p.parse_args()
    root = a.model.resolve()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    for name in ('config.json', 'model.safetensors', 'train_config.json',
                 'policy_preprocessor.json', 'policy_postprocessor.json'):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    post = json.loads((root / 'policy_postprocessor.json').read_text())
    un = next(s for s in post['steps'] if s['registry_name'] == 'unnormalizer_processor')
    if un['config'].get('features', {}).get('action', {}).get('shape') != [6]:
        raise ValueError('Missing/wrong action features: expected shape [6]')
    if un['config']['norm_map']['ACTION'] != 'MEAN_STD':
        raise ValueError('This checker is for MEAN_STD; do not use its math for another scheme')
    if any(s['registry_name'] == 'absolute_actions_processor' and
           s['config'].get('enabled') for s in post['steps']):
        raise ValueError('Relative-action release needs a state-aware validator')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        for step in json.loads((root / name).read_text())['steps']:
            if 'state_file' in step:
                path = (root / step['state_file']).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f'Invalid state_file in {name}: {step["state_file"]}')
    state_file = un.get('state_file')
    if not state_file:
        raise ValueError('Postprocessor does not reference its trained statistics')
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    tensors = load_file(str(root / state_file))
    def stat(name):
        matches = [v for k, v in tensors.items() if k.endswith('action.' + name)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely identify action.{name}: {list(tensors)}')
        return matches[0].float().reshape(6)
    mean, std = stat('mean'), stat('std')
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
        raise ValueError('Invalid action statistics')
    pre = json.loads((root / 'policy_preprocessor.json').read_text())
    normalizer = next(s for s in pre['steps'] if s['registry_name'] == 'normalizer_processor')
    pre_stats = load_file(str(root / normalizer['state_file']))
    for name, expected in (('mean', mean), ('std', std)):
        matches = [v for k, v in pre_stats.items() if k.endswith('action.' + name)]
        if len(matches) != 1 or not torch.allclose(matches[0].float().reshape(6), expected):
            raise ValueError(f'Pre/post action.{name} statistics disagree')
    AutoTokenizer.from_pretrained(str(root / 'tokenizer'), local_files_only=True)
    cfg = PreTrainedConfig.from_pretrained(str(root))
    cfg.pretrained_path = str(root)
    cfg.device = 'cpu'
    _, processor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=str(root),
        preprocessor_overrides={'device_processor': {'device': 'cpu'},
                                 'tokenizer_processor': {'tokenizer_name': str(root / 'tokenizer')}},
    )
    for shape in ((1, 6), (1, 50, 6)):
        for z in (0.0, 1.0, -1.0):
            x = torch.full(shape, z)
            out = processor(x)
            if not torch.isfinite(out).all() or not torch.allclose(
                out.cpu(), (mean + z * std).expand(shape), atol=1e-4, rtol=1e-4
            ):
                raise ValueError(f'Actual postprocessor fails denormalization test {shape}, z={z}')
    if a.load_weights:
        from lerobot.rollout.context import _load_pretrained_policy
        policy = _load_pretrained_policy(cfg)
        print('Fine-tuned weights loaded locally:', type(policy).__name__)
    if a.write_manifest:
        lines = [f'{digest(f)}  {f.relative_to(root).as_posix()}'
                 for f in sorted(root.rglob('*')) if f.is_file()
                 and f.name != 'SHA256SUMS' and '.cache' not in f.relative_to(root).parts]
        (root / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print('PASS: local tokenizer, processor references and real denormalization. Not a task-success test.')


if __name__ == '__main__':
    main()
