#!/usr/bin/env python3
"""Minimal block-diffusion training pipeline over packed training-mixture/v1 shards.

Loads uint16 shards, applies dynamic diffusion masking, trains the smoke
transformer or D1 hybrid, and writes resumable checkpoints. CPU by default,
with optional CUDA AMP and gradient accumulation.
This module is independent of bench/ and the evaluation tooling; validation
here is masked cross-entropy on the bundle's own validation split.

Select the original smoke transformer or the D1 GDN/GQA hybrid. `--p1` stays
blocked until actual GPU validation and P0 measurements are available.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .prepare_corpus import canonical, digest, filehash
    from .d1_model import D1Config, D1LM, backend_info
else:
    from prepare_corpus import canonical, digest, filehash
    from d1_model import D1Config, D1LM, backend_info

SCHEMA = 'training-mixture/v1'
MASK_ID = 0
FORBIDDEN_IDS = (0, 2, 3, 4)  # MASK, PAD, UNK, BOS; EOS=1 is a legal target
RESERVED_IDS = (0, 1, 2, 3, 4)  # MASK, EOS, PAD, UNK, BOS are never masked
SPECIAL_IDS = {'[MASK]': 0, '[EOS]': 1, '[PAD]': 2, '[UNK]': 3, '[BOS]': 4}
CHECKPOINT_VERSION = 4


def load_manifest(bundle, verify_hashes=False):
    """Read a bundle manifest and check its fingerprint and shard sizes."""
    bundle = Path(bundle)
    meta = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(meta, dict) or meta.get('schema') != SCHEMA:
        raise ValueError(f'expected {SCHEMA} manifest in {bundle}')
    claimed = meta.pop('fingerprint', None)
    if claimed != digest(canonical(meta).encode()):
        raise ValueError('bundle manifest fingerprint mismatch')
    meta['fingerprint'] = claimed
    if meta.get('dtype') != '<u2':
        raise ValueError(f"unsupported shard dtype {meta.get('dtype')!r}")
    if type(meta.get('seq_len')) is not int or meta['seq_len'] < 1:
        raise ValueError('seq_len must be a positive integer')
    if meta.get('special_ids') != SPECIAL_IDS:
        raise ValueError('unsupported special token IDs in bundle manifest')
    for split, info in meta['splits'].items():
        if not info['shards']:
            raise ValueError(f'split {split!r} has no shards')
        for shard in info['shards']:
            if (type(shard.get('tokens')) is not int or shard['tokens'] <= 0
                    or shard['tokens'] % meta['seq_len']):
                raise ValueError('shard tokens must be positive whole sequences')
            if shard.get('sequences') != shard['tokens'] // meta['seq_len']:
                raise ValueError('shard sequence count mismatch')
            path = bundle / shard['path']
            if not path.exists():
                raise ValueError(f'missing shard {path}')
            if path.stat().st_size != 2 * shard['tokens']:
                raise ValueError(f'shard size mismatch: {path}')
            if verify_hashes and filehash(path) != shard['sha256']:
                raise ValueError(f'shard hash mismatch: {path}')
        if (info.get('tokens') != sum(s['tokens'] for s in info['shards'])
                or info.get('sequences') != sum(s['sequences'] for s in info['shards'])):
            raise ValueError(f'split {split!r} totals mismatch')
    return meta


def bundle_vocab(bundle):
    """Vocabulary size from the bundle's tokenizer manifest."""
    path = Path(bundle) / 'tokenizer_manifest.json'
    if not path.exists():
        raise ValueError(f'no tokenizer_manifest.json in {bundle}')
    info = json.loads(path.read_text(encoding='utf-8'))
    claimed = info.pop('fingerprint', None)
    if claimed != digest(canonical(info).encode()):
        raise ValueError('tokenizer manifest fingerprint mismatch')
    if info.get('schema') != 'tokenizer/v1' or info.get('special_ids') != SPECIAL_IDS:
        raise ValueError('unsupported tokenizer schema or special token IDs')
    size = info.get('vocab_size')
    if type(size) is not int or not 5 < size <= 65536:
        raise ValueError(f'invalid vocab_size in {path}')
    return size


def bundle_provenance(bundle, meta):
    """Hash the exact manifest, tokenizer manifest, and shards used by a run.

    Unlike load_manifest, this always reads shard bytes: the checkpoint must
    bind the corpus that was actually trained on, not just the manifest's
    declared hashes. Any mismatch is a hard error.
    """
    bundle = Path(bundle)
    manifest_path = bundle / 'manifest.json'
    tokenizer_path = bundle / 'tokenizer_manifest.json'
    if not tokenizer_path.exists():
        raise ValueError(f'no tokenizer_manifest.json in {bundle}')
    declared_tokenizer = meta.get('tokenizer_manifest_sha256')
    actual_tokenizer = filehash(tokenizer_path)
    if declared_tokenizer is not None and declared_tokenizer != actual_tokenizer:
        raise ValueError('tokenizer manifest hash mismatch')
    tokenizer_path = bundle / 'tokenizer.json'
    actual_tokenizer_model = None
    if meta.get('tokenizer_sha256') is not None and not tokenizer_path.is_file():
        raise ValueError('missing tokenizer model declared by manifest')
    if tokenizer_path.exists():
        actual_tokenizer_model = filehash(tokenizer_path)
        if meta.get('tokenizer_sha256') is not None and meta['tokenizer_sha256'] != actual_tokenizer_model:
            raise ValueError('tokenizer model hash mismatch')
    shards = []
    for split, info in meta['splits'].items():
        for shard in info['shards']:
            path = bundle / shard['path']
            actual = filehash(path)
            if actual != shard['sha256']:
                raise ValueError(f'shard hash mismatch: {path}')
            shards.append({'split': split, 'path': shard['path'],
                           'sha256': actual, 'tokens': shard['tokens']})
    return {
        'manifest_sha256': filehash(manifest_path),
        'fingerprint': meta['fingerprint'],
        'tokenizer_manifest_sha256': actual_tokenizer,
        'tokenizer_sha256': actual_tokenizer_model or meta.get('tokenizer_sha256'),
        'seq_len': meta['seq_len'],
        'shards': shards,
    }


class ShardLoader:
    """Seeded, resumable iteration over the packed shards of one split.

    The cursor is (epoch, position in the epoch's shard permutation, sequence
    offset inside the current shard). Batches are contiguous rows of one shard;
    the last batch of a shard may be partial. Shard order per epoch is a pure
    function of (seed, epoch), so resuming a cursor reproduces the exact stream.
    """

    def __init__(self, bundle, split='train', batch_size=8, seed=17,
                 verify_hashes=False, vocab=None, finite=False):
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self.bundle, self.split, self.batch_size, self.seed = Path(bundle), split, batch_size, seed
        self.finite = finite
        meta = load_manifest(self.bundle, verify_hashes)
        info = meta['splits'].get(split)
        if info is None:
            raise ValueError(f'bundle has no split {split!r}')
        self.seq_len = meta['seq_len']
        self.vocab = vocab
        self.shards = [(self.bundle / shard['path'], shard['tokens'] // meta['seq_len'])
                       for shard in info['shards']]
        if not self.shards:
            raise ValueError(f'split {split!r} has no shards')
        self.epoch, self.position, self.sequence = 0, 0, 0
        self._order = self._permutation(0)
        self._memmap = None
        self._memmap_path = None

    def _permutation(self, epoch):
        order = list(range(len(self.shards)))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
        return order

    def state_dict(self):
        return {'epoch': self.epoch, 'position': self.position, 'sequence': self.sequence,
                'seed': self.seed, 'split': self.split}

    def load_state_dict(self, state):
        required = {'epoch', 'position', 'sequence', 'seed', 'split'}
        if set(state) != required:
            raise ValueError(f'invalid loader state {sorted(state)}')
        if state['split'] != self.split:
            raise ValueError(f'loader state split {state["split"]!r} != {self.split!r}')
        if state['seed'] != self.seed:
            raise ValueError(f'loader state seed {state["seed"]} != {self.seed}')
        if not all(type(state[key]) is int and state[key] >= 0
                   for key in ('epoch', 'position', 'sequence', 'seed')):
            raise ValueError('loader state must be nonnegative integers')
        epoch, position, sequence = state['epoch'], state['position'], state['sequence']
        if position > len(self.shards):
            raise ValueError(f'loader position out of range: {position}')
        if position == len(self.shards):
            if sequence != 0:
                raise ValueError('loader sequence must be 0 at end of epoch')
        else:
            rows = self.shards[self._permutation(epoch)[position]][1]
            if sequence >= rows or sequence % self.batch_size:
                raise ValueError(f'loader sequence out of range or not at batch boundary: {sequence}')
        self.epoch, self.position, self.sequence = epoch, position, sequence
        self._order = self._permutation(self.epoch)
        self._memmap, self._memmap_path = None, None

    def validate_progress(self, step, grad_accum, tokens_seen):
        """Bind the cursor to exactly the consumed micro-batches and tokens."""
        if type(step) is not int or step < 1 or type(tokens_seen) is not int:
            raise ValueError('checkpoint step and tokens_seen must be positive integers')
        rows = [self.shards[i][1] for i in self._order]
        batches = [(n + self.batch_size - 1) // self.batch_size for n in rows]
        consumed_batches = (self.epoch * sum(batches) + sum(batches[:self.position])
                            + self.sequence // self.batch_size)
        consumed_tokens = (self.epoch * sum(rows) + sum(rows[:self.position])
                           + self.sequence) * self.seq_len
        if tokens_seen != consumed_tokens:
            raise ValueError('checkpoint tokens_seen inconsistent with loader cursor')
        if consumed_batches != step * grad_accum:
            raise ValueError('checkpoint loader cursor inconsistent with step and grad_accum')

    def __next__(self):
        while True:
            if self.position >= len(self._order):
                if self.finite:
                    raise StopIteration
                self.epoch += 1
                self.position, self.sequence = 0, 0
                self._order = self._permutation(self.epoch)
                self._memmap, self._memmap_path = None, None
            path, rows = self.shards[self._order[self.position]]
            if self._memmap_path != path:
                self._memmap = np.memmap(path, dtype='<u2', mode='r').reshape(-1, self.seq_len)
                self._memmap_path = path
            if self.sequence >= rows:
                self.position, self.sequence = self.position + 1, 0
                continue
            take = min(self.batch_size, rows - self.sequence)
            batch = np.asarray(self._memmap[self.sequence:self.sequence + take], dtype=np.int64)
            self.sequence += take
            if self.sequence >= rows:
                self.position, self.sequence = self.position + 1, 0
            ids = torch.from_numpy(batch)
            self._check(ids)
            return ids

    def _check(self, ids):
        if ids.numel() == 0:
            raise ValueError('empty batch')
        low, high = int(ids.min()), int(ids.max())
        if low < 1 or high >= (self.vocab if self.vocab is not None else 65536):
            raise ValueError(f'target id out of range: min={low} max={high}')
        if torch.isin(ids, torch.tensor(FORBIDDEN_IDS)).any():
            raise ValueError(f'forbidden control id in targets: {FORBIDDEN_IDS}')


def diffusion_mask(targets, *, mask_id=MASK_ID, min_ratio=0.1, max_ratio=0.9,
                   generator=None):
    """Per-token Bernoulli diffusion corruption with a per-sequence rate.

    Each row draws one rate uniformly from [min_ratio, max_ratio], then each
    non-reserved position is replaced independently with that probability.
    EOS (1) and reserved control ids are never masked. A row with at least one
    eligible position is guaranteed at least one masked token so the loss is
    always defined; this is a numerical safeguard, not a second rate draw.
    Returns (masked_inputs, loss_mask); stored targets are never modified.
    """
    if targets.ndim != 2 or targets.shape[1] < 1:
        raise ValueError('targets must be [batch, seq_len]')
    if not 0.0 < min_ratio < max_ratio < 1.0:
        raise ValueError('mask rates must satisfy 0 < min_ratio < max_ratio < 1')
    batch, length = targets.shape
    ratios = min_ratio + (max_ratio - min_ratio) * torch.rand(
        batch, device=targets.device, generator=generator)
    eligible = ~torch.isin(targets, torch.tensor(RESERVED_IDS, device=targets.device))
    draws = torch.rand(batch, length, device=targets.device, generator=generator)
    loss_mask = (draws < ratios[:, None]) & eligible
    missing = eligible & ~loss_mask.any(dim=1, keepdim=True)
    if bool(missing.any()):
        first = eligible.float().argmax(dim=1)
        rows = missing.any(dim=1)
        loss_mask[rows, first[rows]] = True
    return targets.masked_fill(loss_mask, mask_id), loss_mask


def block_diffusion_mask(targets, block_size, *, min_ratio=0.1, max_ratio=0.9,
                         generator=None, frontier=None):
    """Sample one shared frontier block; expose clean prefix and noisy frontier.

    Future tokens are cropped, so neither recurrent nor attention paths can
    access them. Sampling is uniform over blocks with eligible targets in the
    micro-batch; an all-EOS frontier cannot yield a training loss.
    """
    if (type(block_size) is not int or block_size < 1 or targets.ndim != 2
            or not targets.shape[0] or not targets.shape[1] or targets.shape[1] % block_size):
        raise ValueError('sequence length must be divisible by block_size')
    eligible = ~torch.isin(targets, torch.tensor(RESERVED_IDS, device=targets.device))
    valid = eligible.reshape(targets.shape[0], -1, block_size).any(dim=(0, 2)).nonzero().flatten()
    if not len(valid):
        raise ValueError('no eligible targets in any frontier block')
    if frontier is None:
        index = torch.randint(len(valid), (), device=targets.device, generator=generator)
        frontier = int(valid[index])
    if type(frontier) is not int or not 0 <= frontier < targets.shape[1] // block_size:
        raise ValueError('frontier block out of range')
    start, end = frontier * block_size, (frontier + 1) * block_size
    noisy, active_mask = diffusion_mask(targets[:, start:end], min_ratio=min_ratio,
                                        max_ratio=max_ratio, generator=generator)
    inputs = torch.cat((targets[:, :start], noisy), dim=1)
    loss_mask = torch.cat((torch.zeros_like(targets[:, :start], dtype=torch.bool), active_mask), dim=1)
    return inputs, loss_mask


def corrupt_batch(cfg, targets, generator=None):
    if cfg.mask_mode == 'blockwise':
        return block_diffusion_mask(targets, cfg.d1.block_size, min_ratio=cfg.mask_min,
                                    max_ratio=cfg.mask_max, generator=generator)
    return diffusion_mask(targets, min_ratio=cfg.mask_min, max_ratio=cfg.mask_max, generator=generator)


def masked_cross_entropy(logits, targets, loss_mask, reduction='mean'):
    """Cross-entropy over masked positions only."""
    if logits.shape[:2] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError('shape mismatch')
    if not loss_mask.any():
        raise ValueError('empty loss mask')
    return F.cross_entropy(logits[loss_mask], targets[loss_mask], reduction=reduction)


class DiffusionBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * mlp_ratio),
                                 nn.GELU(), nn.Linear(d_model * mlp_ratio, d_model))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class DiffusionLM(nn.Module):
    """Small bidirectional transformer trained with the diffusion objective.

    Full (non-causal) attention over the masked sequence, tied token embeddings.
    This is a pipeline smoke model, not the spec's GDN hybrid architecture.
    """

    def __init__(self, vocab_size, seq_len, d_model=64, n_layers=2, n_heads=4, mlp_ratio=4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.vocab_size, self.seq_len = vocab_size, seq_len
        self.tokens = nn.Embedding(vocab_size, d_model)
        self.positions = nn.Parameter(torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [DiffusionBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.positions, std=0.02)

    def forward(self, inputs):
        if inputs.ndim != 2 or inputs.shape[1] > self.seq_len:
            raise ValueError('expected [batch, seq_len <= max]')
        h = self.tokens(inputs) + self.positions[:inputs.shape[1]]
        for block in self.blocks:
            h = block(h)
        return self.norm(h) @ self.tokens.weight.T


@dataclass(frozen=True)
class TrainConfig:
    bundle: str
    out: str
    seed: int = 17
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 100
    warmup: int = 10
    checkpoint_every: int = 50
    eval_every: int = 50
    mask_min: float = 0.1
    mask_max: float = 0.9
    mask_mode: str = 'classical'
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    grad_clip: float = 1.0
    grad_accum: int = 1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    threads: int = 1
    device: str = 'cpu'
    amp: bool = False
    eval_batches: int = 8
    eval_full: bool = False
    profile: str = 'smoke'
    d1: D1Config = field(default_factory=D1Config)
    gdn_backend: str = 'torch'
    p1: bool = False
    verify_hashes: bool = False


# Run locations and hash checking do not change the training trajectory.
CONFIG_HASH_EXCLUDE = ('out', 'verify_hashes')


def config_hash_of(cfg):
    payload = {key: value for key, value in asdict(cfg).items() if key not in CONFIG_HASH_EXCLUDE}
    return digest(canonical(payload).encode())


def _validate(cfg):
    for name in ('lr', 'grad_clip', 'weight_decay', 'eps'):
        if not math.isfinite(getattr(cfg, name)):
            raise ValueError(f'{name} must be finite')
    for name in ('steps', 'batch_size', 'grad_accum', 'checkpoint_every', 'threads',
                 'd_model', 'n_layers', 'n_heads', 'mlp_ratio', 'eval_batches'):
        if getattr(cfg, name) < 1:
            raise ValueError(f'{name} must be >= 1')
    if cfg.seed < 0:
        raise ValueError('seed must be >= 0')
    if not 0 <= cfg.warmup < cfg.steps:
        raise ValueError('warmup must be in [0, steps)')
    if cfg.eval_every < 0:
        raise ValueError('eval_every must be >= 0')
    if cfg.eval_full and cfg.eval_batches < 1:
        raise ValueError('eval_batches must be >= 1')
    if cfg.lr <= 0 or cfg.grad_clip <= 0:
        raise ValueError('lr and grad_clip must be > 0')
    if not 0.0 < cfg.mask_min < cfg.mask_max < 1.0:
        raise ValueError('mask rates must satisfy 0 < mask_min < mask_max < 1')
    if cfg.mask_mode not in ('classical', 'blockwise'):
        raise ValueError('mask_mode must be classical or blockwise')
    if cfg.weight_decay < 0:
        raise ValueError('weight_decay must be >= 0')
    if not 0.0 <= cfg.beta1 < 1.0 or not 0.0 <= cfg.beta2 < 1.0:
        raise ValueError('betas must be in [0, 1)')
    if cfg.eps <= 0:
        raise ValueError('eps must be > 0')
    if cfg.device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('cuda device requested but no CUDA runtime is available')
    if cfg.amp and cfg.device != 'cuda':
        raise ValueError('amp requires device=cuda')
    if cfg.profile not in ('smoke', 'd1'):
        raise ValueError('profile must be smoke or d1')
    if not isinstance(cfg.d1, D1Config):
        raise ValueError('d1 must be a D1Config')
    if cfg.gdn_backend not in ('torch', 'fla'):
        raise ValueError('gdn_backend must be torch or fla')
    if cfg.p1:
        missing = ['GPU cache/gradient/resume validation and measured P0 throughput are pending']
        if cfg.profile != 'd1':
            missing.append('profile=d1')
        if cfg.device != 'cuda':
            missing.append('device=cuda')
        if not cfg.amp:
            missing.append('amp=true')
        if not cfg.eval_full:
            missing.append('eval_full=true')
        if not cfg.verify_hashes:
            missing.append('verify_hashes=true')
        if cfg.batch_size * cfg.grad_accum != 64:
            missing.append('global batch 64 sequences')
        if missing:
            raise ValueError('P1 requested without prerequisites: ' + ', '.join(missing))
    if (cfg.profile == 'd1') != (cfg.mask_mode == 'blockwise'):
        raise ValueError('D1 requires blockwise masking; smoke requires classical masking')
    if cfg.profile == 'd1' and cfg.gdn_backend == 'fla' and not (cfg.device == 'cuda' and cfg.amp):
        raise ValueError('FLA backend requires device=cuda and amp=true')


def lr_at(step, cfg):
    """Linear warmup then cosine decay to 10% of the peak learning rate."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    decay_steps = cfg.steps - cfg.warmup
    t = ((step - cfg.warmup) / max(1, decay_steps - 1)
         if decay_steps > 1 or cfg.steps == 1 else 1.0)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))))


def build_optimizer(model, cfg):
    """AdamW with recipe-style groups: decay matrices, not norms/biases/embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (param.ndim <= 1 or 'norm' in name or 'bias' in name
                or 'embedding' in name or 'positions' in name or name == 'tokens.weight'):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({'params': decay, 'weight_decay': cfg.weight_decay})
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)


def build_model(cfg, vocab, seq_len):
    if cfg.profile == 'd1':
        return D1LM(vocab, seq_len, cfg.d1, backend=cfg.gdn_backend)
    if cfg.profile != 'smoke':
        raise ValueError('profile must be smoke or d1')
    return DiffusionLM(vocab, seq_len, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio)


def optimizer_step(model, optimizer, scheduler, scaler, grad_clip):
    """Clip unscaled gradients; skip AMP overflow without advancing the schedule."""
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip, error_if_nonfinite=not scaler.is_enabled())
    skipped = not math.isfinite(float(grad_norm))
    if skipped:
        # The norm reduction can overflow even when individual gradients are finite.
        # Do not let GradScaler's per-element check apply the clipped NaNs then.
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
    else:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    return float(grad_norm), skipped


def library_versions():
    versions = {'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__}
    try:
        versions['fla'] = importlib.metadata.version('flash-linear-attention')
    except importlib.metadata.PackageNotFoundError:
        versions['fla'] = None
    return versions


def git_commit():
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_payload(model, optimizer, scheduler, scaler, loader, step, tokens_seen,
                       config_hash, provenance, tokens_processed, tokens_scored):
    return {
        'format_version': CHECKPOINT_VERSION,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'loader': loader.state_dict(),
        'step': step,
        'tokens_seen': tokens_seen,
        'tokens_processed': tokens_processed,
        'tokens_scored': tokens_scored,
        'config_hash': config_hash,
        'provenance': provenance,
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        'library_versions': library_versions(),
        'backend': backend_info(model.backend) if isinstance(model, D1LM) else None,
        'git_commit': git_commit(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_checkpoint(path, payload):
    """Atomic write with fsync and a SHA-256 sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = filehash(tmp)
    os.replace(tmp, path)
    path.with_name(path.name + '.sha256').write_text(
        f'{checksum}  {path.name}\n', encoding='utf-8')


def load_checkpoint(path, verify_integrity=True):
    path = Path(path)
    sidecar = path.with_name(path.name + '.sha256')
    if verify_integrity:
        if not sidecar.exists():
            raise ValueError(f'missing checkpoint checksum sidecar: {sidecar}')
        expected = sidecar.read_text(encoding='utf-8').split()[0]
        actual = filehash(path)
        if actual != expected:
            raise ValueError(f'checkpoint checksum mismatch: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    required = {'format_version', 'model', 'optimizer', 'scheduler', 'scaler',
                'loader', 'step', 'tokens_seen', 'config_hash', 'provenance',
                'rng', 'library_versions', 'tokens_processed', 'tokens_scored'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ValueError(f'not a training checkpoint: {path}')
    if payload['format_version'] != CHECKPOINT_VERSION:
        raise ValueError(f'unsupported checkpoint version: {payload["format_version"]}')
    return payload


def restore_rng(payload):
    rng = payload['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])


def evaluate(cfg, model, vocab, seq_len, batches=8, device='cpu'):
    """Deterministic masked cross-entropy on the validation split.

    The value is weighted by masked tokens, not batch means. With batches=None
    the loader is finite and the whole split is consumed exactly once. D1 full
    validation visits every frontier per batch, with fixed corruption draws.
    """
    loader = ShardLoader(cfg.bundle, 'validation', cfg.batch_size, cfg.seed + 1,
                         cfg.verify_hashes, vocab=vocab, finite=batches is None)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    total, count, batches_seen = 0.0, 0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            while batches is None or batches_seen < batches:
                try:
                    targets = next(loader).to(device)
                except StopIteration:
                    break
                frontiers = (range(seq_len // cfg.d1.block_size)
                             if cfg.profile == 'd1' and batches is None else [None])
                for frontier in frontiers:
                    if frontier is None:
                        inputs, loss_mask = corrupt_batch(cfg, targets, generator)
                    else:
                        inputs, loss_mask = block_diffusion_mask(
                            targets, cfg.d1.block_size, min_ratio=cfg.mask_min,
                            max_ratio=cfg.mask_max, generator=generator, frontier=frontier)
                    masked = int(loss_mask.sum())
                    if not masked:
                        continue  # all-EOS validation block
                    with torch.autocast(device_type=torch.device(device).type,
                                        dtype=torch.float16, enabled=cfg.amp):
                        loss = masked_cross_entropy(model(inputs), targets[:, :inputs.shape[1]], loss_mask)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('non-finite validation masked CE')
                    total += float(loss) * masked
                    count += masked
                batches_seen += 1
    finally:
        model.train(was_training)
    if not count:
        raise ValueError('empty validation loss mask')
    return total / count


def train(cfg, resume=None, log=print):
    _validate(cfg)
    torch.set_num_threads(cfg.threads)
    device = torch.device(cfg.device)
    meta = load_manifest(cfg.bundle, cfg.verify_hashes)
    provenance = bundle_provenance(cfg.bundle, meta)
    vocab = bundle_vocab(cfg.bundle)
    seq_len = meta['seq_len']
    config_hash = config_hash_of(cfg)
    out = Path(cfg.out)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
    model = build_model(cfg, vocab, seq_len)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())
    backend = backend_info(cfg.gdn_backend) if cfg.profile == 'd1' else None
    log(json.dumps({'model': cfg.profile, 'params': params, 'backend': backend,
                    'metric': 'masked_ce'}))
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_at(step, cfg) / cfg.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp)
    loader = ShardLoader(cfg.bundle, 'train', cfg.batch_size, cfg.seed, cfg.verify_hashes,
                         vocab=vocab)
    step, tokens_seen, tokens_processed, tokens_scored = 0, 0, 0, 0
    if resume:
        payload = load_checkpoint(resume)
        if payload['config_hash'] != config_hash:
            raise ValueError('resume rejected: config hash mismatch')
        if payload.get('backend') != backend:
            raise ValueError('resume rejected: GDN backend version mismatch')
        if payload['provenance'] != provenance:
            raise ValueError('resume rejected: corpus/tokenizer provenance mismatch')
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(payload['scheduler'])
        scaler.load_state_dict(payload['scaler'])
        loader.load_state_dict(payload['loader'])
        step, tokens_seen = payload['step'], payload['tokens_seen']
        loader.validate_progress(step, cfg.grad_accum, tokens_seen)
        tokens_processed, tokens_scored = payload['tokens_processed'], payload['tokens_scored']
        if (type(tokens_processed) is not int or type(tokens_scored) is not int
                or not 0 < tokens_scored <= tokens_processed <= tokens_seen):
            raise ValueError('checkpoint processed/scored token counts are inconsistent')
        restore_rng(payload)
        log(f'resumed {resume} at step {step} tokens {tokens_seen}')
    if step >= cfg.steps:
        raise ValueError(f'checkpoint already at step {step} >= steps {cfg.steps}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(
        json.dumps({**asdict(cfg), 'config_hash': config_hash,
                    'provenance': provenance, 'library_versions': library_versions(),
                    'model_params': params, 'metric': 'masked_ce', 'backend': backend,
                    'git_commit': git_commit(),
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                   indent=2, sort_keys=True) + '\n', encoding='utf-8')
    checkpoints, losses, val_losses = [], [], []
    for _ in range(step, cfg.steps):
        micro = []
        masked_total, token_total, processed_total = 0, 0, 0
        for _ in range(cfg.grad_accum):
            targets = next(loader).to(device)
            token_total += targets.numel()
            inputs, loss_mask = corrupt_batch(cfg, targets)
            targets = targets[:, :inputs.shape[1]]
            processed_total += targets.numel()
            masked = int(loss_mask.sum())
            if not masked:
                raise ValueError('empty loss mask in global batch')
            micro.append((inputs, targets, loss_mask, masked))
            masked_total += masked
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for inputs, targets, loss_mask, masked in micro:
            if cfg.amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss = masked_cross_entropy(model(inputs), targets, loss_mask)
            else:
                loss = masked_cross_entropy(model(inputs), targets, loss_mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'non-finite loss at step {step + 1}')
            weighted = loss.float() * (masked / masked_total)
            scaler.scale(weighted).backward()
            loss_sum += float(loss.detach()) * masked
        lr_used = optimizer.param_groups[0]['lr']
        grad_norm, skipped = optimizer_step(model, optimizer, scheduler, scaler, cfg.grad_clip)
        loss = loss_sum / masked_total
        if not math.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step + 1}')
        step += 1
        tokens_seen += token_total
        tokens_processed += processed_total
        tokens_scored += masked_total
        record = {'step': step, 'masked_ce': loss, 'lr': lr_used,
                  'tokens_seen': tokens_seen, 'tokens_processed': tokens_processed,
                  'tokens_scored': tokens_scored, 'grad_norm': grad_norm if not skipped else None,
                  'optimizer_steps': scheduler.last_epoch,
                  'optimizer_step_skipped': skipped}
        if cfg.eval_every and step % cfg.eval_every == 0:
            record['val_masked_ce'] = evaluate(cfg, model, vocab, seq_len,
                                          batches=None if cfg.eval_full else cfg.eval_batches,
                                          device=device)
            val_losses.append(record['val_masked_ce'])
        log(json.dumps(record))
        losses.append(record['masked_ce'])
        if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            path = out / f'checkpoint-{step:06d}.pt'
            save_checkpoint(path, checkpoint_payload(
                model, optimizer, scheduler, scaler, loader, step, tokens_seen,
                config_hash, provenance, tokens_processed, tokens_scored))
            (out / 'checkpoint-last.txt').write_text(path.name, encoding='utf-8')
            checkpoints.append(path.name)
    if not checkpoints or checkpoints[-1] != f'checkpoint-{step:06d}.pt':
        path = out / f'checkpoint-{step:06d}.pt'
        save_checkpoint(path, checkpoint_payload(
            model, optimizer, scheduler, scaler, loader, step, tokens_seen,
            config_hash, provenance, tokens_processed, tokens_scored))
        checkpoints.append(path.name)
        (out / 'checkpoint-last.txt').write_text(path.name, encoding='utf-8')
    return {'config_hash': config_hash, 'params': params, 'metric': 'masked_ce',
            'tokens_processed': tokens_processed, 'tokens_scored': tokens_scored,
            'step': step, 'tokens_seen': tokens_seen, 'losses': losses,
            'val_masked_ce': val_losses, 'checkpoints': checkpoints}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--resume')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--checkpoint-every', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--eval-full', action='store_true')
    p.add_argument('--mask-min', type=float, default=0.1)
    p.add_argument('--mask-max', type=float, default=0.9)
    p.add_argument('--mask-mode', choices=('classical', 'blockwise'), default=None)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--profile', choices=('smoke', 'd1'), default='smoke')
    p.add_argument('--d1-config', type=Path, help='JSON D1Config; defaults to the frozen 16-layer model')
    p.add_argument('--gdn-backend', choices=('torch', 'fla'), default='torch')
    p.add_argument('--p1', action='store_true')
    p.add_argument('--verify-hashes', action='store_true')
    a = p.parse_args(argv)
    try:
        d1 = D1Config(**json.loads(a.d1_config.read_text())) if a.d1_config else D1Config()
    except (OSError, ValueError, TypeError) as error:
        p.error(f'invalid D1 config: {error}')
    cfg = TrainConfig(bundle=a.bundle, out=a.out, seed=a.seed, batch_size=a.batch_size,
                      lr=a.lr, steps=a.steps, warmup=a.warmup,
                      checkpoint_every=a.checkpoint_every, eval_every=a.eval_every,
                      eval_batches=a.eval_batches, eval_full=a.eval_full,
                      mask_min=a.mask_min, mask_max=a.mask_max, mask_mode=a.mask_mode or ('blockwise' if a.profile == 'd1' else 'classical'),
                      d_model=a.d_model, n_layers=a.n_layers,
                      n_heads=a.n_heads, grad_clip=a.grad_clip, grad_accum=a.grad_accum,
                      weight_decay=a.weight_decay, beta1=a.beta1, beta2=a.beta2,
                      eps=a.eps, threads=a.threads, device=a.device, amp=a.amp,
                      profile=a.profile, d1=d1, gdn_backend=a.gdn_backend, p1=a.p1, verify_hashes=a.verify_hashes)
    try:
        summary = train(cfg, resume=a.resume)
    except (ValueError, OSError, RuntimeError, FloatingPointError) as error:
        p.exit(2, f'error: {error}\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'losses'}))


if __name__ == '__main__':
    main()
