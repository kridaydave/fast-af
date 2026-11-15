"""One-process-per-GPU D1 training, preserving the trainer's global batch.

Every rank reads the same global micro-batch and uses a dedicated identical
corruption generator. Only model computation is partitioned. This intentionally
trades duplicated small shard reads for one deterministic cursor and no sampler
padding or extra examples on resume.
"""
from contextlib import nullcontext
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


class DistributedTraining:
    def __init__(self, world_size, device):
        self.world_size = world_size
        self.rank = 0
        self.device = torch.device(device)
        self.owns_group = False
        if world_size == 1:
            if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                raise ValueError('torchrun requires --world-size to match WORLD_SIZE')
            return
        if world_size != 2 or int(os.environ.get('WORLD_SIZE', '1')) != world_size:
            raise ValueError('distributed D1 requires torchrun with exactly two ranks')
        if self.device.type != 'cuda' or torch.cuda.device_count() < world_size:
            raise ValueError('two-GPU training requires two visible CUDA devices')
        local = int(os.environ['LOCAL_RANK'])
        if local not in (0, 1):
            raise ValueError('expected a single node with LOCAL_RANK 0 or 1')
        self.rank = int(os.environ['RANK'])
        if self.rank != local:
            raise ValueError('only single-node distributed training is supported')
        self.device = torch.device('cuda', local)
        torch.cuda.set_device(self.device)
        dist.init_process_group('nccl', timeout=timedelta(seconds=180))
        self.owns_group = True

    def wrap(self, model):
        if self.world_size == 1:
            return model
        return DistributedDataParallel(model, device_ids=[self.device.index],
                                       broadcast_buffers=False, gradient_as_bucket_view=True)

    def sync_context(self, model, final_micro):
        return model.no_sync() if self.world_size > 1 and not final_micro else nullcontext()

    def loss(self, model, inputs, targets, mask, block_size):
        """Return a local SUM; DDP averages gradients, so caller multiplies by world size."""
        part = partition_batch(inputs, targets, mask, self.rank, self.world_size)
        inputs, targets, mask = part
        # A whole active block avoids a dynamic nonzero and supports an all-EOS
        # local partition. Zero-scored ranks must still participate in backward.
        logits = model(inputs, logits_last=block_size)
        values = F.cross_entropy(logits.flatten(0, 1), targets[:, -block_size:].flatten(), reduction='none')
        return (values * mask[:, -block_size:].flatten()).sum()

    def finite(self, value):
        bad = (~torch.isfinite(value)).to(torch.int32)
        if self.world_size > 1:
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        return not bool(bad)

    def loss_sum(self, value):
        if self.world_size > 1:
            dist.all_reduce(value)
        return float(value)

    def save_rng(self):
        state = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(self.device)}
        states: list[dict[str, torch.Tensor] | None] = [None] * self.world_size
        dist.all_gather_object(states, state)
        return {'world_size': self.world_size, 'ranks': states}

    def restore_rng(self, payload):
        if payload['world_size'] != self.world_size or len(payload['ranks']) != self.world_size:
            raise ValueError('resume rejected: distributed world size mismatch')
        state = payload['ranks'][self.rank]
        torch.set_rng_state(state['cpu'])
        torch.cuda.set_rng_state(state['cuda'], self.device)

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()
            self.owns_group = False


def partition_batch(inputs, targets, mask, rank, world_size):
    if inputs.shape[0] % world_size or not 0 <= rank < world_size:
        raise ValueError('global micro-batch must divide evenly across ranks')
    size = inputs.shape[0] // world_size
    return tuple(x[rank * size:(rank + 1) * size] for x in (inputs, targets, mask))
