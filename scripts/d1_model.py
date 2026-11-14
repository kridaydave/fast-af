"""D1 forward-GDN / block-causal GQA denoiser.

The torch recurrence is a differentiable correctness implementation, not a fast
GPU kernel. The optional FLA backend runs the existing chunk_gated_delta_rule
operator. Recurrence: https://arxiv.org/abs/2412.06464 . Cache protocol:
https://arxiv.org/abs/2609.11998 . No claim of reproducing bidirectional Mamba.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import importlib.metadata
import json
import math
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

FLA_COMMIT = '2705ab8ee9a9086bcfa6d079858e4281e01215a1'


@dataclass(frozen=True)
class D1Config:
    d_model: int = 640
    n_layers: int = 16
    attention_every: int = 4
    query_heads: int = 10
    kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden: int = 1728
    gdn_heads: int = 4
    gdn_key_dim: int = 128
    gdn_value_dim: int = 256
    conv_width: int = 4
    block_size: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('D1 dimensions must be positive integers')
        if self.n_layers % self.attention_every:
            raise ValueError('n_layers must be divisible by attention_every')
        if self.query_heads * self.head_dim != self.d_model:
            raise ValueError('query_heads * head_dim must equal d_model')
        if self.query_heads % self.kv_heads:
            raise ValueError('query_heads must be divisible by kv_heads')


@dataclass(frozen=True)
class GDNState:
    recurrent: torch.Tensor
    conv: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class AttentionState:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class D1Cache:
    layers: tuple[GDNState | AttentionState, ...]
    tokens: int


class GenerationResult(TypedDict):
    tokens: torch.Tensor
    tokens_emitted: int
    denoise_passes: int
    commit_passes: int
    prefill_passes: int
    passes: int
    cache: D1Cache | None


def gated_delta_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          decay: torch.Tensor, beta: torch.Tensor,
                          initial_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 delta-rule recurrence with [batch,time,head,feature] inputs.

    q/k are already L2-normalized. State is [batch,head,key,value].
    No input state is mutated, and gradients flow through the initial state.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, decay, beta = (t.float() for t in (q, k, v, decay, beta))
        state = (v.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
                 if initial_state is None else initial_state.float())
        outputs = []
        for t in range(q.shape[1]):
            state = state * decay[:, t].exp()[..., None, None]
            residual = v[:, t] - (state * k[:, t, :, :, None]).sum(-2)
            update = beta[:, t, :, None] * residual
            state = state + k[:, t, :, :, None] * update[..., None, :]
            outputs.append((state * q[:, t, :, :, None]).sum(-2) / math.sqrt(q.shape[-1]))
        return torch.stack(outputs, dim=1), state


def gated_delta_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        decay: torch.Tensor, beta: torch.Tensor,
                        initial_state: torch.Tensor | None = None,
                        chunk_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallelize each chunk with a unit-lower-triangular solve, using PyTorch.

    Same recurrence, normalized q/k and FP32 state as gated_delta_reference.
    The solve computes residual updates together; only chunk boundaries recur.
    Uses ordinary CUDA BLAS operations, not Triton/TF32 or custom CUDA code.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if (q.ndim != 4 or q.shape != k.shape or v.shape[:3] != q.shape[:3]
            or decay.shape != q.shape[:3] or beta.shape != decay.shape or not q.shape[1]):
        raise ValueError('incompatible gated delta tensor shapes')
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, decay, beta = (x.transpose(1, 2).float() for x in (q, k, v, decay, beta))
        batch, heads, length, key_dim = q.shape
        value_dim = v.shape[-1]
        padding = (-length) % chunk_size
        q, k, v = (F.pad(x, (0, 0, 0, padding)) for x in (q, k, v))
        decay, beta = (F.pad(x, (0, padding)) for x in (decay, beta))
        chunks = (length + padding) // chunk_size
        q, k, v = (x.reshape(batch, heads, chunks, chunk_size, x.shape[-1]) for x in (q, k, v))
        cumulative = decay.reshape(batch, heads, chunks, chunk_size).cumsum(-1)
        beta = beta.reshape(batch, heads, chunks, chunk_size, 1)
        lower = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool).tril()
        # Mask before exp: upper-triangle differences can overflow for strong decay.
        differences = cumulative[..., :, None] - cumulative[..., None, :]
        relative_decay = differences.masked_fill(~lower, -torch.inf).exp()
        identity = torch.eye(chunk_size, device=q.device)
        system = identity + ((beta * k) @ k.transpose(-1, -2) * relative_decay).tril(-1)
        rhs = torch.cat((beta * v, beta * cumulative.exp()[..., None] * k), dim=-1)
        transformed = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        base_values, state_weights = transformed.split((value_dim, key_dim), dim=-1)
        state = (v.new_zeros(batch, heads, key_dim, value_dim)
                 if initial_state is None else initial_state.float())
        outputs = []
        for index in range(chunks):
            queries, keys = q[:, :, index] / math.sqrt(key_dim), k[:, :, index]
            log_decay = cumulative[:, :, index]
            updates = base_values[:, :, index] - state_weights[:, :, index] @ state
            attention = queries @ keys.transpose(-1, -2) * relative_decay[:, :, index]
            output = (queries * log_decay.exp()[..., None]) @ state + attention @ updates
            outputs.append(output)
            end_weights = (log_decay[..., -1:] - log_decay).exp()
            state = (state * log_decay[..., -1, None, None].exp()
                     + (keys * end_weights[..., None]).transpose(-1, -2) @ updates)
        output = torch.cat(outputs, dim=2)[:, :, :length].transpose(1, 2).contiguous()
        return output, state


def backend_info(backend: str) -> dict[str, str]:
    if backend not in ('torch', 'chunk', 'fla'):
        raise ValueError('GDN backend must be torch, chunk or fla')
    if backend == 'torch':
        return {'name': 'torch-reference', 'torch': torch.__version__}
    if backend == 'chunk':
        return {'name': 'torch-chunked', 'version': '1', 'chunk_size': '32', 'torch': torch.__version__}
    try:
        distribution = importlib.metadata.distribution('flash-linear-attention')
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        if commit != FLA_COMMIT:
            raise RuntimeError('FLA source revision does not match requirements-d1.txt')
        return {'name': 'fla', 'version': distribution.version, 'commit': commit}
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('FLA is not installed; install requirements-d1.txt on the GPU host') from error


def require_fla_device(device: torch.device | str) -> None:
    """The pinned Triton >=3.3 stack supports SM80+, excluding T4/SM75."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('FLA backend requires CUDA')
    if torch.cuda.get_device_capability(device) < (8, 0):
        raise ValueError('pinned FLA/Triton requires SM80+; T4/SM75 is unsupported')


@lru_cache(maxsize=1)
def fla_operator():
    backend_info('fla')
    try:
        operator = importlib.import_module('fla.ops.gated_delta_rule').chunk_gated_delta_rule
    except ImportError as error:
        raise RuntimeError('FLA backend unavailable; install requirements-d1.txt on the GPU host') from error
    return operator


class CausalConv(nn.Module):
    def __init__(self, channels: int, width: int):
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.empty(channels, 1, width))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        if state is None:
            state = x.new_zeros(x.shape[0], x.shape[1], self.width - 1)
        history = torch.cat((state, x), dim=-1)
        output = F.conv1d(history, self.weight, groups=x.shape[1])
        tail = history[..., -(self.width - 1):] if self.width > 1 else history[..., :0]
        return F.silu(output.transpose(1, 2)), tail


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: D1Config, backend: str):
        super().__init__()
        if backend not in ('torch', 'chunk', 'fla'):
            raise ValueError('GDN backend must be torch, chunk or fla')
        self.cfg, self.backend = cfg, backend
        key, value = cfg.gdn_heads * cfg.gdn_key_dim, cfg.gdn_heads * cfg.gdn_value_dim
        self.q_proj = nn.Linear(cfg.d_model, key, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, key, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, value, bias=False)
        self.a_proj = nn.Linear(cfg.d_model, cfg.gdn_heads, bias=False)
        self.b_proj = nn.Linear(cfg.d_model, cfg.gdn_heads, bias=False)
        self.g_proj = nn.Linear(cfg.d_model, value, bias=False)
        self.o_proj = nn.Linear(value, cfg.d_model, bias=False)
        self.convs = nn.ModuleList([CausalConv(n, cfg.conv_width) for n in (key, key, value)])
        self.output_norm = nn.RMSNorm(cfg.gdn_value_dim, eps=1e-5)
        self.A_log = nn.Parameter(torch.empty(cfg.gdn_heads).uniform_(0, 16).clamp_min(1e-4).log())
        dt = torch.empty(cfg.gdn_heads).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, state=None, offset=0, use_cache=False):
        c = self.cfg
        values, conv_states = [], []
        for index, (projection, conv) in enumerate(zip((self.q_proj, self.k_proj, self.v_proj), self.convs)):
            output, tail = conv(projection(x), None if state is None else state.conv[index])
            values.append(output)
            conv_states.append(tail)
        q, k = [t.reshape(*x.shape[:2], c.gdn_heads, c.gdn_key_dim) for t in values[:2]]
        v = values[2].reshape(*x.shape[:2], c.gdn_heads, c.gdn_value_dim)
        # Keep normalization/decay in FP32, including under fp16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6)
            k = F.normalize(k.float(), dim=-1, eps=1e-6)
            decay = -self.A_log.float().exp() * F.softplus(self.a_proj(x.float()) + self.dt_bias.float())
            beta = self.b_proj(x.float()).sigmoid()
        initial = None if state is None else state.recurrent
        if self.backend == 'fla':
            if x.device.type != 'cuda' or v.dtype != torch.float16:
                raise ValueError('FLA D1 execution requires CUDA fp16 autocast')
            require_fla_device(x.device)
            output, final = fla_operator()(
                q=q.to(v.dtype), k=k.to(v.dtype), v=v, g=decay, beta=beta,
                initial_state=initial, output_final_state=use_cache)
        elif self.backend == 'chunk':
            output, final = gated_delta_chunked(q, k, v, decay, beta, initial)
        else:
            output, final = gated_delta_reference(q, k, v, decay, beta, initial)
        gate = self.g_proj(x).reshape(*x.shape[:2], c.gdn_heads, c.gdn_value_dim)
        output = self.output_norm(output.to(gate.dtype)) * F.silu(gate)
        result = self.o_proj(output.flatten(-2))
        cache = GDNState(final, tuple(conv_states)) if use_cache else None
        return result, cache


class BlockAttention(nn.Module):
    def __init__(self, cfg: D1Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, state=None, offset=0, use_cache=False):
        c = self.cfg
        q = self.q_proj(x).reshape(*x.shape[:2], c.query_heads, c.head_dim).transpose(1, 2)
        k, v = [proj(x).reshape(*x.shape[:2], c.kv_heads, c.head_dim).transpose(1, 2)
                for proj in (self.k_proj, self.v_proj)]
        if state is not None:
            k, v = torch.cat((state.keys, k), dim=2), torch.cat((state.values, v), dim=2)
        query_blocks = (offset + torch.arange(x.shape[1], device=x.device)) // c.block_size
        key_blocks = torch.arange(k.shape[2], device=x.device) // c.block_size
        allowed = key_blocks[None, :] <= query_blocks[:, None]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        result = self.o_proj(output.transpose(1, 2).reshape(*x.shape[:2], c.d_model))
        return result, AttentionState(k, v) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, cfg: D1Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class D1Block(nn.Module):
    def __init__(self, cfg: D1Config, index: int, backend: str):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.mixer = (BlockAttention(cfg) if (index + 1) % cfg.attention_every == 0
                      else GatedDeltaNet(cfg, backend))
        self.norm2 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, state=None, offset=0, use_cache=False):
        mixed, new_state = self.mixer(self.norm1(x), state, offset, use_cache)
        x = x + mixed
        return x + self.ffn(self.norm2(x)), new_state


class D1LM(nn.Module):
    """Unlooped hybrid with NoPE and tied input/output embeddings."""
    def __init__(self, vocab_size: int, seq_len: int, cfg: D1Config | None = None, backend: str = 'torch'):
        super().__init__()
        self.cfg = D1Config() if cfg is None else cfg
        if backend not in ('torch', 'chunk', 'fla'):
            raise ValueError('GDN backend must be torch, chunk or fla')
        if type(vocab_size) is not int or vocab_size < 6:
            raise ValueError('D1 vocab_size must be an integer of at least 6')
        if type(seq_len) is not int or seq_len < self.cfg.block_size or seq_len % self.cfg.block_size:
            raise ValueError('D1 seq_len must be a positive multiple of block_size')
        if backend == 'fla':
            fla_operator()  # Fail at construction, never silently change the backend.
        self.vocab_size, self.seq_len, self.backend = vocab_size, seq_len, backend
        self.tokens = nn.Embedding(vocab_size, self.cfg.d_model)
        blocks = [D1Block(self.cfg, i, backend) for i in range(self.cfg.n_layers)]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(self.cfg.d_model, eps=1e-5)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in blocks:
            for output in (block.mixer.o_proj, block.ffn.down):
                nn.init.normal_(output.weight, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def _run(self, inputs: torch.Tensor, cache: D1Cache | None = None,
             use_cache: bool = False) -> tuple[torch.Tensor, D1Cache | None]:
        offset = 0 if cache is None else cache.tokens
        if inputs.ndim != 2 or not inputs.shape[0] or not 0 < inputs.shape[1] <= self.seq_len - offset:
            raise ValueError('expected nonempty [batch, time] within configured context')
        if cache is not None:
            if offset % self.cfg.block_size or len(cache.layers) != len(self.blocks):
                raise ValueError('cache must end at a block boundary with one state per layer')
        hidden = self.tokens(inputs)
        states: list[GDNState | AttentionState] = []
        for i, block in enumerate(self.blocks):
            hidden, state = block(hidden, None if cache is None else cache.layers[i], offset, use_cache)
            if state is not None:
                states.append(state)
        logits = F.linear(self.norm(hidden), self.tokens.weight)
        return logits, D1Cache(tuple(states), offset + inputs.shape[1]) if use_cache else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._run(inputs)[0]

    def forward_block(self, inputs: torch.Tensor, cache: D1Cache | None = None) -> tuple[torch.Tensor, D1Cache]:
        """Return a candidate cache. The caller commits only clean final tokens."""
        if inputs.ndim != 2 or inputs.shape[1] > self.cfg.block_size:
            raise ValueError('forward_block accepts at most one block')
        logits, candidate = self._run(inputs, cache, use_cache=True)
        assert candidate is not None
        return logits, candidate

    def prefill(self, prefix: torch.Tensor) -> D1Cache | None:
        if prefix.ndim != 2 or prefix.shape[1] % self.cfg.block_size:
            raise ValueError('prefill requires whole clean blocks')
        cache = None
        for start in range(0, prefix.shape[1], self.cfg.block_size):
            _, cache = self.forward_block(prefix[:, start:start + self.cfg.block_size], cache)
        return cache

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, max_new_tokens: int, steps: int = 8) -> GenerationResult:
        """Fixed-length greedy confidence decode, with a separate clean commit pass.

        This is a D1 development decoder, not the frozen benchmark policy.
        Partial prompt blocks remain fixed inside the first active block.
        """
        if (prefix.ndim != 2 or not prefix.shape[0] or type(max_new_tokens) is not int or max_new_tokens < 0
                or type(steps) is not int or steps < 1
                or prefix.shape[1] + max_new_tokens > self.seq_len):
            raise ValueError('invalid generation length or steps')
        if prefix.numel() and (int(prefix.min()) < 1 or int(prefix.max()) >= self.vocab_size
                               or bool(torch.isin(prefix, prefix.new_tensor([2, 3, 4])).any())):
            raise ValueError('generation prefix must contain clean vocabulary tokens')
        was_training = self.training
        self.eval()
        try:
            block_size = self.cfg.block_size
            aligned = prefix.shape[1] // block_size * block_size
            cache = self.prefill(prefix[:, :aligned])
            known = prefix[:, aligned:]
            result = prefix[:, :aligned]
            remaining, denoise, commits = max_new_tokens, 0, 0
            while remaining:
                take = min(block_size - known.shape[1], remaining)
                active = torch.cat((known, prefix.new_zeros(prefix.shape[0], take)), dim=1)
                pending = torch.zeros_like(active, dtype=torch.bool)
                pending[:, known.shape[1]:] = True
                for round_index in range(min(steps, take)):
                    logits, _ = self.forward_block(active, cache)
                    logits = logits.float()
                    logits[..., [0, 2, 3, 4]] = -torch.inf
                    probabilities = logits.softmax(-1)
                    confidence, predictions = probabilities.max(-1)
                    count = math.ceil(int(pending[0].sum()) / (min(steps, take) - round_index))
                    chosen = confidence.masked_fill(~pending, -torch.inf).topk(count, dim=1).indices
                    active.scatter_(1, chosen, predictions.gather(1, chosen))
                    pending.scatter_(1, chosen, False)
                    denoise += 1
                _, cache = self.forward_block(active, cache)
                commits += 1
                result = torch.cat((result, active), dim=1)
                remaining -= take
                known = prefix[:, :0]
            if max_new_tokens == 0:
                result = prefix.clone()
            prefill_passes = aligned // block_size
            if not max_new_tokens and known.shape[1]:
                _, cache = self.forward_block(known, cache)
                prefill_passes += 1
            return {'tokens': result, 'tokens_emitted': max_new_tokens,
                    'denoise_passes': denoise, 'commit_passes': commits,
                    'prefill_passes': prefill_passes,
                    'passes': prefill_passes + denoise + commits, 'cache': cache}
        finally:
            self.train(was_training)
