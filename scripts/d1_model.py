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


class _TensorCoreMM(torch.autograd.Function):
    """FP16 operands with FP32 GEMM output and FP32 first-order gradients.

    torch.bmm(out_dtype=float32) has no native autograd on the measured stack.
    Keep this explicit rather than silently rounding state/results to FP16.
    """
    @staticmethod
    def forward(ctx, a, b):
        a, b = a.half(), b.half()
        ctx.save_for_backward(a, b)
        return torch.bmm(a, b, out_dtype=torch.float32)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        a, b = ctx.saved_tensors
        gradient = gradient.half()
        return (torch.bmm(gradient, b.transpose(-1, -2), out_dtype=torch.float32),
                torch.bmm(a.transpose(-1, -2), gradient, out_dtype=torch.float32))


def tensorcore_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Equal batch dimensions only; callers keep state/solve/decay in FP32."""
    if a.device.type != 'cuda' or b.device != a.device:
        raise ValueError('Tensor Core GDN requires CUDA operands on the same device')
    if a.shape[:-2] != b.shape[:-2] or a.shape[-1] != b.shape[-2]:
        raise ValueError('Tensor Core GDN matmul does not broadcast')
    shape = (*a.shape[:-1], b.shape[-1])
    return _TensorCoreMM.apply(a.reshape(-1, *a.shape[-2:]),
                               b.reshape(-1, *b.shape[-2:])).reshape(shape)


def gated_delta_chunked(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        decay: torch.Tensor, beta: torch.Tensor,
                        initial_state: torch.Tensor | None = None,
                        chunk_size: int = 128, *, tensor_cores: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable chunked delta recurrence with FP32 state and solve."""
    output, final, _ = _gated_delta_chunked(q, k, v, decay, beta, initial_state,
                                           chunk_size, tensor_cores=tensor_cores)
    return output, final


def _gated_delta_chunked(q, k, v, decay, beta, initial_state=None,
                         chunk_size=128, *, tensor_cores=False, collect_states=False, state_stride=None):
    """Parallelize each chunk with a unit-lower-triangular solve, using PyTorch.

    Same recurrence, normalized q/k and FP32 state as gated_delta_reference.
    The solve computes residual updates together; only chunk boundaries recur.
    Uses ordinary CUDA BLAS operations, not Triton/TF32 or custom CUDA code.
    Short inputs use a smaller power-of-two chunk to avoid excess padding.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if (q.ndim != 4 or q.shape != k.shape or v.shape[:3] != q.shape[:3]
            or decay.shape != q.shape[:3] or beta.shape != decay.shape or not q.shape[1]):
        raise ValueError('incompatible gated delta tensor shapes')
    mm = tensorcore_mm if tensor_cores else torch.matmul
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, decay, beta = (x.transpose(1, 2).float() for x in (q, k, v, decay, beta))
        batch, heads, length, key_dim = q.shape
        value_dim = v.shape[-1]
        chunk_size = min(chunk_size, 1 << (length - 1).bit_length())
        stride = chunk_size if state_stride is None else state_stride
        if type(stride) is not int or stride < 1 or chunk_size % stride:
            raise ValueError('state_stride must divide chunk_size')
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
        absolute_decay = cumulative.exp()
        identity = torch.eye(chunk_size, device=q.device)
        system = identity + (mm(beta * k, k.transpose(-1, -2)) * relative_decay).tril(-1)
        rhs = torch.cat((beta * v, beta * absolute_decay[..., None] * k), dim=-1)
        transformed = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        base_values, state_weights = transformed.split((value_dim, key_dim), dim=-1)
        state = (v.new_zeros(batch, heads, key_dim, value_dim)
                 if initial_state is None else initial_state.float())
        # Batch the state-independent work across every chunk before the scan.
        queries = q / math.sqrt(key_dim)
        attention = mm(queries, k.transpose(-1, -2)) * relative_decay
        projected_queries = queries * absolute_decay[..., None]
        end_keys = (k * (cumulative[..., -1:] - cumulative).exp()[..., None]).transpose(-1, -2)
        end_decay = absolute_decay[..., -1, None, None]
        # Recover interior block entry states from the same solved updates.
        # This keeps the clean scan at 128 tokens instead of launching a scan
        # step for every 32-token block. Mask BEFORE exp for strong decay.
        if collect_states and stride < chunk_size:
            ends = torch.arange(stride - 1, chunk_size - 1, stride, device=q.device)
            visible = torch.arange(chunk_size, device=q.device)[None, :] <= ends[:, None]
            relative = cumulative[..., ends, None] - cumulative[..., None, :]
            relative = relative.masked_fill(~visible, -torch.inf).exp()
            interior_keys = (k.unsqueeze(-3) * relative[..., None]).transpose(-1, -2)
            interior_decay = absolute_decay[..., ends, None, None]
            interiors = zip(interior_keys.unbind(2), interior_decay.unbind(2))
        else:
            interiors = iter(())
        outputs, states = [], []
        # Unbind once: indexing the full tensors inside the loop makes backward
        # zero and accumulate full-sized slice gradients for every chunk.
        pieces = zip(*(x.unbind(2) for x in
                       (projected_queries, attention, end_keys, end_decay, base_values, state_weights)))
        for queries, scores, keys, decay_factor, values, weights in pieces:
            if collect_states:
                states.append(state)
            updates = values - mm(weights, state)
            if collect_states and stride < chunk_size:
                middle_keys, middle_decay = next(interiors)
                expanded = updates.unsqueeze(2).expand(*middle_keys.shape[:-2], chunk_size, value_dim)
                middle = state.unsqueeze(2) * middle_decay + mm(middle_keys, expanded)
                states.extend(middle.unbind(2))
            output = mm(queries, state) + mm(scores, updates)
            outputs.append(output)
            state = state * decay_factor + mm(keys, updates)
        output = torch.cat(outputs, dim=2)[:, :, :length].transpose(1, 2).contiguous()
        return output, state, states


def backend_info(backend: str) -> dict[str, str]:
    if backend not in ('torch', 'chunk', 'chunk-tc', 'fla'):
        raise ValueError('GDN backend must be torch, chunk, chunk-tc or fla')
    if backend == 'torch':
        return {'name': 'torch-reference', 'torch': torch.__version__}
    if backend == 'chunk':
        return {'name': 'torch-chunked', 'version': '2', 'chunk_size': '128',
                'short_chunk': 'power-of-two', 'torch': torch.__version__}
    if backend == 'chunk-tc':
        return {'name': 'torch-chunked-tensorcore', 'version': '3', 'chunk_size': '128',
                'short_chunk': 'power-of-two', 'matmul': 'fp16-input-fp32-output-and-gradient',
                'state_solve': 'fp32', 'residual': 'fp32',
                'norm': 'fp32-input-and-weight', 'torch': torch.__version__}
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
        if backend not in ('torch', 'chunk', 'chunk-tc', 'fla'):
            raise ValueError('GDN backend must be torch, chunk, chunk-tc or fla')
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
        values, conv_states = [], []
        for index, (projection, conv) in enumerate(zip((self.q_proj, self.k_proj, self.v_proj), self.convs)):
            output, tail = conv(projection(x), None if state is None else state.conv[index])
            values.append(output)
            conv_states.append(tail)
        q, k, v, decay, beta = self._recurrence_inputs(x, values)
        initial = None if state is None else state.recurrent
        if self.backend == 'fla':
            if x.device.type != 'cuda' or v.dtype != torch.float16:
                raise ValueError('FLA D1 execution requires CUDA fp16 autocast')
            require_fla_device(x.device)
            output, final = fla_operator()(
                q=q.to(v.dtype), k=k.to(v.dtype), v=v, g=decay, beta=beta,
                initial_state=initial, output_final_state=use_cache)
        elif self.backend in ('chunk', 'chunk-tc'):
            if self.backend == 'chunk-tc' and (x.device.type != 'cuda' or v.dtype != torch.float16):
                raise ValueError('chunk-tc requires CUDA fp16 autocast')
            output, final = gated_delta_chunked(q, k, v, decay, beta, initial,
                                                tensor_cores=self.backend == 'chunk-tc')
        else:
            output, final = gated_delta_reference(q, k, v, decay, beta, initial)
        cache = GDNState(final, tuple(conv_states)) if use_cache else None
        return self._output(x, output), cache

    def _recurrence_inputs(self, x, values) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cfg
        q, k = [t.reshape(*x.shape[:2], c.gdn_heads, c.gdn_key_dim) for t in values[:2]]
        v = values[2].reshape(*x.shape[:2], c.gdn_heads, c.gdn_value_dim)
        # Keep normalization/decay in FP32, including under fp16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = F.normalize(q.float(), dim=-1, eps=1e-6)
            k = F.normalize(k.float(), dim=-1, eps=1e-6)
            decay = -self.A_log.float().exp() * F.softplus(self.a_proj(x.float()) + self.dt_bias.float())
            beta = self.b_proj(x.float()).sigmoid()
        return q, k, v, decay, beta

    def _output(self, x, output):
        c = self.cfg
        gate = self.g_proj(x).reshape(*x.shape[:2], c.gdn_heads, c.gdn_value_dim)
        # Same-dtype FP32 norm avoids Torch 2.11's expensive mixed-dtype fallback.
        # Keep FP32 residuals/norms: FP16 residuals failed cached/full parity on T4.
        norm_input = output if self.backend == 'chunk-tc' else output.to(gate.dtype)
        output = self.output_norm(norm_input) * F.silu(gate)
        return self.o_proj(output.flatten(-2))

    def forward_all_blocks(self, clean, noisy):
        """Fork each noisy block from its differentiable CLEAN entry state.

        No noisy state is shared across blocks. Entry convolution histories use
        raw clean projections, exactly as a normal cached prefix would.
        """
        if self.backend == 'fla':
            raise ValueError('all-block supervision supports torch, chunk and chunk-tc backends')
        batch, length, channels = clean.shape
        block = self.cfg.block_size
        count = length // block
        noisy_blocks = noisy.reshape(batch * count, block, channels)
        clean_values, noisy_values = [], []
        for projection, conv in zip((self.q_proj, self.k_proj, self.v_proj), self.convs):
            assert isinstance(conv, CausalConv)
            raw = projection(clean)
            clean_values.append(conv(raw)[0])
            width = conv.width - 1
            history = F.pad(raw.transpose(1, 2), (width, 0))
            if width:
                entries = history.unfold(-1, width, block)[:, :, :count]
                entries = entries.permute(0, 2, 1, 3).reshape(batch * count, raw.shape[-1], width)
            else:
                entries = raw.new_empty(batch * count, raw.shape[-1], 0)
            noisy_values.append(conv(projection(noisy_blocks), entries)[0])
        clean_args = self._recurrence_inputs(clean, clean_values)
        if self.backend == 'torch':
            state = clean.new_zeros(batch, self.cfg.gdn_heads, self.cfg.gdn_key_dim,
                                    self.cfg.gdn_value_dim, dtype=torch.float32)
            outputs, entries = [], []
            for index in range(count):
                entries.append(state)
                q, k, v, decay, beta = [value[:, index * block:(index + 1) * block] for value in clean_args]
                output, state = gated_delta_reference(q, k, v, decay, beta, state)
                outputs.append(output)
            clean_output = torch.cat(outputs, dim=1)
        else:
            if self.backend == 'chunk-tc' and (clean.device.type != 'cuda' or clean_values[2].dtype != torch.float16):
                raise ValueError('chunk-tc requires CUDA fp16 autocast')
            clean_output, _, entries = _gated_delta_chunked(
                *clean_args, chunk_size=max(128, block) if block & (block - 1) == 0 else block,
                tensor_cores=self.backend == 'chunk-tc', collect_states=True, state_stride=block)
        initial = torch.stack(entries[:count], dim=1).flatten(0, 1)
        noisy_args = self._recurrence_inputs(noisy_blocks, noisy_values)
        if self.backend == 'torch':
            noisy_output, _ = gated_delta_reference(*noisy_args, initial)
        else:
            noisy_output, _ = gated_delta_chunked(*noisy_args, initial, chunk_size=block,
                                                  tensor_cores=self.backend == 'chunk-tc')
        return (self._output(clean, clean_output),
                self._output(noisy_blocks, noisy_output).reshape(batch, length, channels))


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

    def forward_all_blocks(self, clean, noisy):
        c = self.cfg
        batch, length, _ = clean.shape
        # Clean queries see <= their own block. Noisy queries see strictly
        # earlier clean blocks and ONLY their own noisy block.
        both = torch.cat((clean, noisy), dim=1)
        q = self.q_proj(both).reshape(batch, 2 * length, c.query_heads, c.head_dim).transpose(1, 2)
        k, v = [proj(both).reshape(batch, 2 * length, c.kv_heads, c.head_dim).transpose(1, 2)
                for proj in (self.k_proj, self.v_proj)]
        blocks = torch.arange(length, device=clean.device) // c.block_size
        earlier = blocks[None, :] < blocks[:, None]
        same = blocks[None, :] == blocks[:, None]
        allowed = torch.cat((torch.cat((earlier | same, torch.zeros_like(same)), dim=1),
                             torch.cat((earlier, same), dim=1)), dim=0)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        result = self.o_proj(output.transpose(1, 2).reshape(batch, 2 * length, c.d_model))
        return result[:, :length], result[:, length:]


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

    def forward_all_blocks(self, clean, noisy):
        clean_mix, noisy_mix = self.mixer.forward_all_blocks(self.norm1(clean), self.norm1(noisy))
        clean, noisy = clean + clean_mix, noisy + noisy_mix
        both = torch.cat((clean, noisy), dim=1)
        both = both + self.ffn(self.norm2(both))
        return both.split(clean.shape[1], dim=1)


class D1LM(nn.Module):
    """Unlooped hybrid with NoPE and tied input/output embeddings."""
    def __init__(self, vocab_size: int, seq_len: int, cfg: D1Config | None = None, backend: str = 'torch'):
        super().__init__()
        self.cfg = D1Config() if cfg is None else cfg
        if backend not in ('torch', 'chunk', 'chunk-tc', 'fla'):
            raise ValueError('GDN backend must be torch, chunk, chunk-tc or fla')
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
             use_cache: bool = False, logits_mask: torch.Tensor | None = None,
             logits_last: int | None = None) -> tuple[torch.Tensor, D1Cache | None]:
        offset = 0 if cache is None else cache.tokens
        if inputs.ndim != 2 or not inputs.shape[0] or not 0 < inputs.shape[1] <= self.seq_len - offset:
            raise ValueError('expected nonempty [batch, time] within configured context')
        if cache is not None:
            if offset % self.cfg.block_size or len(cache.layers) != len(self.blocks):
                raise ValueError('cache must end at a block boundary with one state per layer')
        if logits_mask is not None and (logits_mask.dtype != torch.bool or logits_mask.shape != inputs.shape):
            raise ValueError('logits_mask must be boolean with the input shape')
        if logits_last is not None and (type(logits_last) is not int or not 0 < logits_last <= inputs.shape[1]
                                        or logits_mask is not None):
            raise ValueError('logits_last must select a nonempty suffix, without logits_mask')
        hidden = self.tokens(inputs)
        states: list[GDNState | AttentionState] = []
        for i, block in enumerate(self.blocks):
            hidden, state = block(hidden, None if cache is None else cache.layers[i], offset, use_cache)
            if state is not None:
                states.append(state)
        # Every token still participates in the hybrid. Project only requested
        # loss positions to the vocabulary during training and validation.
        if logits_mask is not None:
            hidden = hidden[logits_mask]
        elif logits_last is not None:
            hidden = hidden[:, -logits_last:]
        logits = F.linear(self.norm(hidden), self.tokens.weight)
        return logits, D1Cache(tuple(states), offset + inputs.shape[1]) if use_cache else None

    def forward(self, inputs: torch.Tensor, *, logits_mask: torch.Tensor | None = None,
                logits_last: int | None = None, clean_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if clean_inputs is not None:
            if (inputs.ndim != 2 or inputs.shape != clean_inputs.shape or not inputs.shape[0]
                    or not 0 < inputs.shape[1] <= self.seq_len or inputs.shape[1] % self.cfg.block_size
                    or logits_last is not None):
                raise ValueError('all-block supervision requires matching whole-block clean/noisy inputs, without logits_last')
            if logits_mask is not None and (logits_mask.shape != inputs.shape or logits_mask.dtype != torch.bool):
                raise ValueError('logits_mask must be boolean with the input shape')
            clean, noisy = self.tokens(clean_inputs), self.tokens(inputs)
            for block in self.blocks:
                assert isinstance(block, D1Block)
                clean, noisy = block.forward_all_blocks(clean, noisy)
            if logits_mask is not None:
                noisy = noisy[logits_mask]
            return F.linear(self.norm(noisy), self.tokens.weight)
        return self._run(inputs, logits_mask=logits_mask, logits_last=logits_last)[0]

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
