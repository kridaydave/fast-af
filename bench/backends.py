"""Model backends for the benchmark harness.

Each backend exposes the same contract:

    class Backend:
        def warmup(self) -> None: ...
        def generate(self, prompt: str, out_len: int) -> Generation: ...

    Generation = {
        "text": str,
        "num_tokens": int,        # generated tokens
        "decode_seconds": float,  # total generation time after tokenization
        "prefill_seconds": float, # isolated prefill diagnostic
        "passes": int,            # denoising forward passes (diffusion only)
        "warm_passes": int,       # cache-population forwards
        "blocks": int,            # committed blocks (diffusion only)
    }

Every backend times the same outer window: all model computation needed to
generate the continuation after host-side tokenization. `prefill_seconds` is
reported separately with `prefill_kind`: AR uses a standalone probe, cached
diffusion records the actual sub-window, and uncached diffusion cannot isolate
prefill because it recomputes the full prefix on every denoising pass.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class Generation:
    text: str
    num_tokens: int
    decode_seconds: float
    prefill_seconds: float = 0.0
    prefill_kind: str = "not_measured"
    # decode accounting (diffusion loop; 0 where not tracked). TPS alone
    # hides how many forward passes bought the tokens -- the spec requires
    # passes and tokens_per_pass in every results row.
    passes: int = 0
    warm_passes: int = 0
    blocks: int = 0
    stop_reason: str = "unknown"
    token_ids: list[int] | None = None
    fill_counts: list[int] | None = None
    fallback_commits: int = 0


class TransformersBackend:
    """Autoregressive models via HuggingFace generate, greedy (do_sample=False),
    batch 1. Robust on free T4 without vLLM's version gymnastics.

    Timing includes the whole model.generate() call, including prefill. A
    separate prefill probe is measured and reported, but TPS always uses the
    total generation window so it remains comparable with other backends.
    """

    def __init__(self, repo: str, quant: str | None, dtype: str,
                 max_model_len: int = 2048):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        kw: dict[str, Any] = dict(torch_dtype=torch_dtype,
                                  device_map=self.device,
                                  trust_remote_code=True)
        if quant == "int4":
            if self.device != "cuda":
                raise ValueError("int4 quantization requires CUDA; run B1/B2 on T4")
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        self.tokenizer = AutoTokenizer.from_pretrained(repo,
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_model_len = max_model_len

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int,
                 force_length: bool = True,
                 add_special_tokens: bool = True) -> Generation:
        import torch
        if out_len < 1:
            raise ValueError(f"out_len must be >= 1, got {out_len}")
        tokenize_kw = {"return_tensors": "pt"}
        if not add_special_tokens:
            tokenize_kw["add_special_tokens"] = False
        enc = self.tokenizer(prompt, **tokenize_kw).to(self.device)
        plen = enc["input_ids"].shape[1]
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_t0 = time.perf_counter()
        with torch.no_grad():
            self.model(**enc, use_cache=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - prefill_t0
        generate_kw = {
            "max_new_tokens": out_len,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if force_length:
            generate_kw["min_new_tokens"] = out_len
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, **generate_kw)
        if self.device == "cuda":
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
        toks = int(out.shape[1] - plen)
        token_ids = out[0, plen:].tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        stop_reason = "max_new_tokens" if toks >= out_len else "eos"
        return Generation(text=text, num_tokens=toks,
                          decode_seconds=max(decode_s, 1e-6),
                          prefill_seconds=max(prefill_s, 1e-6),
                          prefill_kind="standalone_probe",
                          stop_reason=stop_reason, token_ids=token_ids)

    # ---- evaluation-only path ------------------------------------------------
    # Speed generation above is deliberately untouched: eval must not change
    # the timed decode window, forced output length, or tokenizer call.

    def build_eval_prompt(self, prompt: str, mode: str) -> str:
        """Build the exact text fed to the model for an eval run.

        ``completion`` returns the raw HumanEval prompt. ``chat`` applies the
        tokenizer's chat template to a fixed instruction; the instruction is
        model-agnostic and lives in ``bench.humaneval`` so it can be hashed
        and recorded. No model-id or repo-name guessing happens here.
        """
        from .humaneval import EVAL_MODES, chat_messages

        if mode == "completion":
            return prompt
        if mode == "chat":
            if not getattr(self.tokenizer, "chat_template", None):
                raise ValueError(
                    "chat evaluation requires a tokenizer with a chat "
                    "template; set eval_prompt: completion instead")
            return self.tokenizer.apply_chat_template(
                chat_messages(prompt), tokenize=False,
                add_generation_prompt=True)
        raise ValueError(
            f"eval prompt mode must be one of {EVAL_MODES}, got {mode!r}")

    def generate_eval(self, prompt: str, mode: str,
                      max_new_tokens: int = 256) -> Generation:
        """Generate an eval completion without forcing a fixed token count."""
        kwargs = {"force_length": False}
        if mode == "chat":
            kwargs["add_special_tokens"] = False
        return self.generate(self.build_eval_prompt(prompt, mode),
                             max_new_tokens, **kwargs)

    def evaluation_protocol(self, mode: str) -> dict:
        """Validate formatting and retain the effective tokenizer template."""
        import hashlib
        import json
        from .humaneval import build_protocol

        protocol = build_protocol(mode)
        if mode == "chat":
            self.build_eval_prompt("", mode)
            get_template = getattr(self.tokenizer, "get_chat_template", None)
            template = (get_template() if callable(get_template)
                        else self.tokenizer.chat_template)
            serialized = (template if isinstance(template, str)
                          else json.dumps(template, sort_keys=True))
            protocol["tokenizer_chat_template"] = template
            protocol["tokenizer_chat_template_sha256"] = hashlib.sha256(
                serialized.encode("utf-8")).hexdigest()
        protocol["generation"] = {"max_new_tokens": 256, "do_sample": False,
                                   "force_length": False,
                                   "add_special_tokens": mode != "chat"}
        protocol["protocol_sha256"] = hashlib.sha256(json.dumps(
            protocol, sort_keys=True).encode("utf-8")).hexdigest()
        return protocol


class DllmBackend:
    """Masked/block diffusion via the dLLM framework (LLaDA, Dream, ...).

    dLLM provides a tested `generate` for these models, so we call it natively
    and time only the decode (prompt is re-encoded each call, as the spec's
    single-stream contract requires). The decode-policy knobs (steps, mode)
    map to dLLM's generate args; verify against the installed dLLM version.
    """

    def __init__(self, repo: str, mode: str, steps: int, dtype: str):
        from dllm.utils import set_seed
        self.repo = repo
        self.mode = mode
        self.steps = steps
        self.set_seed = set_seed
        if mode != "default":
            raise NotImplementedError(
                f"dLLM mode {mode!r} is not wired; refusing to label a native "
                "decode as Fast-dLLM")

    def _load(self):
        # dLLM exposes per-model classes. LLaDA is the primary target here.
        from dllm.models.llada import LLaDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LLaDA(model_name=self.repo, dtype=torch.float16, device=device)
        model.set_sampling_steps(self.steps)
        return model

    def warmup(self) -> None:
        self._model = self._load()
        self._gen(self._model, "the quick brown fox", 8)

    def _gen(self, model, prompt: str, out_len: int) -> Generation:
        import torch
        if out_len < 1:
            raise ValueError(f"out_len must be >= 1, got {out_len}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=out_len,
                             num_steps=self.steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        num_tokens = getattr(model, "_num_tokens", None)
        if num_tokens is None:
            tokenizer = getattr(model, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(
                    "dLLM backend did not expose generated token count or tokenizer")
            encoded = tokenizer.encode(text)
            num_tokens = len(encoded if isinstance(encoded, list) else encoded.ids)
        stop_reason = "max_new_tokens" if num_tokens >= out_len else "eos_or_backend"
        return Generation(text=text, num_tokens=num_tokens,
                          decode_seconds=max(decode_s, 1e-6),
                          stop_reason=stop_reason)

    def generate(self, prompt: str, out_len: int) -> Generation:
        return self._gen(self._model, prompt, out_len)


class CustomBackend:
    """Generic block-diffusion decode loop for models without a dLLM class.

    The benchmark-week prototype of D1's decode policy: fill `block_size`
    slots with mask tokens, run up to `steps` denoising passes over
    prefix+block, and commit open slots each pass. Two commit schedules:
    `confidence` (default) unmask every slot whose best prediction clears
    `confidence`; `topk` fills a fixed k0 = block_size // steps slots per
    pass by confidence rank, the stock modeling_smoldlm schedule (the
    0.9 threshold is calibrated to greedy max-probs, so temperature > 0
    needs topk -- see research/quality-fixplan-ernie.md defect D3). The
    last pass always fills whatever is still open, so a committed block
    never contains mask tokens. EOS truncates the block the way an AR
    stop would.

    score_fn(seq) -> logits (1, L, V) must pin mask/pad logits to -inf so
    argmax can never pick a mask slot; load_smoldlm() wires this. With
    kv_cache=True the three cache hooks replace score_fn: cache_reset()
    clears the model cache, cache_warm(ids, pos) appends committed tokens
    to it, score_block(block, pos) scores one block against the cached
    prefix. An optional fourth hook, cache_warm_all(ids), replaces the
    chunked prefix warm with a single equivalent forward (the block-causal
    mask path; see load_smoldlm). The other D1 knobs from the spec (0.97
    block early stop, block overlap) belong to the real model's loop, not
    this prototype.

    tokenizer is a `tokenizers.Tokenizer` (encode().ids / decode()).
    """

    MASK_ID = 0  # smoldlm config: mask_token_id = 0
    COMMITS = ("confidence", "topk")

    def __init__(self, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1, temperature: float = 0.0,
                 commit: str = "confidence", kv_cache: bool = False,
                 cache_reset: Callable[[], None] | None = None,
                 cache_warm: Callable[[list[int], int], None] | None = None,
                 score_block: Callable[..., Any] | None = None,
                 cache_warm_all: Callable[[list[int]], None] | None = None):
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if commit not in self.COMMITS:
            raise ValueError(f"commit must be one of {self.COMMITS}, got {commit!r}")
        if kv_cache and not (cache_reset and cache_warm and score_block):
            raise ValueError(
                "kv_cache=True needs cache_reset, cache_warm and score_block "
                "hooks (load_smoldlm wires them)")
        if cache_warm_all is not None and not kv_cache:
            raise ValueError("cache_warm_all needs kv_cache=True")
        if temperature > 0 and commit == "confidence":
            warnings.warn(
                "temperature > 0 with commit='confidence' commonly collapses "
                "to fallback commits; use commit='topk' for the stock schedule",
                RuntimeWarning, stacklevel=2)
        self.block_size = block_size
        self.steps = steps
        self.confidence = confidence
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        # 0 = greedy argmax (the 2026-11-09 baseline behavior); >0 applies
        # the stock modeling_smoldlm gumbel noise. NOTE: the 0.9 confidence
        # threshold is calibrated to greedy max-probs, so temperature with
        # commit="confidence" is expected to stall -- use commit="topk"
        # (the stock schedule) above temperature 0. See research/
        # quality-fixplan-ernie.md defect D3.
        self.temperature = temperature
        self.commit = commit
        self.kv_cache = kv_cache
        self.cache_reset = cache_reset
        self.cache_warm = cache_warm
        self.score_block = score_block
        self.cache_warm_all = cache_warm_all
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _gumbel(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        # mirrors modeling_smoldlm._add_gumbel_noise exactly, including the
        # exp/log-power form, so temperature>0 here behaves like the stock
        # generate() rather than a lookalike
        if temperature == 0:
            return logits
        dtype = torch.float32 if logits.device.type == "mps" else torch.float64
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(noise.clamp(min=1e-20))) ** temperature
        return logits.exp() / gumbel_noise

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def _score(self, seq: torch.Tensor | None, block: list[int],
               pos_offset: int | None) -> torch.Tensor:
        """One forward pass. With kv_cache the prefix lives in the model's
        KV cache, so only the block is fed, at its absolute position; seq
        (the full-sequence tensor) is unused there and may be None."""
        if self.kv_cache and pos_offset is not None:
            block_t = torch.tensor([block], device=self.device)
            return self.score_block(block_t, pos_offset=pos_offset)
        return self.score_fn(seq)

    def _warm(self, ids: list[int]) -> int:
        """Feed committed tokens into the KV cache. With cache_warm_all the
        whole prefix goes through one block-causal forward (tests prove the
        cache equals the chunked schedule's); otherwise block_size chunks,
        the stock modeling_smoldlm warm schedule."""
        if not self.kv_cache or not ids:
            return 0
        if self.cache_warm_all is not None:
            self.cache_warm_all(list(ids))
            return 1
        calls = 0
        for start in range(0, len(ids), self.block_size):
            self.cache_warm(ids[start:start + self.block_size], start)
            calls += 1
        return calls

    def generate(self, prompt: str, out_len: int) -> Generation:
        if out_len < 1:
            raise ValueError(f"out_len must be >= 1, got {out_len}")
        prefix = self.tokenizer.encode(prompt).ids
        out_ids = list(prefix)
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_s = 0.0
        warm_passes = 0
        if self.kv_cache:
            self.cache_reset()
            prefill_t0 = time.perf_counter()
            warm_passes += self._warm(out_ids)
            if self.device == "cuda":
                torch.cuda.synchronize()
            prefill_s = time.perf_counter() - prefill_t0

        emitted = 0
        passes = 0
        blocks = 0
        fill_counts: list[int] = []
        fallback_commits = 0
        stop_reason = "max_new_tokens"
        while emitted < out_len:
            block = [self.MASK_ID] * self.block_size
            # stock schedule: commits-per-pass fixed from the block's
            # initial mask count (modeling_smoldlm.generate)
            k0 = max(1, self.block_size // self.steps)
            # the kv path scores only the block against the cache, so the
            # full-sequence tensor would be dead weight (an H2D copy of the
            # whole prefix+block every pass)
            seq = None if self.kv_cache else torch.tensor(
                [out_ids + block], device=self.device)
            for step in range(self.steps):
                open_slots = [i for i, t in enumerate(block)
                              if t == self.MASK_ID]
                if not open_slots:
                    break
                logits = self._score(seq, block, len(out_ids))   # (1, L, V)
                # fp32 softmax: fp16 across a 49k vocab loses the tail
                # (stock generate() scores in float too)
                probs = torch.softmax(
                    logits[0, -self.block_size:].float(), dim=-1)
                if self.temperature > 0:
                    picked = self._gumbel(
                        logits[0, -self.block_size:],
                        self.temperature).argmax(dim=-1)
                    conf = probs.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
                    tok = picked
                else:
                    conf, tok = probs.max(dim=-1)      # per-slot (block,)
                # one device->host copy per pass, not one per filled slot:
                # int(tok[i]) on a CUDA tensor syncs the stream each time
                conf = conf.float().cpu()
                tok_ids = tok.cpu().tolist()
                passes += 1
                last = step == self.steps - 1
                if self.commit == "topk":
                    # stock schedule: fill k0 slots per pass by confidence,
                    # whatever the threshold says; last pass fills the rest
                    k = len(open_slots) if last else min(k0, len(open_slots))
                    order = sorted(open_slots, key=lambda i: float(conf[i]),
                                   reverse=True)
                    fill = order[:max(1, k)]
                else:
                    fill = (open_slots if last
                            else [i for i in open_slots
                                  if conf[i] >= self.confidence])
                    if not fill:
                        # threshold cleared nothing: commit the single most
                        # confident slot so every pass makes progress instead
                        # of re-scoring an identical block
                        fill = [max(open_slots, key=lambda i: float(conf[i]))]
                        fallback_commits += 1
                fill_counts.append(len(fill))
                for i in fill:
                    if tok_ids[i] == self.MASK_ID:
                        raise RuntimeError(
                            "score_fn let the mask id win argmax at an open "
                            "slot; pin mask/pad logits to -inf")
                    block[i] = tok_ids[i]
                if not self.kv_cache:
                    seq = torch.tensor([out_ids + block], device=self.device)
            blocks += 1
            if self.eos_id in block:
                block = block[:block.index(self.eos_id)]
                out_ids.extend(block)
                emitted += len(block)
                stop_reason = "eos"
                break
            if self.kv_cache:
                # committed block becomes part of the cached prefix; the
                # model reads the cache during denoising but only updates it
                # here (set_cache_mode), same as stock generate()
                self.cache_warm(block, len(out_ids))
                warm_passes += 1
            out_ids.extend(block)
            emitted += len(block)

        if self.device == "cuda":
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids[len(prefix):],
                                     skip_special_tokens=True)
        return Generation(text=text, num_tokens=emitted,
                          decode_seconds=max(decode_s, 1e-6),
                          prefill_seconds=prefill_s, passes=passes,
                          prefill_kind=("timed_subwindow" if self.kv_cache
                                        else "not_separable"),
                          warm_passes=warm_passes, blocks=blocks,
                          stop_reason=stop_reason,
                          token_ids=out_ids[len(prefix):],
                          fill_counts=fill_counts,
                          fallback_commits=fallback_commits)


def load_smoldlm(repo: str, block_size: int, steps: int, confidence: float,
                 dtype: str, temperature: float = 0.0,
                 commit: str = "confidence",
                 kv_cache: bool = False,
                 warm_batch: bool = False,
                 device: str | None = None) -> CustomBackend:
    """Wire CustomBackend to HoangHa/smoldlm-144m.

    The repo ships `modeling_smoldlm.py` as a standalone module (not a
    transformers AutoModel), so fetch it from the repo and import it
    directly. score_fn mirrors the repo's own generate(): forward pass with
    mask/pad logits pinned to -inf.

    kv_cache=True routes decode through the model's own KV cache, the way
    modeling_smoldlm.generate() does: warm committed tokens with
    set_cache_mode(True) (write), score denoising passes with it off (read
    only). Expected to cut per-pass cost from O(prefix+block) to O(block);
    wire only this one flag to measure it against the cached-off baseline.

    warm_batch=True (opt-in) warms the whole prompt in ONE forward instead
    of block_size chunks. The model's attention is is_causal=False, so a
    plain full-prefix warm would leak future chunks into earlier K/V and is
    NOT equivalent. The equivalent form is a block-causal mask: position p
    may attend keys j <= end of p's own block, exactly the position set a
    chunked warm gives it (cache 0..32c-1 plus its own chunk). The stock
    forward takes no mask, so the attention forwards are replaced with a
    mask-capable verbatim copy of the pinned revision's forward. fp64 cache
    equality and fp32 token equality against the chunked schedule are
    pinned by tests/test_smoldlm_cache.py.
    """
    import hashlib
    import importlib.util
    import types
    from pathlib import Path

    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    if warm_batch and not kv_cache:
        raise ValueError("warm_batch warms the KV cache; needs kv_cache=True")
    if device not in (None, "cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    mod_path = hf_hub_download(repo, "modeling_smoldlm.py")
    source_hash = hashlib.sha256(Path(mod_path).read_bytes()).hexdigest()
    if warm_batch and source_hash != "a366731ca477d1edb6217838c4f32d5468aaf85a6c4a4b89215abf91192f7dec":
        raise ValueError("warm_batch requires the validated smoldlm source from revision 64f1177; upstream code changed")
    spec = importlib.util.spec_from_file_location("modeling_smoldlm", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SmolDLM.from_pretrained(repo)
    if dtype == "float16":
        model = model.half()  # match the AR tier's fp16 full-precision run
    # CustomBackend builds its tensors on this same device; without the
    # move a CUDA box (Colab T4) would feed CPU weights a CUDA tensor
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    def score_fn(seq):
        with torch.no_grad():
            logits = model(seq)
        logits[:, :, model.mask_token_id] = -float("inf")
        logits[:, :, model.pad_token_id] = -float("inf")
        return logits

    cache_reset: Callable[[], None] | None = None
    cache_warm: Callable[[list[int], int], None] | None = None
    score_block: Callable[..., Any] | None = None
    cache_warm_all: Callable[[list[int]], None] | None = None
    if kv_cache:
        def cache_reset() -> None:
            model.reset_kv_cache()
            model.set_cache_mode(False)

        def cache_warm(ids: list[int], pos_offset: int) -> None:
            if not ids:
                return
            chunk = torch.tensor([ids], device=dev)
            model.set_cache_mode(True)   # writes K/V for this chunk only
            try:
                # no_grad like score_fn and the stock generate(): a warm
                # forward only stores K/V, an autograd graph through 30
                # layers would be built and immediately discarded
                with torch.no_grad():
                    model(chunk, pos_offset=pos_offset)
            finally:
                model.set_cache_mode(False)

        def score_block(block: torch.Tensor, pos_offset: int) -> torch.Tensor:
            # denoising over the cached prefix: cache_mode off reads the
            # cache without writing; pin mask/pad like score_fn so argmax
            # can never commit a mask slot
            model.set_cache_mode(False)
            with torch.no_grad():
                logits = model(block, pos_offset=pos_offset)
            logits[:, :, model.mask_token_id] = -float("inf")
            logits[:, :, model.pad_token_id] = -float("inf")
            return logits

    if kv_cache and warm_batch:
        def attn_forward_with_warm_mask(self, x, cos, sin):
            # verbatim copy of modeling_smoldlm.MultiHeadAttention.forward
            # (repo revision 64f1177) with ONE changed line, marked below:
            # SDPA reads an optional block-causal _warm_mask. Kept as a copy
            # because the stock forward takes no mask argument and the
            # block-causal equivalence is what makes batched warm correct.
            B, T, _ = x.size()
            x_pre = x
            q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
            q, k = mod.apply_rotary_emb(q, k, cos, sin)
            q, k = self.q_norm(q), self.k_norm(k)
            q = q.transpose(1, 2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            k_current, v_current = k, v
            if self.kv_cache is not None:
                cached_k, cached_v = self.kv_cache
                k = torch.cat([cached_k, k], dim=2)
                v = torch.cat([cached_v, v], dim=2)
            if self.n_kv_head != self.n_head:
                repeats = self.n_head // self.n_kv_head
                k = k.repeat_interleave(repeats, dim=1)
                v = v.repeat_interleave(repeats, dim=1)
            # the one changed line: optional mask, None outside warm = stock
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False,
                attn_mask=getattr(self, "_warm_mask", None))
            if self.cache_mode:
                self.update_cache(k_current, v_current)
            gate = self.w_gate(x_pre).view(
                B, T, self.n_head, self.head_dim).transpose(1, 2)
            y = y * torch.sigmoid(gate)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)
            return self.c_proj(y)

        for blk in model.blocks:
            blk.attn.forward = types.MethodType(attn_forward_with_warm_mask,
                                                blk.attn)

        def cache_warm_all(ids: list[int]) -> None:
            if not ids:
                return
            ids_t = torch.tensor([ids], device=dev)
            n = ids_t.shape[1]
            # block-causal: position p attends keys j <= end of p's own
            # block (the last partial block ends at n-1 on its own). True
            # = participates, per SDPA's bool-mask semantics.
            block_end = (torch.arange(n, device=dev) // block_size + 1) \
                * block_size - 1
            mask = (torch.arange(n, device=dev)[None, :]
                    <= block_end[:, None])[None, None]
            for blk in model.blocks:
                blk.attn._warm_mask = mask
            model.set_cache_mode(True)
            try:
                with torch.no_grad():
                    model(ids_t, pos_offset=0)
            finally:
                # never leave the mask behind: a score pass reading a stale
                # mask would attend the wrong position set
                for blk in model.blocks:
                    blk.attn._warm_mask = None
                model.set_cache_mode(False)

    backend = CustomBackend(
        block_size=block_size, steps=steps, confidence=confidence,
        score_fn=score_fn,
        tokenizer=Tokenizer.from_file(model._tokenizer_path),
        eos_id=model.eos_token_id,
        temperature=temperature, commit=commit, kv_cache=kv_cache,
        cache_reset=cache_reset, cache_warm=cache_warm,
        score_block=score_block, cache_warm_all=cache_warm_all)
    # for tests and the profiling script; the loop itself never touches it
    backend.model = model
    backend.device = dev
    backend.model_source_sha256 = source_hash
    return backend
