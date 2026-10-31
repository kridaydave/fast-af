"""Model backends for the benchmark harness.

Each backend exposes the same contract:

    class Backend:
        def warmup(self) -> None: ...
        def generate(self, prompt: str, out_len: int) -> Generation: ...

    Generation = {
        "text": str,
        "num_tokens": int,        # generated tokens
        "decode_seconds": float,  # time to emit them (excludes prefill)
    }

The harness wraps generate() in a median-of-N timer and computes TPS =
num_tokens / median(decode_seconds). Backends must not do their own timing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class Generation:
    text: str
    num_tokens: int
    decode_seconds: float


class VllmBackend:
    """Autoregressive models driven by vLLM, greedy decode, batch 1.

    vLLM reports per-request timing in `outputs[i].stats.times`; we take
    decode_end - prompt_end as the decode window (prefill excluded).
    """

    def __init__(self, repo: str, quant: str | None, eagle3: bool, dtype: str,
                 gpu_mem_util: float = 0.9):
        from vllm import LLM
        kwargs: dict[str, Any] = dict(
            dtype=dtype,
            gpu_memory_utilization=gpu_mem_util,
            enforce_eager=True,          # stable timing on T4 (no CUDA graphs)
            max_model_len=2048,
            quantization=None,
        )
        if quant == "int4":
            kwargs["quantization"] = "gptq"     # T4 has no AWQ; GPTQ int4
        if eagle3:
            # EAGLE-3 speculative decoding. vLLM's native support targets newer
            # GPUs; if this raises on T4, fall back to llama.cpp spec-decode and
            # implement a CustomBackend (budget a day of friction, per spec).
            kwargs["speculative_config"] = {
                "method": "eagle",
                "model": f"{repo}/eagle3",
            }
        self.llm = LLM(model=repo, **kwargs)

    def warmup(self) -> None:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=8)
        self.llm.generate(["warmup"], sp)

    def generate(self, prompt: str, out_len: int) -> Generation:
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=0.0, max_tokens=out_len)
        out = self.llm.generate([prompt], sp)[0]
        stats = out.outputs[0].stats
        gen_tokens = stats.num_generated_tokens
        decode_s = float(stats.times.decode_end - stats.times.prompt_end)
        return Generation(text=out.outputs[0].text, num_tokens=gen_tokens,
                          decode_seconds=max(decode_s, 1e-6))


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

    def _load(self):
        # dLLM exposes per-model classes. LLaDA is the primary target here.
        from dllm.models.llada import LLaDA
        model = LLaDA(model_name=self.repo, dtype=torch.float16, device="cuda")
        model.set_sampling_steps(self.steps)
        return model

    def warmup(self) -> None:
        self._model = self._load()
        self._gen(self._model, "the quick brown fox", 8)

    def _gen(self, model, prompt: str, out_len: int) -> Generation:
        import torch
        t0 = time.perf_counter()
        out = model.generate(prompt, max_new_tokens=out_len,
                             num_steps=self.steps)
        decode_s = time.perf_counter() - t0
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        # token count via the model's tokenizer when available
        num_tokens = getattr(model, "_num_tokens", None) or out_len
        return Generation(text=text, num_tokens=num_tokens, decode_seconds=max(decode_s, 1e-6))

    def generate(self, prompt: str, out_len: int) -> Generation:
        return self._gen(self._model, prompt, out_len)


class CustomBackend:
    """Generic block-diffusion decode loop for models without a dLLM class.

    D1's own decode policy is prototyped here: block_size masked slots, N
    denoising passes, confidence-threshold unmasking (0.9 per token), block
    overlap, and a block-level early stop at 0.97. This is the honest place to
    benchmark smoldlm-144m and to rehearse D1's loop before the real model.

    Requires a model that scores masked positions (masked-token logits). The
    model must expose `score_masked(input_ids) -> logits` or be wrapped to do
    so. Verify the smoldlm repo's API and wire `score_fn` accordingly.
    """

    MASK_ID = 0  # smoldlm config: mask_token_id = 0

    def __init__(self, repo: str, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1, max_new_tokens: int = 1024):
        self.repo = repo
        self.block_size = block_size
        self.steps = steps
        self.confidence = confidence
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.max_new_tokens = max_new_tokens

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int) -> Generation:
        import torch
        enc = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        prefix = enc["input_ids"][0]
        t0 = time.perf_counter()

        out_ids = list(prefix.tolist())
        revealed = 0
        while revealed < out_len:
            block = [self.MASK_ID] * self.block_size
            seq = torch.tensor([out_ids + block], device="cuda")
            prev_conf = 0.0
            for _ in range(self.steps):
                logits = self.score_fn(seq)            # (1, L, V)
                probs = torch.softmax(logits[0, -self.block_size:], dim=-1)
                conf, tok = probs.max(dim=-1)          # per-slot (block,)
                conf = conf.float().cpu()
                if prev_conf > 0.97:                    # block-level early stop
                    break
                prev_conf = conf.min().item()
                conf_mask = conf >= self.confidence
                if conf_mask.all():
                    break
                chosen = tok[conf_mask].tolist()
                for i, c in enumerate(conf_mask.tolist()):
                    if c:
                        block[i] = chosen.pop(0)
                seq = torch.tensor([out_ids + block], device="cuda")
            # commit block
            new = block
            out_ids.extend(new)
            revealed += self.block_size
            if self.eos_id in new:
                break

        decode_s = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids[len(prefix):],
                                     skip_special_tokens=True)
        return Generation(text=text, num_tokens=revealed,
                          decode_seconds=max(decode_s, 1e-6))
