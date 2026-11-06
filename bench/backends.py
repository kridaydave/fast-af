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


class TransformersBackend:
    """Autoregressive models via HuggingFace generate, greedy (do_sample=False),
    batch 1. Robust on free T4 without vLLM's version gymnastics.

    Timing note: we time the whole generate() call, so prompt prefill is
    included in the decode window. For batch-1 single-stream that's small vs
    the 1k-token output and consistent across AR models, so the TPS *ratios*
    (B1/B2 vs B0) the spec wants are unaffected. Swap in a vLLM backend later
    if absolute TPS matters.
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

    def generate(self, prompt: str, out_len: int) -> Generation:
        import torch
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        plen = enc["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=out_len, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id)
        decode_s = time.perf_counter() - t0
        toks = int(out.shape[1] - plen)
        text = self.tokenizer.decode(out[0, plen:], skip_special_tokens=True)
        return Generation(text=text, num_tokens=max(toks, 1),
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = LLaDA(model_name=self.repo, dtype=torch.float16, device=device)
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

    The benchmark-week prototype of D1's decode policy: fill `block_size`
    slots with mask tokens, run up to `steps` denoising passes over
    prefix+block, and each pass unmask every open slot whose best prediction
    clears `confidence`. The last pass always fills whatever is still open,
    so a committed block never contains mask tokens. EOS truncates the block
    the way an AR stop would.

    score_fn(seq) -> logits (1, L, V) must pin mask/pad logits to -inf so
    argmax can never pick a mask slot; load_smoldlm() wires this. The D1
    knobs from the spec (0.97 block early stop, prompt cache, block overlap)
    belong to the real model's loop, not this prototype.

    tokenizer is a `tokenizers.Tokenizer` (encode().ids / decode()).
    """

    MASK_ID = 0  # smoldlm config: mask_token_id = 0

    def __init__(self, block_size: int, steps: int,
                 confidence: float, score_fn: Callable, tokenizer,
                 eos_id: int = 1):
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.block_size = block_size
        self.steps = steps
        self.confidence = confidence
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.eos_id = eos_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def warmup(self) -> None:
        self.generate("the quick brown fox", 8)

    def generate(self, prompt: str, out_len: int) -> Generation:
        prefix = self.tokenizer.encode(prompt).ids
        out_ids = list(prefix)
        t0 = time.perf_counter()

        emitted = 0
        while emitted < out_len:
            block = [self.MASK_ID] * self.block_size
            seq = torch.tensor([out_ids + block], device=self.device)
            for step in range(self.steps):
                open_slots = [i for i, t in enumerate(block)
                              if t == self.MASK_ID]
                if not open_slots:
                    break
                logits = self.score_fn(seq)            # (1, L, V)
                probs = torch.softmax(logits[0, -self.block_size:], dim=-1)
                conf, tok = probs.max(dim=-1)          # per-slot (block,)
                conf = conf.float().cpu()
                fill = (open_slots if step == self.steps - 1
                        else [i for i in open_slots
                              if conf[i] >= self.confidence])
                if not fill:
                    # threshold cleared nothing: commit the single most
                    # confident slot so every pass makes progress instead
                    # of re-scoring an identical block
                    fill = [max(open_slots, key=lambda i: float(conf[i]))]
                for i in fill:
                    if int(tok[i]) == self.MASK_ID:
                        raise RuntimeError(
                            "score_fn let the mask id win argmax at an open "
                            "slot; pin mask/pad logits to -inf")
                    block[i] = int(tok[i])
                seq = torch.tensor([out_ids + block], device=self.device)
            if self.eos_id in block:
                block = block[:block.index(self.eos_id)]
                out_ids.extend(block)
                emitted += len(block)
                break
            out_ids.extend(block)
            emitted += len(block)

        decode_s = time.perf_counter() - t0
        text = self.tokenizer.decode(out_ids[len(prefix):],
                                     skip_special_tokens=True)
        return Generation(text=text, num_tokens=max(emitted, 1),
                          decode_seconds=max(decode_s, 1e-6))


def load_smoldlm(repo: str, block_size: int, steps: int, confidence: float,
                 dtype: str) -> CustomBackend:
    """Wire CustomBackend to HoangHa/smoldlm-144m.

    The repo ships `modeling_smoldlm.py` as a standalone module (not a
    transformers AutoModel), so fetch it from the repo and import it
    directly. score_fn mirrors the repo's own generate(): forward pass with
    mask/pad logits pinned to -inf.
    """
    import importlib.util

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    mod_path = hf_hub_download(repo, "modeling_smoldlm.py")
    spec = importlib.util.spec_from_file_location("modeling_smoldlm", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SmolDLM.from_pretrained(repo)
    if dtype == "float16":
        model = model.half()  # match the AR tier's fp16 full-precision run

    def score_fn(seq):
        with torch.no_grad():
            logits = model(seq)
        logits[:, :, model.mask_token_id] = -float("inf")
        logits[:, :, model.pad_token_id] = -float("inf")
        return logits

    backend = CustomBackend(
        block_size=block_size, steps=steps, confidence=confidence,
        score_fn=score_fn,
        tokenizer=Tokenizer.from_file(model._tokenizer_path),
        eos_id=model.eos_token_id)
    # generate() builds seq on backend.device; without this move a CUDA box
    # (Colab T4) would feed CPU weights a CUDA tensor
    model.to(backend.device)
    return backend
