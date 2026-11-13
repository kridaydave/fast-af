"""Opt-in per-phase profiler for the smoldlm cached policy.

Times exactly what the benchmark timing window contains (cache-warm
forwards, denoising score passes, host-side bookkeeping) without changing
what generate() computes. This is a diagnostic tool, not a benchmark:
numbers it prints are for finding where the seconds go, never for quoting
as TPS.

Examples:
    python -m scripts.profile_smoldlm_cache --out-len 128
    python -m scripts.profile_smoldlm_cache --microbench
    python -m scripts.profile_smoldlm_cache --device cuda --sync  # T4, instrumented

On CPU, forward passes are synchronous, so per-call wall times are solid.
On CUDA, per-call times measure launch-side overhead unless --sync is
passed, which adds explicit synchronization around each phase. --sync is
forbidden in the benchmark path (it would inflate the measured window);
this script only.
"""
from __future__ import annotations

import argparse
import time

import torch


def _timed(fn, synchronize=None):
    """Wrap a hook so every call records its wall time. `synchronize`, when
    given, brackets each call so the recorded time is device-complete
    (profiler-only; the benchmark path never syncs inside its window)."""
    times: list[float] = []

    def wrapped(*args, **kwargs):
        if synchronize is not None:
            synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if synchronize is not None:
            synchronize()
        times.append(time.perf_counter() - t0)
        return out

    return wrapped, times


def profile(args) -> None:
    from bench.backends import load_smoldlm

    backend = load_smoldlm(
        args.repo, block_size=args.block_size, steps=args.steps,
        confidence=args.confidence, dtype=args.dtype,
        commit=args.commit, kv_cache=True, warm_batch=args.warm_batch,
        device=args.device)
    synchronize = (torch.cuda.synchronize
                   if args.sync and backend.device == "cuda" else None)

    backend.cache_warm, warm_times = _timed(backend.cache_warm, synchronize)
    if backend.cache_warm_all is not None:
        backend.cache_warm_all, warm_all_times = _timed(backend.cache_warm_all, synchronize)
    else:
        warm_all_times = []
    backend.score_block, score_times = _timed(backend.score_block, synchronize)

    from bench.run import _SPEED_PROMPTS, fit_prompt
    prompt, plen = fit_prompt(_SPEED_PROMPTS, backend.tokenizer,
                              args.prompt_len)
    print(f"[setup] device={backend.device} dtype={args.dtype} "
          f"prompt_len={plen} out_len={args.out_len} "
          f"block_size={args.block_size} steps={args.steps} "
          f"warm_batch={args.warm_batch}")

    # one untimed warmup so first-call jit/lazy-init noise stays out
    backend.generate(prompt, min(args.out_len, 32))
    warm_times.clear(); warm_all_times.clear(); score_times.clear()

    for run in range(args.runs):
        t0 = time.perf_counter()
        g = backend.generate(prompt, args.out_len)
        outer_wall = time.perf_counter() - t0
        total = g.decode_seconds
        warm_s = sum(warm_times) + sum(warm_all_times)
        score_s = sum(score_times)
        unattributed_s = total - warm_s - score_s
        n_warm = len(warm_times) + len(warm_all_times)
        n_score = len(score_times)
        print(f"[run {run}] total={total * 1e3:.1f}ms  "
              f"warm={warm_s * 1e3:.1f}ms ({n_warm} forwards, "
              f"mean {(sum(warm_times) + sum(warm_all_times)) / max(n_warm, 1) * 1e3:.2f}ms)  "
              f"score={score_s * 1e3:.1f}ms ({n_score} passes, "
              f"mean {sum(score_times[-n_score:] or [0]) / max(n_score, 1) * 1e3:.2f}ms)  "
              f"unattributed={unattributed_s * 1e3:.1f}ms  "
              f"outer_wall={outer_wall * 1e3:.1f}ms  "
              f"passes={g.passes} warm_passes={g.warm_passes} "
              f"blocks={g.blocks} diagnostic_tps={g.num_tokens / max(total, 1e-6):.1f}")
        warm_times.clear(); warm_all_times.clear(); score_times.clear()

    # what the model itself reports as the warm subwindow, for cross-check
    print(f"[check] prefill_kind={g.prefill_kind} "
          f"prefill_seconds={g.prefill_seconds * 1e3:.1f}ms")


def microbench(args) -> None:
    """Standalone costs of the ops the cached path leans on, at the shapes
    this run actually produces. Device-labeled; CPU numbers do not predict
    T4 kernel times, they only rank the ops."""
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    B, H_Q, H_KV, D = 1, 9, 3, 64
    block = args.block_size
    cache_len = args.prompt_len + args.out_len

    def bench(label, fn, n=50):
        for _ in range(5):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        print(f"[micro] {label:44s} {ms:7.3f} ms  ({dev})")

    cached_k = torch.randn(B, H_KV, cache_len, D, device=dev, dtype=dtype)
    cached_v = torch.randn_like(cached_k)
    new_k = torch.randn(B, H_KV, block, D, device=dev, dtype=dtype)
    new_v = torch.randn_like(new_k)
    bench("update_cache torch.cat (k+v, per layer)",
          lambda: (torch.cat([cached_k, new_k], dim=2),
                   torch.cat([cached_v, new_v], dim=2)))

    k_cat = torch.cat([cached_k, new_k], dim=2)
    v_cat = torch.cat([cached_v, new_v], dim=2)
    bench("GQA repeat_interleave (k+v, per layer)",
          lambda: (k_cat.repeat_interleave(3, dim=1),
                   v_cat.repeat_interleave(3, dim=1)))

    q = torch.randn(B, H_Q, block, D, device=dev, dtype=dtype)
    k9 = k_cat.repeat_interleave(3, dim=1)
    v9 = v_cat.repeat_interleave(3, dim=1)
    bench("SDPA q=32 vs cached kv, no mask (per layer)",
          lambda: torch.nn.functional.scaled_dot_product_attention(
              q, k9, v9, is_causal=False))
    # batched-warm shape: one forward over the whole prompt, block-causal
    n = args.prompt_len
    k_warm = cached_k[:, :, :n].repeat_interleave(3, dim=1)
    v_warm = cached_v[:, :, :n].repeat_interleave(3, dim=1)
    q_warm = torch.randn(B, H_Q, n, D, device=dev, dtype=dtype)
    block_end = (torch.arange(n, device=dev) // block + 1) * block - 1
    mask = (torch.arange(n, device=dev)[None, :]
            <= block_end[:, None])[None, None]
    bench("SDPA warm fwd q=kv=1024, no mask (chunk-equiv invalid)",
          lambda: torch.nn.functional.scaled_dot_product_attention(
              q_warm, k_warm, v_warm, is_causal=False))
    bench("SDPA warm fwd q=kv=1024, block-causal mask",
          lambda: torch.nn.functional.scaled_dot_product_attention(
              q_warm, k_warm, v_warm, is_causal=False, attn_mask=mask))

    # autograd cost of one warm-style chunk forward (the benchmark never
    # needs a graph here; stock modeling_smoldlm.generate runs no_grad)
    import importlib.util

    from huggingface_hub import hf_hub_download

    mod_path = hf_hub_download(args.repo, "modeling_smoldlm.py")
    spec = importlib.util.spec_from_file_location("modeling_smoldlm", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.SmolDLM.from_pretrained(args.repo).to(dev).to(dtype)
    chunk = torch.randint(3, 1000, (1, block), device=dev)
    model.reset_kv_cache()
    model.set_cache_mode(False)

    def fwd(grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            model(chunk, pos_offset=0)

    bench("warm chunk forward, grad ON", lambda: fwd(True), n=20)
    bench("warm chunk forward, no_grad", lambda: fwd(False), n=20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="HoangHa/smoldlm-144m")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--dtype", default="float32",
                   choices=("float32", "float16"))
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--confidence", type=float, default=0.9)
    p.add_argument("--commit", default="confidence",
                   choices=("confidence", "topk"))
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--warm-batch", action="store_true",
                   help="single block-causal warm forward instead of chunks")
    p.add_argument("--sync", action="store_true",
                   help="cuda only: sync around phases for exact per-call "
                        "times (profiler-only, never benchmark-legal)")
    p.add_argument("--microbench", action="store_true",
                   help="run the op microbench instead of the loop profile")
    args = p.parse_args()
    if min(args.runs, args.block_size, args.steps, args.out_len, args.prompt_len) < 1:
        p.error("runs and all lengths/steps must be positive")
    if args.microbench:
        microbench(args)
    else:
        profile(args)


if __name__ == "__main__":
    main()
