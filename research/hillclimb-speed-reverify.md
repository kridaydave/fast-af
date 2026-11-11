# Re-verification of `hillclimb-speed-verify-ernie.md` — MAYBE list settled

Review sub-agent, 2026-11-11. Re-verification pass. Every verdict below is backed by a
source I opened in this session; the quote is copied from that source. No existing file was
edited. No GPU runs. Where a fetch failed I write FETCH FAILED and do not guess.

## Sources opened in this pass

| Source | Status | Used for |
|---|---|---|
| `research/hillclimb-speed-verify-ernie.md`, `hillclimb-speed-iceman.md`, `hillclimb-speed-iceman-verify.md` | read | claims under audit |
| `spec/edgefast-diff-100m.md`, `bench/BENCHMARK_WEEK.md`, `results_t4.yaml`, `bench/backends.py`, `README.md` | read | repo claims / line numbers |
| `/home/kriday/OpenBot/Shared/fastest-llm-4-candidates.md` | read | 1-3k provenance |
| `HoangHa/smoldlm-144m` model card | fetched | defaults, greedy warning |
| cached `modeling_smoldlm.py` (snapshot 64f1177…) | read | KV-cache implementation |
| AutoAWQ README (github.com/casper-hansen/AutoAWQ) | fetched | Turing support |
| AutoAWQ_kernels README + setup (github.com/casper-hansen/AutoAWQ_kernels) | fetched | kernel CC requirement |
| mit-han-lab/llm-awq README | fetched | AWQ context |
| vLLM quantization supported-hardware page + `awq.py` | fetched | AWQ on Turing |
| NVIDIA T4 product page (specs) | fetched | 65 FP16 TFLOPS, 320+ GB/s |
| bitsandbytes PyPI 0.50.2 | fetched | sm75 support |
| causal-conv1d `setup.py` | fetched | sm75 build target |
| fla (flash-linear-attention) README | fetched | platform claims |
| Triton README (compatibility) | fetched | CC 8.0+ requirement |
| arXiv 2610.26040 abs + HTML (BPD) | fetched | 10.35x / Table 1 |
| arXiv 2609.11998 abs + HTML (fixed-state) | fetched | PPL 15.3/6.4 / Table 5 |
| arXiv 2505.22618 HTML (Fast-dLLM) | fetched | A100 / 27.6x / Table 4 |
| ACL 2026.findings-emnlp.1492 (BlockBatch) | fetched (abstract) | 1.33x / hardware |
| PMLR v306 index (DFlash) | **FETCH FAILED** (>5 MB) | DFlash hardware not settled |

---

## Item 1 — Does AWQ run on a T4 (sm75) today?

**Verdict: YES.** The blanket "T4 has no AWQ" is wrong at the kernel, library and serving layers.

Primary quotes, in order of directness:

1. AutoAWQ README (Prerequisites):
   > "Your NVIDIA GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported."
   And in the news list: "[2023/10] Mistral (Fused Modules), Bigcode, Turing support, Memory Bug Fix (Saves 2GB VRAM)".

2. AutoAWQ_kernels README (Requirements):
   > "NVIDIA: GPU: Must be compute capability 7.5 or higher. CUDA Toolkit: Must be 11.8 or higher."
   And the build note:
   > "`COMPUTE_CAPABILITIES`: You can specify specific compute capabilities to compile for: `COMPUTE_CAPABILITIES="75,80,86,87,89,90" pip install -e .`"

3. vLLM supported-hardware table (docs.vllm.ai/en/stable/features/quantization/):
   > "| AWQ | ❌ | ✅︎ | ✅︎ | ✅︎ | ✅︎ | ... |" — columns are Volta, Turing, Ampere, Ada, Hopper. Turing is the second column and is ✅.
   The vLLM source `vllm/model_executor/layers/quantization/awq.py` states:
   > "def get_min_capability(cls) -> int: # The AWQ kernel only supports Turing or newer GPUs. return 75"
   A 2023 issue (`vllm-project/vllm#1063`, "AWQ does not support Turing GPUs") was closed by PR #1252;
   Marlin-on-Turing (sm75) was merged 2025-12-16 (PR #29901, "add marlin kernel support for turing (sm75) (e.g. 2080ti / T4)").

Caveats that do **not** rescue the spec's blanket claim:

- AutoAWQ is archived (repo banner: "This repository was archived by the owner on May 11, 2025. It is now read-only.") and deprecated ("AutoAWQ is officially deprecated and will no longer be maintained."). The maintained path is vLLM / llm-compressor.
- `pip install autoawq` "includes no external kernels and relies on Triton for inference" — and current Triton requires CC 8.0+ (see Item 5), so the **default AutoAWQ install may not work on T4**; the `autoawq[kernels]` build (CUDA 11.8+, sm75 gencode) or vLLM is the T4-viable path. This is a packaging caveat, not an architecture ban.
- `HoangHa/smoldlm-144m` is a custom architecture ("Not compatible with HuggingFace transformers AutoModel"), so AWQ cannot be pointed at it without a custom quantized checkpoint/kernel. AWQ is a viable T4 path for a standard HF model (e.g. Qwen3.5-0.8B), which is exactly the B1 target.

**Lines that must change:**

- `spec/edgefast-diff-100m.md` line 106, currently:
  > "T4 has no bf16 and no AWQ support. Full-precision tier is fp16; quantized tier is GPTQ int4 or llama.cpp Q4_K_M."
  Replace with:
  > "T4 has no bf16. AWQ **does** run on sm75 (AutoAWQ claims CC 7.5+; vLLM lists AWQ as supported on Turing and sets `get_min_capability() = 75`; Marlin sm75 kernels merged 2025-12). AWQ is a candidate quantized tier alongside GPTQ int4 / llama.cpp Q4_K_M. AutoAWQ is archived; vLLM is the maintained AWQ path. Note: the custom smoldlm-144m arch is not AWQ-quantizable without a custom kernel."
- `bench/BENCHMARK_WEEK.md` line 20, currently:
  > "B1 = Qwen3.5-0.8B int4 (bitsandbytes 4bit, compute fp16). T4 has no AWQ, so GPTQ/bnb only."
  Replace the last sentence with:
  > "AWQ is not excluded on T4: AutoAWQ/vLLM support sm75. Add an AWQ arm (Qwen3.5-0.8B AWQ) alongside bnb int4 and GPTQ; AutoAWQ is archived, so use vLLM or `autoawq[kernels]`."
- `bench/BENCHMARK_WEEK.md` line 75, currently:
  > "5. T4 has no bf16, no AWQ. fp16 + bnb int4 only."
  Replace with:
  > "5. T4 has no bf16. AWQ runs on sm75 (AutoAWQ/vLLM); quantized tier is AWQ / GPTQ / bnb int4 / llama.cpp Q4_K_M. smoldlm-144m itself needs a custom kernel for AWQ."

---

## Item 2 — Where does "1-3k tok/s" come from?

**Verdict: the candidates doc's "1-3k" is unsourced and is an editorial downgrade of the spec's ~5k. Neither is a measurement.**

Spec, `spec/edgefast-diff-100m.md` line 56 (exact):
> "Back-of-envelope on T4: 100M int4 ≈ 0.06 GB per pass, T4 bandwidth ~320 GB/s → ~5,000 passes/s ceiling, times ~3 tokens per pass ≈ 15k tok/s paper math; expect a third of that after scheduling overhead and remasking waste."

Arithmetic: 0.06 GB/pass × 320 GB/s⁻¹ → ~5,333 passes/s; × ~3 tokens/pass ≈ 16k; "a third" ≈ 5.3k. So the spec's number is **~5k**, not 1-3k.

Candidates doc, `fastest-llm-4-candidates.md` line 16 (exact):
> "**Why it is fastest:** diffusion emits several tokens per pass; GDN has no KV growth; the caches cut redundant compute. T4 baseline 350 tok/s; with the stack, realistic 1-3k tok/s."

The candidates doc's own "Sources (primary, fetched this session)" section (lines 64-84) contains **no entry** that yields 1-3k. The only previous-work entry is `spec/edgefast-diff-100m.md` (line 84). There is no T4 measurement, no paper number, and no arithmetic behind 1-3k. It is a hand-tuned haircut of the spec's ~5k.

Additional provenance nuance (from the code and Item 3): the spec's ~5k is a **memory-bandwidth ceiling** that implicitly assumes a *short* per-pass (cached) workload. The current uncached harness re-reads 1,056-2,048 tokens per pass and is compute-bound (measured 350 tok/s is ~12.7x slower than the ~4,430 tok/s weight-read ceiling), so the spec's ceiling does not describe the current loop. The cached path is what moves toward it.

**Action:** cite 1-3k only as an unsourced estimate in the candidates doc, or replace it with "spec arithmetic ≈5k ceiling; no T4 measurement above 350 tok/s uncached." Do not present 1-3k as grounded.

---

## Item 3 — The KV-cache gap: would it raise 350 tok/s, and by what mechanism?

### 3.1 The code claim is CONFIRMED

Cached copy `modeling_smoldlm.py` (snapshot 64f1177303ae2c6f021791d1b80bcd92ea911151):

- `MultiHeadAttention` owns `self.kv_cache`, `cache_mode`, `reset_cache()`, `update_cache(k, v)` (lines 112-125).
- `forward` concatenates cached K/V and runs `F.scaled_dot_product_attention(q, k, v, is_causal=False)` (lines 147-158); it writes to the cache only when `cache_mode` is True (line 160-161).
- `SmolDLM.set_cache_mode` / `reset_kv_cache` (lines 240-246); `forward(idx, pos_offset=0)` slices RoPE by `pos_offset` (lines 248-260).
- `generate()` (lines 330-457): defaults `steps=10, temperature=0.7`; warms the cache over full prompt blocks with `model.set_cache_mode(True); ... model(block_ids, pos_offset=pos_offset)` (lines 368-376); each denoise step runs `logits = model(block, pos_offset=pos_offset)` with `cache_mode` False (line 407), so the cached prefix is read and the 32-token block is recomputed; after a block it folds the block into the cache with one more `model(block, pos_offset=pos_offset)` (lines 435-438).

Our harness `bench/backends.py`:

- `score_fn(seq)` calls `model(seq)` with the cache left at its default (`kv_cache is None`, `cache_mode=False`) and pins mask/pad logits (lines 235-240).
- `CustomBackend.generate` builds `seq = torch.tensor([out_ids + block])` and runs `logits = self.score_fn(seq)` **inside** the step loop, every pass (lines 173-197).
- `load_smoldlm` never calls `set_cache_mode` (lines 213-250).

So: **the model ships a cached generate; our 350 tok/s is the uncached loop.** Confirmed from both files, not from the note.

### 3.2 Mechanism: how many tokens does the uncached path re-read per pass?

`out_ids` starts at the 1024-token prefix and grows 32 tokens per block. The uncached pass forwards the **entire** sequence `prefix + all prior blocks + current block`:

- Block 0: L = 1024 + 32 = 1,056 tokens.
- Block 31: L = 1024 + 992 + 32 = 2,048 tokens.
- Average over 32 blocks: L ≈ 1,552 tokens per pass.

The cached path forwards **only the 32-token current block** and reads the frozen prefix K/V. So the uncached path recomputes **33-64x more tokens per pass** (1,056-2,048 vs 32), across all 30 layers' Q/K/V/O and SwiGLU projections.

### 3.3 Expected gain — ESTIMATE, not measurement

Model constants (from `DEFAULT_CONFIG` and the model card): 144.47M params, 30 layers, d_model 576, 9 q-heads / 3 kv-heads, head_dim 64, MLP 1,536, vocab 49,152.
- Embedding = 49,152 × 576 = 28.31M; non-embedding ≈ 144.47 − 28.31 = **116.16M**.
- T4 fp16: **65 FP16 TFLOPS** ("Mixed Precision (FP16/FP32) 65 FP16 TFLOPS", NVIDIA T4 specs), **320+ GB/s** ("Bandwidth 320+ GB/s").

Uncached per-pass cost (why the baseline is 350, not 5,000):

- Linear+MLP FLOPs ≈ 2 × 116.16M × L = 232.3M × L. At avg L=1,552: **361 GFLOP**; attention ≈ 4·L²·d_model ≈ 5.5 GFLOP (small).
- Measured per-pass time: 1,024 tok ÷ 350 tok/s = 2.926 s; 32 blocks × 8 steps = 256 passes → **11.43 ms/pass**.
- Implied effective compute = 361 GFLOP ÷ 11.43 ms = **31.6 TFLOPS** (~49% of peak). The loop is **compute-bound on the prefix linear layers**, not memory-bound.
- Cross-check: weight read = 144.47M × 2 B = 289 MB → 0.90 ms/pass at 320 GB/s; 256 passes → 231 ms → a memory-bound ceiling of ~4,430 tok/s. Measured 350 tok/s is ~12.7x slower than that, confirming the loop is compute-bound, not memory-bound.

Cached per-pass cost (the floor):

- Linear+MLP on 32 tokens = 232.3M × 32 = **7.43 GFLOP** (0.24 ms at 31.6 TFLOPS).
- Weight read = 289 MB (all layers + tied lm_head) → **0.90 ms**.
- KV read: 2 (K,V) × 3 kv-heads × 64 head_dim × 2 B = 768 B/token/layer × 30 layers = 23,040 B/token. At L=1,552: 35.8 MB; at L=2,048: 47.2 MB.
- Total bytes ≈ 289 + 36 = **325 MB** → **1.02 ms/pass** bandwidth floor.
- Per block: with the harness's 8-step policy, 8 denoise passes + 1 cache-finalize pass = **9 passes**; with the model's default 10-step policy, 10 + 1 = **11 passes**. At the 1.02 ms floor: 9 × 1.02 = 9.2 ms/block (harness policy) or 11.2 ms/block (model default).
- Ceiling: 32 blocks × 9.2 ms = **294 ms → ~3,480 tok/s (9.9x)** with the harness policy; 32 × 11.2 = **359 ms → ~2,850 tok/s (8.1x)** with the model default, if kernels were perfectly efficient.

Realistic (eager PyTorch, 30 layers × many small kernels per forward): launch overhead dominates at 32-token GEMMs. The figures below use the 9-pass harness-policy case (the honest experiment); the 11-pass model-default case is ~18% slower.
- Pessimistic ~4-5 ms/pass → 36-45 ms/block → 1.15-1.44 s → **~710-890 tok/s (2.0-2.5x)**.
- Central ~2-3 ms/pass → 18-27 ms/block → 0.58-0.86 s → **~1,190-1,780 tok/s (3.4-5.1x)**.
- Optimistic ~1.3 ms/pass → 11.7 ms/block → 0.37 s → **~2,740 tok/s (7.8x)**.

**Estimate: enabling the cache plausibly moves 350 tok/s to ~700-3,500 tok/s, i.e. ~2x-10x, with a central estimate of ~1,200-1,800 tok/s (~3.5-5x).** The token-reduction is 33-64x, but the wall-clock gain is bounded by the ~325 MB/pass weight read and launch overhead, so it is not 33-64x. This is an estimate from the arithmetic above, not a measurement.

### 3.4 The catch: quality and the timing contract

1. **Quality / policy confound.** The model's stock `generate` uses **10 steps, top-k (`tokens_per_step = max(1, n_masked // steps)` = 3/step), temperature 0.7 (Gumbel)** and stops at EOS. Our 350 baseline is **8 steps, confidence 0.9, greedy argmax, forced 1,024 tokens**. Swapping in the stock generate changes cache *and* policy at once, so the tok/s delta is not attributable to the cache. To isolate the cache, wire only `set_cache_mode` / prefix warmup and keep the harness policy (8 steps / 0.9 / greedy). Then separately score the model's default policy (the model card warns "Greedy decoding degenerates — Temperature=0 causes mode collapse... Always use temperature >= 0.5", so the default policy may improve sample quality).
2. **Timing contract (1k in / 1k out).** `CustomBackend` times decode only: `prefix = self.tokenizer.encode(prompt).ids` is before `t0` (backends.py lines 166-168). With the cache, the prefix warmup is a **full 1,024-token forward (≈ prefill)** and must be excluded from the timer to preserve "excludes prefill". The per-block **cache-finalize forward is inside the decode window** and must be included (it is an extra forward the harness does not currently pay). The stock `generate` stops at EOS, so to compare at 1k/1k you must force 1,024 emitted tokens (the harness does; the stock generate does not).
3. **Correctness.** The cache is exact for the clean prefix. The model is trained block-causal ("Cache the completed block and move to the next 32 positions"; BD3-LM staircase masks), so cached inference matches training. The uncached full-seq forward uses `is_causal=False` over the whole sequence, so prefix K/V are recomputed with the block present (bidirectional contamination) — the **uncached path is the approximation**, not the cached one. Greedy outputs should match closely but not bit-exactly (different attention reduction order = float nondeterminism). A >1% greedy mismatch is a red flag; a small mismatch is expected. The fixed-state paper makes the same point: a cache is exact only when the model is trained with the same block-causal objective ("the cache is exact because the models are trained with the objective used at decode time").

---

## Item 4 — The two UNCONFIRMED-then-confirmed claims

### 4.1 BPD 10.35x (arXiv 2610.26040) — CONFIRMED at paper level; MISLEADING for T4/100M

Exact source: arXiv:2610.26040v1, Section 6 (Conclusion):
> "Experiments across three diffusion language models and five benchmarks demonstrate favorable quality–efficiency trade-offs, achieving up to 10.35× higher throughput than vanilla decoding with a QTG of 9.46×."

Table 1 (Dream-7B, MBPP row): Vanilla TPS **12.69**, QTG 1.00×; Ours TPS **131.36**, QTG **9.46×**. 131.36 ÷ 12.69 = **10.35**.

Hardware and scale, Section 4.1:
> "We conduct all experiments on a single NVIDIA RTX A6000 GPU with 48 GB of memory. We use semi-autoregressive decoding with a block size of 32 and a maximum generation length of 256 tokens."
> Models: "LLaDA-8B-Instruct, LLaDA-1.5, and Dream-7B."

So: **A6000, Dream-7B, MBPP, 256-token generation, 10.35x.** Not a T4 or 100M number. ERNIE's claim is exactly right.

### 4.2 Fixed-state PPL 15.3@4 vs 6.4@32 (arXiv 2609.11998) — CONFIRMED at paper level; MISLEADING for T4/100M

Exact source: arXiv:2609.11998v1, **Table 5** ("One dial, S, trades quality for speed. Per-block denoising budget S (3B, cached generation, 64→128-token continuations, 512 samples)"). Attn column, Gen-PPL:
> S=4 **15.26**; S=8 12.10; S=16 9.22; S=32 **6.44**.

Section 5.3:
> "raising S from 4 to 32 monotonically lowers generative perplexity (15.3→6.4 for Attn, 17.1→6.5 for Hybrid, 18.5→6.5 for Mamba)".

Hardware, scale, model size:
- Models, Section 4: "three 3B-parameter denoisers"; "We trained with sequence length of 1024 tokens."
- Hardware: "All efficiency measurements use a single NVIDIA H100 80 GB in bf16 with PyTorch SDPA attention kernels."
- Metric: "generative perplexity (scored by GPT-2 large)"; continuations "64→128-token".

So: **3B, H100 80 GB, 1024-token training context, 64→128-token continuations, GPT-2-large Gen-PPL, 15.26→6.44.** Not a 100M/T4 curve. ERNIE's "3B model on H100 at 1024 context" is correct; the continuation length (64→128 tokens) is the detail ERNIE omitted.

### 4.3 Bonus: 297/231 tok/s (same paper) — CONFIRMED; spec comparison MISLEADING

Section 5.1:
> "Mamba holds a flat 6.8 ms per step, 7.5 GB of memory, and ~296 tok/s"; "2.6× higher throughput (297 vs. 114 tok/s)"; Hybrid "2.0× on throughput (231 vs. 114 tok/s)".
Abstract:
> "At 256k tokens ... the Mamba cache delivers 4.3x lower latency, 11x less memory, and 2.6x higher single-stream throughput ... reaching 14x the aggregate throughput."

These are **3B, single H100 80 GB, at 256k context**. The repo already measures **350 tok/s for 144M on a T4 at 1k context**, above 297. So `spec` line 56's "at 8x smaller with the same decode trick, meaningfully higher than both is the target" is **MISLEADING**: it omits H100 vs T4 and 256k vs 1k, and the repo number already exceeds the 3B/H100/256k number. Action: compare at matched hardware/context, or drop the 297/231 floor.

---

## Item 5 — Other maybe / unconfirmed / plausible items in the ERNIE note

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 5.1 | Qwen3.5 GDN "may have used fallback kernels" | **STILL UNCLEAR** | `results_t4.yaml` notes say "Qwen3.5 GDN may have used fallback kernels"; no kernel trace, profiler log, or backend selection record exists in the repo. No primary source can settle a repo-internal "may have". Action: log selected kernel/backend in the next run. |
| 5.2 | bitsandbytes int4 slower than fp16 at batch 1 on T4 | **CONFIRMED measurement; "unsupported sm75" is FALSE** | `results_t4.yaml`: 14.744 < 18.562. bitsandbytes PyPI 0.50.2: "SM60+ minimum SM75+ recommended", QLoRA 4-bit ✅. So sm75 is supported; the regression is a kernel/config effect, not an unsupported-arch effect. The *mechanism* (dequant/packing overhead vs kernel path) remains a hypothesis → mechanism STILL UNCLEAR. |
| 5.3 | causal-conv1d on sm75 | **YES** | `causal-conv1d/setup.py`: `cc_flag.append("arch=compute_75,code=sm_75")` (then compute_80, 87, 90…). Turing is an explicit build target. |
| 5.4 | flash-linear-attention on sm75 | **NO on the current stack** | fla README: "All implementations are platform-agnostic and verified on NVIDIA, AMD, and Intel hardware." But fla is Triton-based, and Triton's README compatibility section says: "Supported Hardware: NVIDIA GPUs (Compute Capability 8.0+)". T4 is 7.5, so current fla kernels are not supported on T4 (only an older Triton that supported 7.5, if pinned, could run them). Action: treat fla as unavailable on T4 unless a pinned older Triton is verified. |
| 5.5 | vLLM AWQ on Turing | **YES** | vLLM supported-hardware table: AWQ ✅ Turing; `get_min_capability() = 75`; Marlin sm75 merged 2025-12-16. |
| 5.6 | BlockBatch 1.33x / 26.6% NFE (ACL 2026.findings-emnlp.1492) | **CONFIRMED (abstract); hardware STILL UNCLEAR** | Abstract: "BlockBatch reduces denoising NFEs by 26.6% on average and achieves a 1.33× average end-to-end speedup, with comparable average accuracy." No hardware stated in the abstract; PDF fetch was not retried (prior note: >5 MB). |
| 5.7 | DFlash "6x+ lossless, 2.5x over EAGLE-3" hardware | **FETCH FAILED** | PMLR v306 index returned "Response too large (exceeds 5MB limit)". Not settled; abstract-level claim stands, hardware unconfirmed. |
| 5.8 | README "nothing here has run on a T4 yet" | **CONFIRMED STALE** | `README.md` lines 30-32: "No GPU on the dev box, so nothing here has run on a T4 yet." `results_t4.yaml` has T4 results (smoldlm 349.979, etc., "Lean T4 setup on Colab"). Action: update or remove the stale caveat. |
| 5.9 | Cross-backend timing asymmetry | **CONFIRMED** | `backends.py` module contract line 12: `decode_seconds` = "time to emit them (excludes prefill)". But `TransformersBackend` times the whole `model.generate` (`t0` before generate, `decode_s` after, lines 74-79), including prefill; `CustomBackend` excludes the prefix encode (lines 166-168). So 350 vs 18.6 is not like-for-like; the LFM2.5 12-token row is mostly prefill. Action: either exclude prefill for both or report prefill separately. |
| 5.10 | Fast-dLLM 27.6x hardware/baseline | **CONFIRMED** | arXiv 2505.22618v3 §4.1: "All experiments are conducted on an NVIDIA A100 80GB GPU." Table 4, 8-shot, gen 1024: No Cache **0.7 tok/s (1x)** → DualCache **19.3 tok/s (27.6x)**. §4.3: "speedup for DualCache increases from 19.6× in 5-shot to 27.6× in 8-shot for generation length 1024." |
| 5.11 | DepCap / EB-Decode / CoDiLA / DScale (2026 methods) | **Not re-fetched this pass** | ERNIE's quotes stand as abstract-level claims (A100/4xA100, 7-8B, gen 256/512); no T4/100M evidence. Not decision-critical for the spec/runbook; status unchanged. |

---

## Summary table

| Item | Verdict | Evidence | Action needed in spec / runbook |
|---|---|---|---|
| 1. AWQ on T4 (sm75) | **YES** | AutoAWQ README "Compute Capability 7.5. Turing and later"; AutoAWQ_kernels "7.5 or higher" + `COMPUTE_CAPABILITIES="75,..."`; vLLM table AWQ ✅ Turing; `get_min_capability()=75`; Marlin sm75 merged 2025-12 | Change `spec` line 106 and `BENCHMARK_WEEK` lines 20 & 75: AWQ is a candidate T4 tier; AutoAWQ archived, use vLLM; smoldlm custom arch needs a custom kernel |
| 2. "1-3k tok/s" | **Unsourced downgrade of spec's ~5k** | `spec` line 56: "~5,000 passes/s ceiling × ~3 tokens ≈ 15k paper math; expect a third" (~5k); candidates line 16 has no source for 1-3k | Cite 1-3k as an unsourced estimate, or replace with "spec arithmetic ≈5k ceiling; no T4 measurement above 350 uncached" |
| 3. KV cache raises 350? | **YES, plausibly ~2x-10x (est. 700-3,500 tok/s; central ~1,200-1,800)** | Cached `generate()` in `modeling_smoldlm.py` (steps=10, temp 0.7, warmup + per-block cache); `CustomBackend` re-forwards 1,056-2,048 tokens/pass, uncached | Add an experiment: wire `set_cache_mode`/warmup, keep harness policy; separate policy test; exclude prefill from timer; include cache-finalize pass; force 1k out |
| 4a. BPD 10.35x | **CONFIRMED (paper) / MISLEADING (T4/100M)** | arXiv 2610.26040 §6 "up to 10.35×"; Table 1 Dream-7B MBPP 12.69→131.36; §4.1 A6000, 256 tokens | Keep as paper motivation only; do not multiply 350 by 10.35 |
| 4b. PPL 15.3@4 vs 6.4@32 | **CONFIRMED (paper) / MISLEADING (T4/100M)** | arXiv 2609.11998 Table 5 Attn: 15.26 (S=4) / 6.44 (S=32); §4 3B, 1024 train ctx, H100; 64→128-token continuations | Label as 3B/H100/1024-ctx; not a 100M/T4 curve |
| 5.1 Qwen GDN fallback | **STILL UNCLEAR** | `results_t4.yaml` "may have used fallback kernels"; no trace | Log kernel/backend selection in next run |
| 5.2 bnb int4 cause | **Supported on sm75; mechanism unclear** | PyPI "SM60+ minimum SM75+ recommended", QLoRA ✅; measured 14.744 < 18.562 | Drop "unsupported arch" explanation; keep as kernel/config hypothesis |
| 5.3 causal-conv1d sm75 | **YES** | setup.py `arch=compute_75,code=sm_75` | No change; installable on T4 |
| 5.4 fla sm75 | **NO (current Triton CC 8.0+)** | fla README platform-agnostic; Triton README "Compute Capability 8.0+" | Mark fla unavailable on T4 unless pinned older Triton |
| 5.5 vLLM AWQ Turing | **YES** | vLLM table AWQ ✅ Turing; min cap 75; Marlin sm75 merged | No change |
| 5.6 BlockBatch | **CONFIRMED abstract; hardware unclear** | ACL abstract: "26.6% ... 1.33× average end-to-end speedup"; no hardware in abstract | Cite as abstract-only |
| 5.7 DFlash hardware | **FETCH FAILED** | PMLR v306 >5 MB | Retry or leave abstract-only |
| 5.8 README stale | **CONFIRMED STALE** | README lines 30-32 vs `results_t4.yaml` | Update/remove "nothing here has run on a T4 yet" |
| 5.9 Timing asymmetry | **CONFIRMED** | backends.py line 12 vs lines 74-79 / 166-168 | Exclude prefill for both or report separately |
| 5.10 Fast-dLLM 27.6x | **CONFIRMED (paper)** | A100, LLaDA-8B, 8-shot, gen 1024, 0.7→19.3 tok/s | Keep as paper motivation; no transfer |
| 5.11 2026 methods | **Not re-fetched** | ERNIE abstract quotes; no T4/100M | Optional; cite as methods, not multipliers |

---

## Updated top-5 T4 experiments (KV-cache re-run scored honestly)

Prerequisite (not an experiment): fix the timing contract. `TransformersBackend` must exclude prefill (or report it separately), and the smollm2/lfm2.5 runs must emit 1,024 tokens. Without this, every cross-backend ratio is suspect. Log per-pass time, tokens/pass, passes/block, and sample text for every arm.

All targets are 5-run medians at batch 1, 1k prompt / 1k output, same prompt set. Reference: 349.979 tok/s uncached.

### 1. KV-cache re-run (wire the model's existing cache; keep the harness policy)

- **Hypothesis:** the uncached loop re-forwards 1,056-2,048 tokens per pass across all 30 layers; the model's cached path forwards only 32 tokens and reads frozen prefix K/V. This should raise tok/s several-fold, not 1.5x.
- **What to run:** (a) current `CustomBackend` 32/8/0.9/greedy; (b) same policy but with `set_cache_mode(True)` prefix warmup and `model(block, pos_offset)` decode, plus the per-block cache-finalize forward; (c) the model's stock `generate` (10 steps, top-k, temp 0.7) as a separate policy arm. Same prompt, forced 1,024 emitted tokens for (a)/(b).
- **Number to hit (estimate-based):** ≥2x over 349.979 (≥700 tok/s) as the conservative gate; stretch 3x (≥1,050 tok/s). Do **not** set 525 (1.5x) as the target — the arithmetic says 1.5x is too low and would pass a broken cache.
- **Kill criterion:** <1.5x (525 tok/s) after correct wiring; greedy output mismatch >1% vs the uncached path (position/cache bug); or the cached path produces worse sample sanity than the model's own `generate` at the same settings. A small (<1%) mismatch is expected float noise; a large one is a bug.
- **Why it is independent:** the note says "implement a prefix cache"; the cache already exists. This is wiring + measurement, and the estimate predicts a much larger win than the note's 1.5x.

### 2. Prompt-length scaling and cache correctness

- **Hypothesis:** the uncached loop's cost grows with prompt length (it re-forwards the prefix every pass); the cached path should be roughly flat. The 1k/1k number hides this; the spec's 32k context makes it decisive.
- **What to run:** cached vs uncached at prompt 128 / 512 / 1024 / 2048, output 1024, same decode policy, 5 runs. Log tok/s and greedy token match between cached and uncached (same seed/policy).
- **Number to hit:** at 2048 prompt, cached ≥1.5x uncached; cached tok/s drop from 128 to 2048 ≤10%; greedy output match ≥99%.
- **Kill criterion:** <1.2x at 2048, or mismatch >1% (cache/position bug). If the cached path is not flat, the cache is not wired correctly.

### 3. Decode-policy sweep on the cached path

- **Hypothesis:** 10 steps / top-k / temp 0.7 is not the speed optimum; fewer steps with temp 0.7 may hold quality and raise tok/s, while the harness's 0.9 + greedy is both slower and lower quality.
- **What to run:** grid steps {4, 8, 10, 16} × block {16, 32} × commit {top-k (model default), confidence 0.9} × temperature {0, 0.7}, on the cached path, fixed prompts, 5 runs per cell. Log tok/s, tokens/pass, passes/block, repetition rate, mask-token rate.
- **Number to hit:** best config ≥1.2x over the Experiment 1 winner, with sample sanity no worse than the model's own 10-step generate.
- **Kill criterion:** no config ≥1.1x without obvious degeneration (repetition, mask artifacts, sample-sanity score worse than baseline).

### 4. Quantization A/B on SmolLM2-135M: fp16 vs bnb int4 vs AWQ int4

- **Hypothesis:** the Qwen bnb regression is backend-specific, not a universal "int4 is slower at batch 1". AWQ GEMM is designed for batch <8 and **is supported on sm75** (Item 1), so it may beat fp16 on a 135M model at batch 1; bnb may not.
- **What to run:** `HuggingFaceTB/SmolLM2-135M-Instruct`, batch 1, 1k/1k, 5 runs, 50 HumanEval, three arms: fp16, bnb int4 (`load_in_4bit=True`, compute fp16), AWQ int4 (vLLM AWQ, or `autoawq[kernels]` with CUDA 11.8+; not the default Triton-only AutoAWQ install). Log tok/s, pass@1, load/fallback status.
- **Number to hit:** AWQ ≥1.2x over fp16 (≥28.7 tok/s if fp16 is 23.9), pass@1 within 2 points; bnb ≤1.05x over fp16 (reproduce the regression on a second model).
- **Kill criterion:** AWQ fails to load on sm75 (then report fp16 as the T4 default and stop pursuing AWQ for the 100M model), or <1.1x over fp16. If bnb beats fp16 on SmolLM2, the Qwen regression is model/config-specific, not a T4-wide fact.

### 5. Block early stop (min-confidence 0.97) on the cached path

- **Hypothesis:** many blocks finish before the fixed 8-10 steps; a block-level min-confidence score above 0.97 can fill the remaining slots and skip rounds, raising tok/s without quality loss.
- **What to run:** cached path, best config from Experiment 3, with and without early stop. Log mean passes/block, rounds saved, tok/s, sample sanity. Use the spec rule: block score = min confidence over remaining masked slots; threshold 0.97.
- **Number to hit:** ≥10% mean pass reduction and ≥1.1x tok/s over the Experiment 3 winner, same prompts and emitted length.
- **Kill criterion:** <3% pass reduction or <1.02x tok/s, or early stop commits a visibly worse block (mask artifacts, broken code).

**Bonus (cheap, alongside Experiment 1):** log the SDPA backend (`flash_sdp_enabled`, `mem_efficient_sdp_enabled`, `math_sdp_enabled`). On T4, FlashAttention-2 is unavailable (Ampere+), so expect the math/mem-efficient path. Try `torch.compile(mode="reduce-overhead")` on the cached decode; if it fuses the 30-layer 32-token kernels, it may lift the cached path toward the bandwidth floor. Kill if compile fails/falls back or <1.05x. This is a kernel check, not a substitute for the cache win.