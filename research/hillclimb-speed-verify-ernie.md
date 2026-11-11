# Independent verification of `research/hillclimb-speed-iceman.md` and its self-audit

Auditor: independent second reviewer. I did not write either note and I am not defending them.
Date: 2026-11-11.
Method: read both notes and the repo files; opened the primary sources (arXiv abs + HTML, Hugging Face model card + `modeling_smoldlm.py`, AutoAWQ README, FlashAttention README, bitsandbytes PyPI, NVIDIA T4 page, ACL Anthology, PMLR). No GPU runs. No existing file edited. Every quote below is copied from the source I opened; where I could not open a source, I say so.

---

## 0. Sources opened, and what failed

| Source | Status | What I used it for |
|---|---|---|
| `research/hillclimb-speed-iceman.md` | read | claims under audit |
| `research/hillclimb-speed-iceman-verify.md` | read | self-audit verdicts under audit |
| `results_t4.yaml`, `results_t4_tier1.yaml`, `results_t4_smoldlm.yaml`, `results_smoke.yaml` | read | measured numbers |
| `bench/backends.py`, `bench/config.yaml`, `bench/run.py`, `bench/BENCHMARK_WEEK.md`, `README.md`, `spec/edgefast-diff-100m.md` | read | repo claims |
| `HoangHa/smoldlm-144m` model card | fetched | model type, block size, defaults, greedy warning |
| `HoangHa/smoldlm-144m/modeling_smoldlm.py` | fetched | whether the model already has a KV cache |
| arXiv 2505.22618 abs + HTML | fetched | Fast-dLLM |
| arXiv 2509.26328 abs + HTML | fetched | Fast-dLLM v2 |
| arXiv 2506.06295 abs + HTML | fetched | dLLM-Cache |
| arXiv 2610.26040 abs + HTML | fetched | BPD |
| arXiv 2503.09573 abs | fetched | BD3-LM |
| arXiv 2609.11998 abs + HTML | fetched | fixed-state paper |
| arXiv 2604.15750 abs + HTML | fetched | DepCap |
| arXiv 2609.16450 abs + HTML | fetched | EB-Decode |
| arXiv 2603.20216 abs (+ HTML via search snippet) | fetched | CoDiLA |
| ACL Anthology 2026.findings-emnlp.1492 | fetched | BlockBatch |
| PMLR v306 chen26bv | fetched | DFlash |
| AutoAWQ GitHub README | fetched | AWQ on Turing, small-batch quantization |
| FlashAttention GitHub README | fetched | Turing support |
| bitsandbytes PyPI 0.50.2 | fetched | sm75 support |
| NVIDIA T4 product page | fetched (spec table read) | T4 hardware |
| BlockBatch PDF | **FETCH FAILED** (response > 5 MB) | abstract page used instead |
| bitsandbytes GitHub raw README | **FETCH FAILED** (404) | PyPI page used instead |
| Qwen3.5-0.8B / LFM2.5-350M model cards | not fetched | repo claims only; not independently verified |

---

## 1. Headline findings

1. **The 350 tok/s baseline is a strawman, not the model's native decode.** `HoangHa/smoldlm-144m` ships a cached, block-wise `generate()` in `modeling_smoldlm.py` (KV cache, 10 steps, top-k unmasking, temperature 0.7). The repo's `CustomBackend` bypasses that cache, re-runs the full `prefix + block` forward every pass, uses 8 steps, a 0.9 confidence threshold, and greedy argmax. The model card explicitly says greedy decoding degenerates. So 350 tok/s is a lower bound from a deliberately uncached, non-default-policy loop. The note's #1 experiment ("implement a prefix cache") is really "call the cache the model already has". This is the biggest correction in this review.

2. **"The spec's 1-3k tok/s claim" is a misattribution.** The spec never says 1-3k. It says: "100M int4 ≈ 0.06 GB per pass, T4 bandwidth ~320 GB/s → ~5,000 passes/s ceiling, times ~3 tokens per pass ≈ 15k tok/s paper math; expect a third of that after scheduling overhead and remasking waste." That is ~5k, not 1-3k. The "1-3k tok/s" figure is from `fastest-llm-4-candidates.md` ("T4 baseline 350 tok/s; with the stack, realistic 1-3k tok/s"), not the spec.

3. **"T4 has no AWQ support" is wrong.** AutoAWQ's README says: "Your NVIDIA GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported." T4 is sm75. The spec's blanket claim is false, and the note repeats it as fact. AWQ is a candidate T4 quantization path and should be tested, not dismissed.

4. **The paper multipliers are real but belong to other hardware and scales.** Fast-dLLM's 27.6x is A100, LLaDA-8B, 8-shot, 1024-token generation, and the baseline is 0.7 tok/s; the optimized absolute throughput is 19.3 tok/s. BPD's 10.35x is in the full paper (Section 6), not the abstract, and is A6000, Dream-7B, MBPP, 256-token generation. The fixed-state PPL 15.3/6.4 is real (Table 5, Attn column) but is a 3B model on H100 at 1024 context. None of these transfer to a 100M model on a T4 as a multiplier.

5. **The self-audit is wrong on two "UNCONFIRMED" verdicts.** The BPD 10.35x figure is in the paper body (Section 6), and the fixed-state 15.3/6.4 figure is in Table 5. I confirm both at paper level. They are still MISLEADING for a 100M/T4 target, but "UNCONFIRMED" is the wrong label.

6. **The note's "web search failed, 2026 unknown" is a session limit, not a field limit.** I found and verified several 2026 methods: DepCap (5.63x), EB-Decode (3.53-18.76x), BlockBatch (1.33x avg over Fast-dLLM), CoDiLA (2x), DFlash (6x+ lossless, 2.5x over EAGLE-3), DScale (43.9-48.8% over DFlash). All are 7-8B-class or A100/A6000; none is a 100M/T4 number, but they fill the note's gap.

7. **Repo inconsistencies.** `README.md` still says "nothing here has run on a T4 yet" while `results_t4.yaml` contains T4 results. `bench/backends.py`'s module contract says `decode_seconds` "excludes prefill", but `TransformersBackend` times the whole `model.generate` including prefill; `CustomBackend` times decode only. So the AR and diffusion rows are not timed the same way.

---

## 2. Repo claims, checked against the actual files

### 2.1 Measured numbers (`results_t4.yaml`)

| Claim | Verdict | Evidence |
|---|---|---|
| smoldlm-144m median 349.979 tok/s, 1024 tokens/run | CONFIRMED | `results_t4.yaml`: `median_tps: 349.979`, `tokens_per_run: [1024 x5]`, `prompt_tokens: 1024` |
| Qwen3.5-0.8B fp16 18.562 tok/s, pass@1 0.06 | CONFIRMED | `median_tps: 18.562`, `passed: 3`, `pass@1: 0.06`, 1024 tokens/run |
| Qwen3.5-0.8B int4 14.744 tok/s, pass@1 0.02 | CONFIRMED | `median_tps: 14.744`, `passed: 1`, `pass@1: 0.02`, 1024 tokens/run |
| SmolLM2-135M 23.912 tok/s, 307 tokens/run, pass@1 0.0 | CONFIRMED | `median_tps: 23.912`, `tokens_per_run: [307 x5]`, `passed: 0` |
| LFM2.5-350M 41.437 tok/s, 12 tokens/run, pass@1 0.1 | CONFIRMED | `median_tps: 41.437`, `tokens_per_run: [12 x5]`, `passed: 5` |
| Lean env missing flash-linear-attention, causal-conv1d, dLLM | CONFIRMED | `results_t4.yaml` notes: "Lean T4 setup ... without flash-linear-attention, causal-conv1d, or dLLM." |
| Qwen GDN "may have used fallback kernels" | UNCONFIRMED | The file says "may have"; no kernel trace in the repo. Cannot confirm. |
| smoldlm is speed-only; sample degenerate | CONFIRMED | `humaneval` absent for smoldlm; sample is repeated `]]]]`; model card: "Greedy decoding degenerates — Temperature=0 causes mode collapse" |

**Important measurement caveat the note understates:** LFM2.5 emitted 12 tokens in a window that also includes a 1025-token prefill. `TransformersBackend` times the whole `model.generate` call (`t0` before `generate`, `decode_s` after), so 41.4 tok/s is mostly a prefill rate, not a decode rate. SmolLM2's 307/1024 is less extreme but still not a 1k-out number. The note says "not comparable"; that is correct, but the LFM2.5 number is closer to meaningless as a decode rate.

### 2.2 `bench/backends.py` and `bench/config.yaml`

| Claim | Verdict | Evidence |
|---|---|---|
| CustomBackend: block_size 32, steps 8, confidence 0.9 | CONFIRMED | `config.yaml`: `block_size: 32`, `steps: 8`, `confidence: 0.9`; `run.py` passes them to `load_smoldlm` |
| Unmask slots with `conf >= 0.9`; top-1 commit if none; forced fill on last pass | CONFIRMED | `backends.py`: `fill = ... [i for i in open_slots if conf[i] >= self.confidence]`; `if not fill: fill = [max(open_slots, key=...)]`; `if step == self.steps - 1: fill = open_slots` |
| No block early stop, no prompt cache, no block overlap | CONFIRMED | No such code in `CustomBackend`; the docstring says the D1 knobs "belong to the real model's loop, not this prototype" |
| Timing includes full forward over `prefix + block` each pass | CONFIRMED | `seq = torch.tensor([out_ids + block])`; `logits = self.score_fn(seq)` inside the step loop |
| `TransformersBackend` times full `model.generate`, including prefill | CONFIRMED | `t0 = time.perf_counter()` before `self.model.generate(...)`; `decode_s = time.perf_counter() - t0` |
| int4 uses `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` | CONFIRMED | `backends.py` lines 58-59 |
| `CustomBackend` times decode only (prefix encode outside timer) | CONFIRMED | `prefix = self.tokenizer.encode(prompt).ids` before `t0`; `t0` after `out_ids = list(prefix)` |

**Repo bug the note does not flag:** the module docstring says `decode_seconds` is "time to emit them (excludes prefill)", but `TransformersBackend` includes prefill and `CustomBackend` does not. The cross-backend comparison (350 vs 18.6) is therefore not a like-for-like timing comparison, even though both used 1k in / 1k out. The absolute effect on Qwen is small (prefill ~1s of a ~55s run), but the LFM2.5 12-token row is dominated by prefill.

### 2.3 `HoangHa/smoldlm-144m` — the model is not what the note's baseline implies

Model card, fetched:

- "SmolDLM is a 144M parameter block diffusion language model"
- "Block size 32 tokens"
- "steps | 10 | Denoising steps per block. More steps = higher quality, slower generation."
- "temperature | 0.7 | Sampling temperature. 0 = greedy (not recommended for dLLMs), higher = more random."
- "Greedy decoding degenerates — Temperature=0 causes mode collapse (known dLLM artifact). Always use temperature >= 0.5."
- "This is an early research checkpoint ... trained for 25K steps (~26B tokens of 100B available)."
- "Not compatible with HuggingFace transformers AutoModel."

`modeling_smoldlm.py`, fetched:

- `MultiHeadAttention` has `self.kv_cache`, `set_cache_mode`, `reset_kv_cache`, `update_cache`.
- `forward` uses `F.scaled_dot_product_attention(q, k, v, is_causal=False)` and concatenates cached K/V when `kv_cache` is set.
- `generate()` warms the cache on full prompt blocks: `model.set_cache_mode(True); for i in range(n_full): ... model(block_ids, pos_offset=pos_offset)`.
- Per block it runs `logits = model(block, pos_offset=pos_offset)` with the cache on, unmasks `tokens_per_step = max(1, n_masked // steps)` by `topk`, and uses `_add_gumbel_noise(logits, temperature)` with `temperature=0.7`.
- The cache is exact for the clean prefix; the model is trained block-causal, so the prefix K/V are reusable.

**Conclusion:** the model already implements the "prefix cache" the note ranks as experiment #1. The harness's `CustomBackend` calls `model(seq)` with `cache_mode` left at its default (`False`), so it recomputes the prefix every pass. The 350 tok/s number measures the uncached loop, not the model's native decode. The note's "current loop re-scores `prefix+block` every pass" is true, but its proposed fix is not new engineering; it is wiring to an existing code path. The target should be "match the model's own cached generate", not "invent a cache".

### 2.4 Hardware and kernel claims

| Claim | Verdict | Evidence |
|---|---|---|
| T4 is sm75, 16 GB, ~320 GB/s, no bf16 | CONFIRMED | NVIDIA T4 page: "Turing Tensor Cores 320", "Capacity 16 GB GDDR6", "Bandwidth 320+ GB/s", "Mixed Precision (FP16/FP32) 65 FP16 TFLOPS"; no BF16 row. FlashAttention README: "bf16 requires Ampere, Ada, or Hopper GPUs." |
| FlashAttention-style Ampere+ kernels unavailable/degraded on Turing | CONFIRMED | FlashAttention README: "FlashAttention-2 with CUDA currently supports: Ampere, Ada, or Hopper GPUs ... For Turing GPUs (T4, RTX 2080), see the separate flash-attention-turing repo, which supports a core subset of FlashAttention features on Turing." |
| AWQ not supported on Turing | **WRONG** | AutoAWQ README: "Your NVIDIA GPU(s) must be of Compute Capability 7.5. Turing and later architectures are supported." Also "[2023/10] ... Turing support". Caveat: AutoAWQ is archived (May 2025) and vLLM's AWQ path may differ; the blanket claim is still false. |
| bitsandbytes int4 runs on T4 but is slower at batch 1 | CONFIRMED for the measurement; UNCONFIRMED for the cause | `results_t4.yaml`: 14.744 < 18.562. bitsandbytes PyPI: "NVIDIA GPU cuda SM60+ minimum SM75+ recommended", QLoRA 4-bit supported. So sm75 is supported; "suboptimal kernel path" is speculation, not a documented fact. |
| causal-conv1d / flash-linear-attention sm75 support | UNCONFIRMED | No primary support matrix fetched. The note correctly marks this unconfirmed. |
| vLLM AWQ on Turing | UNCONFIRMED | Not fetched. The AutoAWQ kernel claim is enough to reject "no AWQ", but the vLLM serving path is a separate question. |

---

## 3. Paper claims, checked against the primary text

### 3.1 Fast-dLLM (arXiv 2505.22618)

| Claim in the note | Verdict | Quote / evidence |
|---|---|---|
| Up to 27.6x throughput over vanilla dLLM baselines, KV cache + confidence-aware parallel decoding, LLaDA/Dream | CONFIRMED | Abstract: "Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to **27.6× throughput** improvement with minimal accuracy loss". |
| "especially with long prefill" | CONFIRMED (full text) | §4.3: "as the prefill length increases from 5-shot to 8-shot, the speedup obtained by both versions of KV Cache rises significantly (e.g., speedup for DualCache increases from 19.6× in 5-shot to 27.6× in 8-shot for generation length 1024)." Also Table 5: DualCache grows from 9.4x at gen 256 to 27.6x at gen 1024. |
| Not a T4 or 100M number | CONFIRMED, but the note omits the hardware and the weak baseline | §4.1: "All experiments are conducted on an NVIDIA A100 80GB GPU." Figure 1: "We use NVIDIA A100 GPU with a single batch size". Table 4: LLaDA 8-shot, gen 1024, No Cache **0.7 tok/s (1x)**, DualCache **19.3 tok/s (27.6x)**. The 27.6x is against a 0.7 tok/s one-token-per-pass baseline; the optimized absolute throughput is 19.3 tok/s on an 8B model. |
| Parallel decoding alone gives up to 13.3x | CONFIRMED (not in the note) | §1: "This approach accelerates inference by 13.3× while preserving output quality." Table 4: 0.7 → 9.3 tok/s (13.3x). |

**Adversarial read:** 27.6x is a real paper number, but it is a ratio against a pathologically slow baseline (0.7 tok/s). Multiplying the repo's 350 tok/s by 27.6 would give ~9,660 tok/s, which is not a supported inference. The note uses it as "paper motivation" rather than a target, so it is not an outright error, but the table's placement next to 350 tok/s invites the wrong mental arithmetic. The correct takeaway is: caching + parallel decoding is the lever; the multiplier is not portable.

### 3.2 Fast-dLLM v2 (arXiv 2509.26328)

| Claim | Verdict | Quote / evidence |
|---|---|---|
| ~1B tokens fine-tuning, 500x reduction vs Dream | CONFIRMED | Abstract: "requiring only approximately 1B tokens of fine-tuning. This represents a 500x reduction in training data compared to full-attention diffusion LLMs such as Dream (580B tokens)". |
| Hierarchical caching: block-level cache + sub-block cache | CONFIRMED | Abstract: "a block-level cache that stores historical context representations across blocks, and a sub-block cache that enables efficient parallel generation within partially decoded blocks." |
| Up to 2.5x speedup over standard AR decoding, no quality loss | CONFIRMED | Abstract: "achieves up to 2.5x speedup over standard AR decoding without compromising generation quality." |
| Not the same baseline as 350 tok/s for a 144M block diffusion | CONFIRMED reasoning | §4.1: "We conduct adaptation experiments on the Qwen-2.5 1.5B and 7B Instruct models"; "64 NVIDIA A100 GPUs for training"; Figure 5: "on both NVIDIA A100 and H100 GPUs"; Figure 1: "Fast-dLLM v2 (7B) achieves 2.54x higher throughput than Qwen2.5-7B-Instruct". |

**Adversarial read:** the 2.5x is over AR decoding of the *same adapted 1.5B/7B model* on A100/H100, not over a 144M block-diffusion model on a T4. The note's caution is correct.

### 3.3 dLLM-Cache (arXiv 2506.06295)

| Claim | Verdict | Quote / evidence |
|---|---|---|
| Training-free adaptive caching, prompt caching + partial response updates | CONFIRMED | Abstract: "a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity." |
| Up to 9.1x FLOPs reduction on LongBench-HotpotQA | CONFIRMED | Abstract: "achieves up to 9.1x FLOPs reduction on LongBench-HotpotQA while maintaining competitive output quality." |
| Tested on LLaDA 8B and Dream 7B | CONFIRMED | Abstract: "including LLaDA 8B and Dream 7B". |
| ICML 2026 acceptance | CONFIRMED | arXiv page: "Comments: Accepted by ICML 2026". |
| 7B evidence only; FLOPs != tok/s | CONFIRMED | §4.1: "All experiments were conducted on the NVIDIA RTX 4090 GPUs." Table 1: LLaDA 8B GSM8K 7.32 → 23.19 TPS (3.17x), FLOPs 16.12T → 3.21T (5.02x); GPQA 5.12 → 25.23 TPS (4.93x), FLOPs 22.97T → 3.20T (7.18x). Table 2: Dream 7B GPQA 5.80 → 30.95 TPS (5.33x). The wall-clock TPS gains are much smaller than the FLOPs gains. |

**Adversarial read:** the note's "FLOPs reduction != tok/s on T4" is correct, and the paper's own RTX 4090 TPS numbers prove it (e.g., 7.18x FLOPs vs 4.93x TPS on GPQA). The note does not give the hardware; I add it: RTX 4090, 7-8B models.

### 3.4 BPD (arXiv 2610.26040)

| Claim | Verdict | Quote / evidence |
|---|---|---|
| Central pivots in contiguous masked intervals, parallel decode, logarithmic depth | CONFIRMED | Abstract: "favors central pivots within contiguous masked intervals and decodes the selected positions in parallel. Balanced pivots yield logarithmic decoding depth in the interval length." |
| Confidence-based prefilling + confidence-thresholded center selection | CONFIRMED | Abstract: "combining confidence-based prefilling with confidence-thresholded center selection." |
| Evaluated on LLaDA and Dream across 5 math/code/instruction benchmarks | CONFIRMED | Abstract: "on LLaDA and Dream across five mathematical reasoning, code generation, and instruction-following benchmarks." |
| "10.35x not in the fetched abstract" (self-audit) | CONFIRMED as a statement about the abstract, but **the full paper has it** | §6 Conclusion: "achieving up to 10.35× higher throughput than vanilla decoding with a QTG of 9.46×." Table 1: Dream-7B MBPP, Vanilla 12.69 TPS → Ours 131.36 TPS = 10.35x. |
| 10.35x is a T4/100M number | **MISLEADING** | §4.1: "We conduct all experiments on a single NVIDIA RTX A6000 GPU with 48 GB of memory." "block size of 32 and a maximum generation length of 256 tokens." Models: LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B. The 10.35x is Dream-7B MBPP, 256 tokens, A6000. |
| No small-model (100M) evidence | CONFIRMED | Abstract and §4.1 list only 7-8B models. |

**Self-audit disagreement:** the self-audit's "UNCONFIRMED" on 10.35x is wrong. The abstract does not contain it, but the paper's Section 6 does, and Table 1 gives the exact 12.69 → 131.36 calculation. The right verdict is CONFIRMED (paper) / MISLEADING (T4/100M).

### 3.5 BD3-LM (arXiv 2503.09573)

| Claim | Verdict | Quote / evidence |
|---|---|---|
| Block diffusion interpolates AR and diffusion; KV cache + parallel token sampling | CONFIRMED | Abstract: "a class of block diffusion language models that interpolate between discrete denoising diffusion and autoregressive models ... improving inference efficiency with KV caching and parallel token sampling." |
| ICLR 2025 Oral | CONFIRMED | arXiv page: "Comments: ICLR 2025 Oral." |

### 3.6 Fixed-state paper (arXiv 2609.11998)

| Claim | Verdict | Quote / evidence |
|---|---|---|
| Generative PPL 15.3 at 4 steps vs 6.4 at 32 | CONFIRMED at paper level | Table 5: Attn Gen-PPL S=4 **15.26**, S=8 12.10, S=16 9.22, S=32 **6.44**. The note's 15.3/6.4 matches the Attn column. |
| The setting is a 100M/T4-quality curve | **MISLEADING** | §4: "three 3B-parameter denoisers"; "All efficiency measurements use a single NVIDIA H100 80 GB in bf16"; "We trained with sequence length of 1024 tokens"; Table 5: "cached generation, 64→128-token continuations, 512 samples". It is a 3B model on H100 at 1024 context, scored by GPT-2-large PPL, not a 100M/T4 curve. |
| 297 tok/s single-stream for 3B pure-Mamba at 256k, 231 for hybrid | CONFIRMED | §5.1: "Mamba holds a flat 6.8 ms per step, 7.5 GB of memory, and ~296 tok/s"; "297 vs. 114 tok/s"; "Hybrid ... 231 vs. 114 tok/s". |
| The spec's "at 8x smaller ... meaningfully higher than both is the target" | **MISLEADING** | The 297/231 numbers are 3B on H100 at **256k context**. The repo already measures **350 tok/s** for a 144M model on a **T4 at 1k context**. The repo number is higher than 297 despite a 20x weaker GPU and 256x shorter context. The spec's comparison omits both hardware and context length, so it does not establish a useful floor. |
| 4.3x lower latency, 11x less memory, 2.6x throughput, 14x aggregate | CONFIRMED | Abstract: "At 256k tokens ... the Mamba cache delivers 4.3x lower latency, 11x less memory, and 2.6x higher single-stream throughput ... reaching 14x the aggregate throughput." |

**Self-audit disagreement:** the self-audit's "UNCONFIRMED" on 15.3/6.4 is wrong. Table 5 gives the exact numbers. The right verdict is CONFIRMED (paper, 3B/H100/1024 ctx) / MISLEADING (100M/T4).

---

## 4. Self-audit verdicts: where I agree and disagree

| Self-audit claim | My verdict | Why |
|---|---|---|
| Repo baseline numbers CONFIRMED | AGREE | All numbers match `results_t4.yaml`. |
| Lean-env caveat CONFIRMED | AGREE | File notes match. |
| Prototype decode loop CONFIRMED | AGREE | Code matches. |
| `TransformersBackend` prefill timing CONFIRMED | AGREE | Code matches; but the self-audit does not flag the cross-backend asymmetry (AR includes prefill, diffusion does not). |
| Spec "no AWQ" CONFIRMED | **DISAGREE** | The self-audit confirms the spec *says* it, not that it is true. AutoAWQ supports CC 7.5+ (Turing). The spec claim is wrong. |
| Fixed-state PPL 15.3/6.4 UNCONFIRMED | **DISAGREE** | Table 5 confirms 15.26/6.44. CONFIRMED at paper level; MISLEADING at 100M/T4. |
| BPD 10.35x UNCONFIRMED | **DISAGREE** | Section 6 and Table 1 confirm 10.35x (Dream-7B MBPP, A6000). CONFIRMED at paper level; MISLEADING at T4/100M. |
| Fast-dLLM "especially with long prefill" UNCONFIRMED | **DISAGREE** | §4.3 and Table 5 confirm it. CONFIRMED (full text). |
| Fast-dLLM 27.6x CONFIRMED with hardware caveat | AGREE, and strengthen | The self-audit says "abstract does not state GPU"; the full text says A100 80GB and gives the 0.7 tok/s baseline. Add both. |
| Fast-dLLM v2 2.5x CONFIRMED | AGREE, and strengthen | A100/H100, Qwen2.5 1.5B/7B. |
| dLLM-Cache 9.1x CONFIRMED | AGREE, and strengthen | RTX 4090; TPS gains 1.28-5.33x, not 9.1x. |
| BPD method CONFIRMED | AGREE | Abstract matches. |
| BD3-LM CONFIRMED | AGREE | Abstract matches. |
| int4 mechanism UNCONFIRMED | AGREE | Mechanism is hypothesis; bnb supports sm75, so "bad sm75 path" is not established. |
| 2026 methods beyond BPD UNCONFIRMED/UNKNOWN | **DISAGREE** | I found DepCap, EB-Decode, BlockBatch, CoDiLA, DFlash, DScale. All are 7-8B/A100/A6000, none T4/100M, but the field is not unknown. |
| "Web search failed (HTTP 402)" | CONFIRMED for that session; not a field fact | My search worked. The note's 2026 gap is a session limitation. |

---

## 5. What the notes missed in the speed field

### 5.1 The model already has the cache

The single most important missing fact: `modeling_smoldlm.py` implements `kv_cache`, `set_cache_mode`, `reset_kv_cache`, `update_cache`, and a `generate()` that warms the prefix cache and decodes block-by-block. The note's #1 experiment is "implement a prefix cache"; the correct experiment is "use the existing one and measure the delta". This changes the expected effect size and the engineering cost.

### 5.2 Cross-backend timing asymmetry

`TransformersBackend` includes prefill in `decode_seconds`; `CustomBackend` does not. The 350 vs 18.6 comparison is not like-for-like. The effect is small for Qwen (1k prefill in a ~55s run) but large for LFM2.5 (12 tokens, 1k prefill). Any future table should either exclude prefill for both or report prefill separately.

### 5.3 AWQ on Turing

The spec/note's "no AWQ" is wrong. AutoAWQ supports Turing (CC 7.5+), and its README explicitly says: "GEMM (quantized): Much faster than FP16 at batch sizes below 8" and "At small batch sizes with small 7B models, we are memory-bound ... quantized models faster because your weights are 3x smaller". This is directly relevant to the note's int4 question: the bnb regression is not evidence that all 4-bit is slower at batch 1. AWQ GEMM is designed for exactly that regime. Caveat: AutoAWQ is archived; vLLM's AWQ path may differ; and the smoldlm model is custom, so AWQ would need a custom kernel or a different model for the A/B.

### 5.4 2026 decode methods (all verified, none T4/100M)

| Method | Source | Verified number | Hardware / scale |
|---|---|---|---|
| DepCap | arXiv 2604.15750 | "up to 5.63x speedup"; "average 3.57x speedup over vanilla decoding" (LLaDA family, 8 task-model pairs) | "single NVIDIA A100 40GB GPU"; LLaDA-8B-Instruct, Dream-v0-base-7B, LLaDA-1.5; gen 256 |
| EB-Decode | arXiv 2609.16450 | "3.53-18.76x higher throughput than the vanilla decoding method and up to 1.58x higher throughput over ... Fast-dLLM" | "4xA100" (Table 6); LLaDA-8B-Instruct, Dream-v0-Instruct-7B, LLaDA-1.5; gen 256/512; routers trained (not training-free) |
| BlockBatch | ACL 2026.findings-emnlp.1492 | "reduces denoising NFEs by 26.6% on average and achieves a 1.33x average end-to-end speedup" | Three dLLMs, four datasets; PDF fetch failed, hardware not confirmed from abstract |
| CoDiLA | arXiv 2603.20216 | "maintains the base model's accuracy through dynamic (threshold-based) parallelism at 2x speedup" (HTML body) | Dream-Coder-Instruct-7B; 0.6B auxiliary AR; ICML 2026 |
| DFlash | PMLR v306:15343-15355 | "over 6x lossless acceleration ... up to 2.5x higher speedup than ... EAGLE-3" | ICML 2026; models/tasks not in abstract |
| DScale | arXiv 2609.37532 | "43.9% and 48.8% over DFlash" (Qwen3-8B/4B, concurrency 8-32) | "A100-40GB"; high-concurrency spec-decode, not batch-1 latency |

**Adversarial read:** none of these is a 100M/T4 number. DepCap's "vanilla" is one-token-per-step, so 3.57x is against a weak baseline. EB-Decode needs trained routers (0.6M + 0.03M params), so it is not a drop-in training-free policy. BlockBatch targets low-batch latency, which is the T4 regime, but its 1.33x is an average over 7-8B models on unspecified hardware. Treat them as method candidates, not multipliers.

### 5.5 Small-batch quantization on Turing/sm75

- AutoAWQ: "GEMV (quantized): 20% faster than GEMM, only batch size 1"; "GEMM (quantized): Much faster than FP16 at batch sizes below 8". RTX 4090 benchmarks: TinyLlama 1B GEMV batch 1 decode 548.993 tok/s (64 prefill) / 431.468 tok/s (2048 prefill); Mistral 7B GEMM batch 1 decode 156.317 / 114.355 tok/s. These are 4090 numbers, not T4, but they show the direction.
- bitsandbytes: sm75 is supported ("SM60+ minimum, SM75+ recommended"), so the Qwen regression is not an unsupported-architecture story. It is a kernel/config story, and the note correctly leaves the mechanism unconfirmed.
- The note's "int4 may hurt at batch 1" is true for the measured bnb Qwen config, but false as a general statement: AWQ GEMM is explicitly faster than FP16 below batch 8.

### 5.6 FLOPs vs wall clock

The dLLM-Cache paper is the cleanest example: 9.1x FLOPs reduction, but 1.28-5.33x TPS on RTX 4090. The note says this; I add the numbers. The same caution applies to Fast-dLLM (27.6x is throughput, not FLOPs, but against a 0.7 tok/s baseline) and BPD (10.35x is throughput, A6000, 256 tokens).

---

## 6. Verdict table

| # | Claim | Verdict | Truth |
|---:|---|---|---|
| 1 | smoldlm-144m 349.979 tok/s, 1024 tok/run | CONFIRMED | `results_t4.yaml` |
| 2 | Qwen3.5-0.8B fp16 18.562, pass@1 0.06 | CONFIRMED | `results_t4.yaml` |
| 3 | Qwen3.5-0.8B int4 14.744, pass@1 0.02 | CONFIRMED | `results_t4.yaml` |
| 4 | SmolLM2-135M 23.912, 307 tok/run, pass@1 0.0 | CONFIRMED | `results_t4.yaml` |
| 5 | LFM2.5-350M 41.437, 12 tok/run, pass@1 0.1 | CONFIRMED | `results_t4.yaml`; 41.4 is mostly prefill, not decode |
| 6 | Lean env missing fla/causal-conv1d/dLLM | CONFIRMED | `results_t4.yaml` notes |
| 7 | Qwen GDN used fallback kernels | UNCONFIRMED | File says "may have"; no trace |
| 8 | CustomBackend 32/8/0.9, top-1 fallback, forced fill | CONFIRMED | `config.yaml`, `backends.py` |
| 9 | No early stop / cache / overlap in prototype | CONFIRMED | `backends.py` |
| 10 | Prototype re-scores `prefix+block` every pass | CONFIRMED | `backends.py` |
| 11 | `TransformersBackend` includes prefill; `CustomBackend` does not | CONFIRMED | `backends.py`; cross-backend asymmetry is real |
| 12 | T4 sm75, 16 GB, 320+ GB/s, no bf16 | CONFIRMED | NVIDIA T4 page; FlashAttention README |
| 13 | FlashAttention unavailable/degraded on Turing | CONFIRMED | FlashAttention README (separate `flash-attention-turing` fork) |
| 14 | AWQ not supported on Turing | **WRONG** | AutoAWQ: "Compute Capability 7.5. Turing and later architectures are supported." |
| 15 | bitsandbytes int4 slower than fp16 at batch 1 on T4 | CONFIRMED (measurement) | 14.744 < 18.562 |
| 16 | Cause: bad sm75 kernel path | UNCONFIRMED | bnb supports sm75; mechanism not established |
| 17 | Fast-dLLM 27.6x | CONFIRMED (paper) / MISLEADING (T4/100M) | A100, LLaDA-8B, 8-shot, gen 1024; baseline 0.7 tok/s → 19.3 tok/s |
| 18 | Fast-dLLM gains grow with prefill | CONFIRMED | §4.3, Table 5 |
| 19 | Fast-dLLM v2 2.5x over AR | CONFIRMED (paper) / MISLEADING (T4/100M) | A100/H100, Qwen2.5 1.5B/7B |
| 20 | dLLM-Cache 9.1x FLOPs reduction | CONFIRMED (paper) / MISLEADING as tok/s | RTX 4090, LLaDA 8B/Dream 7B; TPS 1.28-5.33x |
| 21 | dLLM-Cache ICML 2026 | CONFIRMED | arXiv comments |
| 22 | BPD method description | CONFIRMED | Abstract |
| 23 | BPD 10.35x | CONFIRMED (paper) / MISLEADING (T4/100M) | §6; Dream-7B MBPP, 12.69 → 131.36 TPS, A6000, 256 gen |
| 24 | BPD has no 100M evidence | CONFIRMED | Only 7-8B models |
| 25 | BD3-LM block diffusion + KV cache | CONFIRMED | Abstract |
| 26 | Fixed-state PPL 15.3@4 vs 6.4@32 | CONFIRMED (paper) / MISLEADING (T4/100M) | Table 5 Attn: 15.26 / 6.44; 3B, H100, 1024 ctx, 64-128-token continuations |
| 27 | Fixed-state 297 tok/s Mamba, 231 hybrid at 256k | CONFIRMED | §5.1; H100, 3B, 256k |
| 28 | Spec's "8x smaller → meaningfully higher" target | MISLEADING | Repo already 350 tok/s on 144M/T4/1k, above 297; spec omits hardware and context |
| 29 | "Spec's 1-3k tok/s claim" | **WRONG / MISATTRIBUTED** | Spec says ~15k paper math, expect ~5k; 1-3k is from `fastest-llm-4-candidates.md` |
| 30 | 100M int4 ~0.06 GB/pass; ~5,000 passes/s ceiling | CONFIRMED (arithmetic) | 0.1B × 0.5 B = 0.05 GB; 320/0.06 ≈ 5,333 |
| 31 | 350 tok/s is the model's native block-diffusion baseline | **MISLEADING** | It is the uncached 8-step/0.9/greedy harness loop; the model ships a cached 10-step/0.7/top-k generate |
| 32 | smoldlm greedy sample degenerate | CONFIRMED | Model card: "Greedy decoding degenerates"; sample is repeated `]]]]` |
| 33 | README "nothing here has run on a T4 yet" | CONFIRMED as text / STALE | `results_t4.yaml` has T4 results |
| 34 | Web search failed; 2026 coverage unknown | CONFIRMED for that session / WRONG as a field statement | I found DepCap, EB-Decode, BlockBatch, CoDiLA, DFlash, DScale |
| 35 | DepCap 5.63x, 3.57x avg | CONFIRMED (paper) / MISLEADING (T4/100M) | A100 40GB, 7-8B, gen 256; vanilla is one-token-per-step |
| 36 | EB-Decode 3.53-18.76x, 1.58x over Fast-dLLM | CONFIRMED (paper) / MISLEADING (T4/100M) | 4xA100, 7-8B, gen 256/512; routers trained |
| 37 | BlockBatch 1.33x avg, 26.6% NFE reduction | CONFIRMED (abstract) / hardware UNCONFIRMED | ACL 2026; PDF fetch failed |
| 38 | CoDiLA 2x with 0.6B auxiliary AR | CONFIRMED (HTML body) / MISLEADING (T4/100M) | Dream-Coder-7B, ICML 2026 |
| 39 | DFlash 6x+ lossless, 2.5x over EAGLE-3 | CONFIRMED (abstract) / hardware UNCONFIRMED | PMLR 306, ICML 2026 |
| 40 | DScale 43.9-48.8% over DFlash | CONFIRMED (HTML body) / not batch-1 | A100-40GB, concurrency 8-32 |

---

## 7. Corrections

1. **Replace "the measured 350 tok/s baseline" with "the measured 350 tok/s uncached-loop baseline".** The model's own `generate()` is cached and uses a different policy. The fair baseline is the cached path; 350 is a lower bound. Any target expressed as "1.3x over 350" may already be met by wiring the existing cache.

2. **Delete "the spec's 1-3k tok/s claim".** The spec says ~15k paper math and "expect a third" (~5k). The 1-3k figure is from `fastest-llm-4-candidates.md`. Cite it there, or drop it.

3. **Delete "AWQ not supported on Turing".** AutoAWQ supports CC 7.5+; T4 is sm75. Replace with: "AWQ is a candidate T4 path; test it. AutoAWQ is archived, and vLLM's AWQ path is unverified on T4."

4. **Correct the BPD verdict.** 10.35x is in the paper (Section 6, Table 1), not just the candidates file. It is Dream-7B MBPP, 12.69 → 131.36 TPS, A6000, 256 tokens. It is not a T4/100M number.

5. **Correct the fixed-state verdict.** 15.26 at S=4 and 6.44 at S=32 are in Table 5 (Attn column). The setting is 3B, H100, 1024 context, 64-128-token continuations, GPT-2-large PPL. It is not a 100M/T4 curve.

6. **Correct the "especially with long prefill" verdict.** Fast-dLLM §4.3 and Table 5 confirm it: DualCache grows from 9.4x at gen 256 to 27.6x at gen 1024, and 19.6x → 27.6x from 5-shot to 8-shot.

7. **Add the Fast-dLLM baseline context.** 27.6x is 0.7 tok/s → 19.3 tok/s on A100, LLaDA-8B, 8-shot, 1024-token generation. Do not present it as a multiplier on 350 tok/s.

8. **Add the dLLM-Cache hardware and wall-clock numbers.** RTX 4090, LLaDA 8B/Dream 7B; 9.1x is FLOPs, while TPS gains are 1.28-5.33x.

9. **Fix the timing contract.** `bench/backends.py` says decode excludes prefill, but `TransformersBackend` includes it. Either exclude prefill for both or report prefill separately. The LFM2.5 12-token row is mostly prefill.

10. **Fix the stale README.** "Nothing here has run on a T4 yet" is contradicted by `results_t4.yaml`. Update the caveat or remove it.

11. **Fix the spec's 297/231 comparison.** The fixed-state numbers are 3B on H100 at 256k context. The repo's 350 tok/s on 144M/T4 at 1k context is already above 297 tok/s. The spec's "meaningfully higher than both" target is not a useful floor; compare at the same context length and hardware, or drop the comparison.

12. **Replace "2026 unknown" with the verified list.** DepCap, EB-Decode, BlockBatch, CoDiLA, DFlash, DScale. All are 7-8B/A100/A6000, none T4/100M; cite them as methods, not multipliers.

13. **Soften the int4 mechanism.** bnb supports sm75; the measured regression is a kernel/config fact, not an unsupported-architecture fact. AWQ GEMM is designed to be faster than FP16 below batch 8, so "int4 may hurt at batch 1" is backend-specific.

14. **Add the model-card decode policy.** smoldlm's default is 10 steps, top-k, temperature 0.7, and greedy degenerates. The harness's 8 steps / 0.9 threshold / greedy is a custom policy, not the model's default.

15. **Add the "no AWQ" correction to the spec, not just the note.** The spec is the source of the wrong claim; the note repeats it. The spec should say "AWQ is untested on T4; AutoAWQ claims Turing support" rather than "no AWQ".

---

## 8. My independent top-5 speed experiments for a free T4

All targets are 5-run medians at batch 1, 1k prompt / 1k output unless stated. The measured reference is 349.979 tok/s from the uncached loop. All experiments must log tokens/run, sample text, and pass/fail on a quality sanity check; a speed win with garbage output is a kill.

### Experiment 1 — Use the model's existing cached decode path

- **Hypothesis:** `CustomBackend` is slow because it recomputes `prefix + block` every pass and ignores `modeling_smoldlm.py`'s KV cache. Calling the model's own cached `generate()` (or replicating its `set_cache_mode(True)` block loop) will raise tok/s materially at 1k/1k, with better sample quality because it uses temperature 0.7 instead of greedy.
- **What to run:** (a) current `CustomBackend` 32/8/0.9/greedy; (b) model-native cached loop, block 32, 10 steps, top-k, temperature 0.7; (c) same cached loop with 8 steps, to isolate cache vs policy. Same prompt, same 1024 output, 5 runs.
- **Number to hit:** cached path ≥ **525 tok/s** (1.5x over 349.979), 1024 emitted tokens, 5-run median.
- **Kill criterion:** < **402 tok/s** (1.15x) after a correct cache wiring, or the cached path produces worse sample sanity than the model's own `generate()` at the same settings (e.g., more repetition, mask artifacts). If the cache changes greedy outputs by >1%, treat it as a correctness bug, not a speed win.
- **Why it is independent:** the note says "implement a prefix cache"; the cache already exists. This is a wiring + measurement experiment, not a research project.

### Experiment 2 — Prompt-length scaling and cache correctness

- **Hypothesis:** the uncached loop's cost grows with prompt length because every pass re-reads the prefix; the cached path should be roughly flat. The current 1k/1k number hides this, and the spec's 32k context makes it decisive.
- **What to run:** cached vs uncached at prompt lengths 128 / 512 / 1024 / 2048, output 1024, same decode policy, 5 runs each. Log tok/s and greedy token match between cached and uncached (same seed/policy).
- **Number to hit:** at 2048 prompt, cached ≥ **1.5x** uncached; cached tok/s drop from 128 to 2048 ≤ **10%**; greedy output match ≥ **99%**.
- **Kill criterion:** < **1.2x** at 2048, or output mismatch > **1%** (cache/position bug). If the cached path is not flat, the cache is not being used correctly.
- **Why it is independent:** it validates Experiment 1 and catches the position-offset / block-causal cache bugs that a single 1k run would miss.

### Experiment 3 — Decode-policy sweep on the native cached path

- **Hypothesis:** the model's default 10 steps / top-k / temperature 0.7 is not the speed optimum. Fewer steps (4-8) with temperature 0.7 may hold sample quality and raise tok/s; the harness's 0.9 threshold + greedy is both slower and lower quality.
- **What to run:** grid steps {4, 8, 10, 16} × block {16, 32} × commit {top-k (model default), confidence 0.9} × temperature {0, 0.7}, on the cached path, fixed prompt set, 5 runs per cell. Log tok/s, tokens/pass, passes/block, and a sample-sanity score (repetition rate, mask-token rate).
- **Number to hit:** best config ≥ **455 tok/s** (1.3x over 349.979) with sample sanity no worse than the model's own 10-step generate.
- **Kill criterion:** no config ≥ **402 tok/s** (1.15x) without obvious degeneration (repetition, mask artifacts, or a sample-sanity score worse than baseline).
- **Why it is independent:** it corrects the note's sweep, which runs on the uncached loop and uses greedy only.

### Experiment 4 — Quantization A/B on SmolLM2-135M: fp16 vs bnb int4 vs AWQ int4

- **Hypothesis:** the Qwen bnb int4 regression is backend-specific, not a universal "int4 is slower at batch 1". AWQ GEMM is designed for batch <8 and supports Turing, so it may beat fp16 on a 135M model at batch 1; bnb may not.
- **What to run:** `HuggingFaceTB/SmolLM2-135M-Instruct`, batch 1, 1k/1k, 5 runs, 50 HumanEval, three arms: fp16, bnb int4 (`load_in_4bit=True`, compute fp16), AWQ int4 (AutoAWQ GEMM if it loads on sm75). Log tok/s, pass@1, and load/fallback status.
- **Number to hit:** AWQ int4 ≥ **1.2x** over fp16 (≥ **28.7 tok/s** if fp16 is 23.9), pass@1 within **2 points**; bnb int4 ≤ **1.05x** over fp16 (reproduce the regression on a second model).
- **Kill criterion:** AWQ fails to load on sm75, or < **1.1x** over fp16; then report fp16 as the T4 default and stop pursuing AWQ for the 100M model. If bnb beats fp16 on SmolLM2, the Qwen regression is model/config-specific, not a T4-wide fact.
- **Why it is independent:** it tests the spec's wrong "no AWQ" claim and separates "bnb kernel" from "4-bit is slow" with a real T4 A/B.

### Experiment 5 — Block early stop (min-confidence 0.97) on the cached path

- **Hypothesis:** many blocks finish before the fixed 8-10 steps; a block-level min-confidence score above 0.97 can fill the remaining slots and skip rounds, raising tok/s without quality loss.
- **What to run:** cached path, best config from Experiment 3, with and without early stop. Log mean passes/block, rounds saved, tok/s, and sample sanity. Use the spec's rule: block score = min confidence over remaining masked slots; threshold 0.97.
- **Number to hit:** ≥ **10%** mean pass reduction and ≥ **1.1x** tok/s over the Experiment 3 winner, same prompt set and emitted length.
- **Kill criterion:** < **3%** pass reduction or < **1.02x** tok/s, or early stop commits a visibly worse block (mask artifacts, broken code) in the sanity check.
- **Why it is independent:** the note lists early stop as a spec knob; this runs it on the cached path, after the cache and policy are fixed, so the pass-count effect is measurable rather than confounded by the uncached loop.

**Bonus (cheap, do alongside Experiment 1):** log which SDPA backend the T4 uses (`flash_sdp_enabled`, `mem_efficient_sdp_enabled`, `math_sdp_enabled`) and try `torch.compile(mode="reduce-overhead")` on `score_fn`. If the T4 is on the math backend, a fused kernel may give ≥1.15x with identical outputs. Kill if compile fails/falls back or <1.05x. This is a kernel check, not a substitute for the cache/policy wins.