# Verification of `research/hillclimb-speed-iceman.md`

Audit date: 2026-11-11
Auditor: iceman-free (self-audit, review mode). I treat the original note as a claim to audit, not a source. I do not defend it.

Method: I read the original note and repo files (`results_t4.yaml`, `bench/backends.py`, `README.md`, `bench/BENCHMARK_WEEK.md`, `spec/edgefast-diff-100m.md`, `/home/kriday/OpenBot/Shared/fastest-llm-4-candidates.md`). I fetched arXiv abstract pages for every paper cited in the note: 2505.22618, 2509.26328, 2506.06295, 2610.26040, 2503.09573. Web search failed in this session (HTTP 402), so I could not survey 2026 literature beyond fetched/cited sources. I did not guess any unreachable content.

## Repo claims (checked against files in this repo)

| Claim in note | Source checked | Verdict | Evidence |
|---|---|---|---|
| Baseline numbers: smoldlm-144m 350.0, qwen 18.6, qwen int4 14.7, smollm2 23.9 (307 tok/run), lfm2.5 41.4 (12 tok/run), humaneval pass@1 0.06/0.02/0.0/0.1 as tabled | `results_t4.yaml` | CONFIRMED | `results_t4.yaml`: `smoldlm-144m speed.median_tps 349.979`, `qwen3.5-0.8b 18.562`, `qwen3.5-0.8b-int4 14.744`, `smollm2-135m 23.912` with `tokens_per_run [307,...]`, `lfm2.5-350m 41.437` with `tokens_per_run [12,...]`. HumanEval: qwen 3/50 pass@1 0.06, qwen-int4 1/50 0.02, smollm2 0/50 0.0, lfm2.5 5/50 0.1. Notes match lean env and degeneracy caveats. |
| Lean env missing flash-linear-attention, causal-conv1d, dLLM kernels; Qwen GDN may have used fallback kernels | `results_t4.yaml` notes | CONFIRMED | `results_t4.yaml` notes: "Lean T4 setup on Colab (edgefast-bench-3, USW4A2) without flash-linear-attention, causal-conv1d, or dLLM. Qwen3.5 GDN may have used fallback kernels..." |
| Prototype decode loop: block_size 32, 8 steps, 0.9 confidence unmasking, top-1 commit when pass clears nothing, forced fill on last pass. Missing 0.97 block early stop, prompt cache, block overlap. Timing includes full forward over `prefix + block` each pass. | `bench/backends.py`, `README.md` | CONFIRMED | `CustomBackend.__init__` takes `block_size, steps, confidence` (defaults used in `load_smoldlm` call sites/config context give 32/8/0.9 in note; `load_smoldlm` signature requires args). `generate`: unmask slots with `conf[i] >= self.confidence`; if none, `fill = [max(open_slots, key=lambda i: float(conf[i]))]` ("commit the single most confident slot"); last pass fills all open slots (`if step == self.steps - 1 ... open_slots`). Builds `seq = torch.tensor([out_ids + block], device=self.device)` and runs `logits = self.score_fn(seq)` every pass. No block early stop, no prompt cache, no block overlap in `CustomBackend`. `README.md` states same caveats. |
| `TransformersBackend` times full `model.generate` call including prompt prefill; int4 uses `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)` | `bench/backends.py` | CONFIRMED | `generate` does `t0 = time.perf_counter()` before `self.model.generate(...)` and returns `decode_seconds = time.perf_counter() - t0` (full call). Int4 path sets `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`. |
| Spec says T4 is sm75, no bf16, no AWQ on Turing; quantized tier fp16 + bnb int4 or GGUF Q4_K_M; D1 2x bandwidth claim for int4 is ceiling; fallback to fp16 if QAT degrades | `spec/edgefast-diff-100m.md` | CONFIRMED | Spec: "Colab free T4 (16 GB, sm75, ~320 GB/s)", "T4 has no bf16 and no AWQ support. Full-precision tier is fp16; quantized tier is GPTQ int4 or llama.cpp Q4_K_M." "At 100M the quality cost of int4 hurts more... if QAT degrades the pilot badly, fall back to fp16..." Int4 bandwidth ceiling framing is consistent with spec's bandwidth math. |
| Spec defines block early stop 0.97, block score as min confidence over remaining slots, adjacent work Prophet (not paper-measured in repo) | `spec/edgefast-diff-100m.md` | CONFIRMED | Spec: "Above 0.97, fill every remaining slot..." "Block score defaults to the MIN confidence across remaining masked slots..." "(Adjacent published work: Prophet early-stopping; block-score definition is ours.)" |
| "Fixed-state paper" generative PPL 15.3 at 4 steps vs 6.4 at 32 (as cited in spec/note) | `spec/edgefast-diff-100m.md` (note cites "per spec") | UNCONFIRMED (primary not fetched) | The note attributes this to "fixed-state paper: generative PPL 15.3 at 4 steps vs 6.4 at 32 (paper says, per spec)". Spec references "fixed-state paper" (arXiv 2609.11998 in spec context) but I did not fetch that paper's primary text in this session. I cannot quote the sentence from the fixed-state paper itself. Spec-level claim is present; paper-level confirmation not done. |
| `fastest-llm-4-candidates.md` claims BPD "up to 10.35x throughput vs vanilla diffusion decoding" (arXiv 2610.26040 cited there) | `/home/kriday/OpenBot/Shared/fastest-llm-4-candidates.md` | MISLEADING / UNCONFIRMED from abstract (see paper audit) | The file states "BPD pivot decoding: balance-aware pivot selection, up to 10.35x throughput vs vanilla diffusion decoding (arXiv 2610.26040)." I fetched the arXiv abstract for 2610.26040; the abstract does not state "10.35x". I did not fetch full PDF/HTML body beyond abstract text provided. Magnitude is not confirmed by the fetched abstract. |

## Paper claims (verified against fetched arXiv abstracts)

### 1. Fast-dLLM (arXiv:2505.22618)

Fetched: https://arxiv.org/abs/2505.22618 (v3, 3 Jul 2025). Authors: Wu et al.

| Claim in note | Verdict | Quoted evidence from fetched source | Notes |
|---|---|---|---|
| Reports up to 27.6x throughput over vanilla dLLM baselines in their settings with KV cache + confidence-aware parallel decoding (training-free, LLaDA/Dream). Not T4 or 100M. | CONFIRMED (with hardware/scale caveat correct) | Abstract: "Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to **27.6$\times$ throughput** improvement with minimal accuracy loss..." and "introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models..." and "propose a confidence-aware parallel decoding strategy..." | Abstract states throughput improvement and models (LLaDA, Dream) but does not state GPU type, batch size, or model-parameter size in the abstract text fetched. The note correctly says "not a T4 or 100M number" and treats as paper setting. |
| Block-wise approximate KV cache exploits high similarity across adjacent steps; large end-to-end gains especially with long prefill | CONFIRMED (abstract-level) | Abstract: "a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop." | "High similarity across adjacent steps" and "especially with long prefill" are plausible interpretations but the specific long-prefill claim is not explicitly quoted in abstract; mechanism/caching reuse is confirmed. Treat the detailed prefill emphasis as UNCONFIRMED from abstract alone. |

### 2. Fast-dLLM v2 (arXiv:2509.26328)

Fetched: https://arxiv.org/abs/2509.26328 (v1, 30 Sep 2025). Authors: Wu et al.

| Claim in note | Verdict | Quoted evidence from fetched source | Notes |
|---|---|---|---|
| Block diffusion dLLM adapted from pretrained AR models with ~1B tokens fine-tuning; hierarchical caching (block-level cache across blocks, sub-block cache inside partially decoded blocks) + parallel decoding; up to 2.5x speedup over standard AR decoding without compromising generation quality; evaluated across diverse benchmarks | CONFIRMED | Abstract: "Fast-dLLM v2, a carefully designed block diffusion language model (dLLM) that efficiently adapts pretrained AR models into dLLMs for parallel text generation, requiring only approximately 1B tokens of fine-tuning." "a hierarchical caching mechanism: a block-level cache that stores historical context representations across blocks, and a sub-block cache that enables efficient parallel generation within partially decoded blocks. Coupled with our parallel decoding pipeline, Fast-dLLM v2 achieves up to 2.5x speedup over standard AR decoding without compromising generation quality. Extensive experiments across diverse benchmarks..." | "500x reduction vs Dream (580B tokens)" is also in abstract but note did not claim it; not audited as a claim. |
| Recipe converts pretrained AR models and reports results in that context; transfer to tiny from-scratch 100M block-diffusion model unconfirmed. 2.5x over AR not same baseline as 350 tok/s for 144M block diffusion. | CONFIRMED as reasoning / MISLEADING risk correctly flagged | Supported by abstract ("adapts pretrained AR models..."). | Note's caution is sound and matches what abstract says. |

### 3. dLLM-Cache (arXiv:2506.06295)

Fetched: https://arxiv.org/abs/2506.06295 (v3, 6 Jun 2026). Authors: Liu et al.

| Claim in note | Verdict | Quoted evidence from fetched source | Notes |
|---|---|---|---|
| Training-free adaptive caching for dLLMs (long-interval prompt caching + partial response updates guided by feature similarity), exploits static prompt and mostly stable tokens across adjacent denoising steps; up to 9.1x FLOPs reduction on LongBench-HotpotQA; maintains competitive output quality; tested on LLaDA 8B and Dream 7B. | CONFIRMED | Abstract: "dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity." "Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1x FLOPs reduction on LongBench-HotpotQA while maintaining competitive output quality." | Correct. |
| "ICML 2026 acceptance noted in fetched arXiv page" (as stated in note section 5) | CONFIRMED | Fetched page shows: "Comments: Accepted by ICML 2026". |  |
| 7B evidence only / FLOPs reduction != tok/s on T4 | CONFIRMED as limitation (reasoning) | Abstract reports LLaDA 8B and Dream 7B and "FLOPs reduction"; no tok/s or T4 hardware stated. | Note's caution is correct. |

### 4. BPD (arXiv:2610.26040)

Fetched: https://arxiv.org/abs/2610.26040 (v1, 19 Oct 2026). Authors: Lee et al.

| Claim in note | Verdict | Quoted evidence from fetched source | Notes |
|---|---|---|---|
| Balance-Aware Pivot Decoding selects central pivots within contiguous masked intervals and decodes selected positions in parallel; balanced pivots give logarithmic decoding depth in interval length; combines confidence-based prefilling with confidence-thresholded center selection; evaluated on LLaDA and Dream across 5 math/code/instruction benchmarks. | CONFIRMED | Abstract: "we propose Balance-Aware Pivot Decoding (BPD), which favors central pivots within contiguous masked intervals and decodes the selected positions in parallel. Balanced pivots yield logarithmic decoding depth in the interval length. Our practical method accounts for model uncertainty by combining confidence-based prefilling with confidence-thresholded center selection. We evaluate the resulting quality-efficiency tradeoff on LLaDA and Dream across five mathematical reasoning, code generation, and instruction-following benchmarks." | Correct. |
| "fastest-llm-4-candidates.md ... claims up to 10.35x throughput vs vanilla diffusion decoding (arXiv 2610.26040 cited there); that number is not in the fetched abstract, so I cannot confirm magnitude from abstract alone (unconfirmed from primary text I fetched)." | CONFIRMED (note's own uncertainty is correct) | Abstract fetched contains no "10.35x" figure. | The 10.35x claim in `fastest-llm-4-candidates.md` is UNCONFIRMED by abstract; I could not fetch full paper body in this pass. Treat 10.35x as UNCONFIRMED, not verified. |
| "2026 alternative..." / "No small-model (100M) evidence in abstract" | CONFIRMED | Submission date on page: 19 Oct 2026; abstract lists only LLaDA and Dream. |  |

### 5. BD3-LM (arXiv:2503.09573)

Fetched: https://arxiv.org/abs/2503.09573 (v3, 17 May 2025). Authors: Arriola et al.

| Claim in note | Verdict | Quoted evidence from fetched source | Notes |
|---|---|---|---|
| BD3-LM / block-diffusion in the BD3-LM / fixed-state style context (note cites paper in sources/decoding context) | CONFIRMED as cited paper exists and matches block diffusion framing | Abstract: "we introduce a class of block diffusion language models that interpolate between discrete denoising diffusion and autoregressive models. Block diffusion overcomes key limitations..." "supporting flexible-length generation and improving inference efficiency with KV caching and parallel token sampling." "ICLR 2025 Oral." | Note does not make a specific numeric claim from BD3-LM except naming it; naming/citation is correct. |

## Other claims and inferences in the note

| Claim in note | Verdict | Notes |
|---|---|---|
| "At batch 1 decode... weight-only int4 saves memory bandwidth on weight reads but adds dequantization/packing overhead..." and "common small-batch failure mode..." but "no T4+bitsandbytes benchmark source fetched in this session to pin the exact crossover (unconfirmed from primary benchmark)" | MISLEADING if read as proven cause; partially reasonable mechanism but UNCONFIRMED | The measured regression (18.6 -> 14.7) is CONFIRMED. The causal explanation is hypothesis/mechanism; note labels parts unconfirmed, which is good, but still presents a confident narrative. Keep as hypothesis, not fact. |
| "T4 is Turing (sm75)... bitsandbytes 4-bit kernels and their fast paths vary by architecture; a suboptimal path on sm75 would explain a regression. Unconfirmed without a bitsandbytes kernel-support matrix citation fetched here." | UNCONFIRMED | No primary bnb/sm75 kernel matrix fetched. |
| "For a 100M block-diffusion model, weights are already tiny (~0.06 GB at int4 per spec math). Dequant overhead has less to amortize..." | CONFIRMED as arithmetic consistent with spec, but performance implication UNCONFIRMED | Spec gives 100M int4 ~0.06 GB context; the speed implication is inference, not measured. |
| "2x bandwidth claim for int4 ... is a ceiling from bytes/pass, not a guaranteed tok/s win. On T4 batch 1 it can be negative." | CONFIRMED (measured negative on Qwen int4) | Matches `results_t4.yaml`. |
| "Big lever because current loop re-scores `prefix+block` every pass. Must match model's attention type..." | CONFIRMED (implementation fact) / UNCONFIRMED (magnitude) | Rescoring is real per `backends.py`. Tok/s gain magnitude not measured. |
| "Block overlap... on single-stream T4 likely limited. Unconfirmed." | UNCONFIRMED | Reasonable but not measured. |
| "Web search failed in this session (HTTP 402), so 2026 coverage beyond fetched/cited sources is incomplete..." (sections 5 and end) | CONFIRMED | Matches this audit run too. |

## Weak spots and missing coverage (speed field)

I cannot add verified 2026 claims from web search because search failed (HTTP 402). I can only state gaps with source status.

| Area | Status | What is missing / weak in original note |
|---|---|---|
| Newer 2026 decode methods beyond BPD | UNCONFIRMED / NOT SURVEYED | Note only discusses BPD as 2026 method; no verified survey of other 2026 methods (e.g. newer schedulers/caches/spec variants) was possible without search. Any addition here would be guessing, so I do not add one. |
| Small-batch quantization on Turing/sm75 (bnb/GGUF/TensorRT paths, kernel support, measured decode tok/s) | WEAK / UNCONFIRMED | Note has one measured point (Qwen bnb int4 slower) but no primary bnb/Turing support matrix or GGUF T4 benchmarks. Missing: GGUF Q4_K_M on llama.cpp for comparable small models, exllama/vllm Turing constraints (if relevant), and whether dequant overhead differs by backend. I cannot fetch primary benchmarks now. |
| Block-diffusion speedups 2025-2026 beyond fetched set | WEAK / PARTLY COVERED | Fast-dLLM v1/v2 and dLLM-Cache covered by abstracts. Missing verified numbers for other block-diffusion systems, few-step distillation methods with speed/quality tradeoffs on small models, and T4-scale measurements. "Few-step distillation" is mentioned in spec/candidates context but not verified as speed numbers in this note. |
| FLOPs vs wall-clock | PARTLY ADDRESSED | Note correctly notes FLOPs reduction != tok/s for dLLM-Cache, but similar care is needed for all paper multipliers (different hardware, batch, prefill length, model size). |
| Spec-decode / diffusion-drafter options on T4 for 0.8B class | NOT COVERED in this note | `BENCHMARK_WEEK.md` and candidates mention DFlash2/EAGLE-3 friction; hillclimb note focuses on diffusion decode knobs but omits AR spec-decoding alternatives as speed experiments for the AR baseline path except int4 caution. |

## Claims in the original note that do NOT survive verification (as stated)

1. **"up to 10.35x throughput vs vanilla diffusion decoding" for BPD (arXiv 2610.26040), as repeated from `fastest-llm-4-candidates.md` context in note's reasoning**: DOES NOT SURVIVE as verified fact. Verdict UNCONFIRMED. Fetched abstract of 2610.26040 does not contain this number. (The note itself flagged this as unconfirmed, which is correct, but the number must not be treated as verified.)
2. **Fast-dLLM "large end-to-end gains especially with long prefill" stated as a specific emphasis in note text**: DOES NOT FULLY SURVIVE as quoted/confirmed detail from abstract. Abstract confirms block-wise approximate KV cache and up to 27.6x throughput, but not the "especially with long prefill" phrasing as a verified quote in fetched text. Verdict UNCONFIRMED (detail).
3. **Causal explanation for int4 slowdown on T4 (dequant/packing overhead, sm75 kernel path, memory-bound crossover)**: DOES NOT SURVIVE as proven cause. Measured slowdown is CONFIRMED; mechanism is UNCONFIRMED hypothesis.
4. **"Fixed-state paper: generative PPL 15.3 at 4 steps vs 6.4 at 32"**: DOES NOT SURVIVE as primary-source-verified in this audit. Verdict UNCONFIRMED here (spec cites it; primary paper text not fetched/quoted by me).
5. **Any implied transfer of Fast-dLLM 27.6x, Fast-dLLM v2 2.5x, dLLM-Cache 9.1x, or BPD gains to 100M smoldlm on T4 batch 1**: DOES NOT SURVIVE. Note often warns, but ranked expectations and "paper motivation" can still read too strong. Verdict MISLEADING if taken as transferable multipliers (correct to treat as non-transferable).
6. **"2026 methods that beat confidence-based parallel unmasking for small models" (beyond BPD suggestion)**: DOES NOT SURVIVE as known/verified. Verdict UNCONFIRMED/UNKNOWN due to search failure and lack of small-model evidence in fetched BPD abstract.

## Corrected top-5 experiment list

Corrections applied: remove unverified multipliers as targets where not supported, keep targets tied to measured 350 tok/s baseline and repo reality, treat int4/kernel work as measurement-first, and require correctness/quality checks. All speedups here are hypotheses to measure on T4, not paper-transfer guarantees.

1. **Prefix/prefix-state reuse in `CustomBackend` (correctness-first cache experiment)**
   - Hypothesis: avoiding redundant work from rescoring full `prefix+block` every pass (`bench/backends.py`) can improve tok/s for `smoldlm-144m` at 1k/1k.
   - Target: >= 1.15x median tok/s over 349.979 (>= ~402 tok/s) in 5-run median, same block/steps/confidence and 1024 emitted tokens, with sample text no worse by quick sanity check.
   - Why corrected: original >=450 tok/s was an unverified stretch; Fast-dLLM/dLLM-Cache motivate caching but their multipliers are on different models/hardware (Fast-dLLM: "up to 27.6$\times$ throughput" on LLaDA/Dream; dLLM-Cache: "up to 9.1x FLOPs reduction" on LLaDA 8B/Dream 7B). Not transferable.
   - Kill: < 1.05x with correct outputs, implementation too invasive/risky, or quality collapse.

2. **Decode-policy sweep on measured baseline (steps, block size, confidence)**
   - Hypothesis: fixed 8 steps / block 32 / conf 0.9 is not optimal for speed-quality point.
   - Target: best config >= 1.2x median tok/s (>= ~420 tok/s) with comparable or better sample sanity than baseline; log tokens/pass and passes/block.
   - Why corrected: keep simple, directly measurable. Fast-dLLM abstract supports confidence-aware parallel decoding as useful mechanism, but not this exact target on T4/144M.
   - Kill: no config >= 1.1x without clear degeneration.

3. **Block early stop (spec rule: min confidence over remaining slots, threshold 0.97)**
   - Hypothesis: some blocks finish early; fewer passes/block can raise tok/s.
   - Target: >= 10% mean pass reduction and >= 1.08x tok/s over experiment 2 winner, same prompt set and emitted length.
   - Why corrected: spec defines rule but gives no measured magnitude; keep modest and require pass-count evidence.
   - Kill: < 3% pass reduction or < 1.02x tok/s.

4. **Unmask selector A/B: confidence-only vs BPD-style pivot selection (prototype behind flag)**
   - Hypothesis: BPD's central-pivot idea may change rounds/quality tradeoff.
   - Target: >= 1.05x tok/s at similar quality, or clearly better quality at similar tok/s, vs best of 2/3.
   - Why corrected: BPD abstract confirms method and evaluation on LLaDA/Dream but gives no throughput number in fetched abstract and no 100M/T4 evidence. 10.35x claim is UNCONFIRMED. Treat as exploratory.
   - Kill: no win after one fair implementation pass.

5. **Measurement hygiene and T4 backend reality check (before claiming kernel/quant wins)**
   - Hypothesis: comparability and backend choice dominate reported speed more than speculative kernel wins on sm75.
   - Target: (a) fix length comparability for smollm2/lfm2.5 (307/12 tokens in `results_t4.yaml`) and rerun; (b) for any int4 path, require >= 1.05x over same-model fp16 on same T4 harness to claim speedup (Qwen int4 is currently 14.744 < 18.562, CONFIRMED regression); (c) record install/fallback status for each kernel.
   - Why corrected: original ranked kernel/env low but mixed concerns; this makes the "no fake speedup" gate explicit and grounded in measured repo data.
   - Kill: a backend/kernal cannot run or falls back on T4, or fails to beat fp16/clean baseline by >= 1.05x.

## Final note

This audit confirms the core measured repo facts (results numbers, backend behavior, lean-env caveats) and confirms the existence/main claims of the cited papers at abstract level where quoted. It does not confirm paper speed multipliers on T4/100M, does not confirm BPD's 10.35x figure from fetched abstract, does not verify the fixed-state paper PPL numbers from primary text in this pass, and cannot fill 2026 gaps because web search failed (HTTP 402).
