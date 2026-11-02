# EdgeFast-Swarm — virtual big model from fast small ones (v1, idea-stage)

Drafted 2026-11-01 (user idea, "let me cook" session). This is a v2-direction spec: it depends on D1 (see `edgefast-diff-100m.md`) existing and being fast. Nothing here blocks or changes the v1 plan.

## The idea

We can't train one big model, but we have effectively unlimited free compute (many accounts). So we don't build one big brain — we build a **swarm**: many instances of the same fast 100M architecture, each trained for one job, orchestrated by a harness into a single "virtual big model."

Think MoE with the welds replaced by glue:

| | Real MoE | The swarm |
|---|---|---|
| Routing | per token, inside one model | per task, by the harness |
| Experts | same shape, co-trained, gradients flow | separate models, trained independently |
| Adding/fixing an expert | retrain the whole stack | swap one file |
| Cost of a routing decision | a matrix multiply | a model call + handoff |
| Hardware | one giant machine | N free accounts, one per specialist |

Closest published cousin: Mixture-of-Agents (layered small models combined beat bigger ones on some benchmarks) — but MoA is a quality play; this is a **speed** play, which is the unclaimed twist.

## Why this architecture is what makes it viable

The swarm's whole tax is per-hop latency. Every "summon the verifier" is a round trip, which is why nobody routing a 70B dense model can afford 20 specialists per turn. Our spec is a per-hop latency killer:

- GDN hybrid → every expert instance is cheap to run.
- Block diffusion → short bursty outputs (a verdict, a patch, a rank) come out in parallel instead of token-by-token. Specialist work is *exactly* this: short, frequent, small.
- The harness carries the long context (files, tool output, conversation); specialists receive small focused slices and make local calls. Small models are good at exactly that.

Slow models are forced to be monoliths because every invocation is expensive. Fast models can afford to be a swarm.

## The specialist roster

Strong fits (verifiable, local, small-input tasks):

- **Verifier** — judge a hunk/diff: broken or fine. Judging is easier than doing.
- **Patcher** — produce the fix for one focused slice.
- **Diff-ranker** — rank N candidate fixes.
- **Router** — decide which specialist (if any) the current turn needs.

Parked (wrong size class, revisit only if compute or bigger checkpoints appear):

- **Design** — needs vision + taste; a 100M can't hold it.
- **Research** — mostly retrieval; that's the harness's job, not a model's.

## Training: the bug-injection loop

The engine. Take real, good codebases. Inject synthetic issues. Run the swarm-in-a-harness to fix them. Diff the result against `main`.

The payoff that makes this better than generic agentic RL: **we know exactly where we injected the bug and what the original looked like**, so every layer of the system gets a free ground-truth reward from one trajectory:

- **Whole-system reward:** final diff vs `main`.
- **Verifier reward:** its broken/fine call on a hunk vs the known injected location.
- **Patcher reward:** its diff vs the original lines.
- **Router reward:** did summoning the specialist help the trajectory?

Credit assignment — the nightmare of multi-model RL — is solved by construction.

**Order matters:** train components on local rewards first (dense, free), then polish the swarm end-to-end (diff vs `main` as the trajectory reward). End-to-end-only from scratch is noisy and slow.

**Latency discipline:** budget specialist calls per turn like a token budget. If success starts meaning "summon 6 experts per turn," we've rebuilt a slow model out of fast parts.

**Account mapping:** unlimited accounts map one-to-one onto specialists — each account trains one specialist's loop in parallel, same tokenizer, same training loop, different data.

## Relation to the 100M spec (v1)

- v1 stays a single model, single-shot, no tool calling — unchanged. The swarm is v2 and presumes D1's speed thesis is proven first.
- The spec's deferred knobs anticipate this: the L1 verification skill is the swarm's Verifier in embryo; adaptive steps fused with verification becomes a router decision.
- Same architecture, stamped out N times with different data. No new model design needed.

## Open questions (before any of this is real)

- Minimum viable swarm: probably router + verifier + backbone. Does 3 experts already beat the backbone alone on injected-bug fix rate at acceptable added latency?
- Harness spec: what exactly gets passed to a specialist (file slice? hunk + surrounding lines?), and how its output is wired back. This is most of the engineering.
- Real-codebase bug injection: which repos, which injection methods (mutation-style operator swaps? LLM-authored issues?), how to avoid injection artifacts the model learns to pattern-match instead of actually debugging.
- Block diffusion + multi-turn agentic loops: the 100M spec flags interactive RL as v2 for a reason; the swarm is the first thing that actually needs it.
