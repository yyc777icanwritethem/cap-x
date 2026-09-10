# Adaptive CaP-Bench failure-analysis pilot

## Goal

This pilot is not intended to reproduce the full CaP-Bench leaderboard. It is designed to find informative failure cases for studying why LLM-generated robot code fails, what logic is missing relative to a correct/oracle solution, and whether non-trivial control-flow structures such as branches, loops, state checks, verification, retries, and recovery are required.

## Initial task set

We use four Robosuite tasks chosen to cover distinct sources of difficulty:

| Task | Main reason to include it |
| --- | --- |
| Cube Stack | placement accuracy, sequential manipulation, verification/retry |
| Nut Assembly | geometry, rigid transforms, precision insertion, conditional reasoning |
| Spill Wipe | coverage, loops/nested loops, stopping conditions |
| Two-arm Handover | long-horizon sequential coordination, state transfer, collision-aware recovery |

Cube Lift and Cube Restack are deferred. Two-arm Lift is also deferred initially; it can be added later if simultaneous coordination becomes an important axis.

## Initial tier set

Run six tiers:

- S2: single-turn, visual/non-privileged high-level API
- S3: single-turn, reduced/low-level API
- M1: multi-turn + stdout/stderr feedback
- M2: M1 + raw RGB feedback to the coding model
- M3: M1 + textual visual-differencing feedback
- M4: M3 + reduced/low-level API

S1 and S4 are deferred because privileged perception and example-removal are not the main questions in the first failure-analysis pass.

## Seed policy

Start with seed 1 only:

4 tasks x 6 tiers x 1 seed = 24 Qwen trials.

Do not automatically expand to a fixed multi-seed matrix. Add seed 2 only when one of the following is true:

1. a failure has a potentially interesting structural/root-cause explanation and we want to test whether it repeats;
2. a result could plausibly be an initialization/physics accident rather than a model/program failure;
3. we need a matched success/failure comparison for the same task and tier.

Use seed 3 only when seeds 1 and 2 disagree on a key claim or when a particularly important finding needs one more replication.

## Model strategy

Primary screening model:

`openrouter/qwen/qwen3.8-27b`

This is the cost-efficient model used to discover candidate failures. The pilot should cap generation at 8192 tokens unless a specific case demonstrates that the cap is truncating valid programs.

For M2, the coding model must actually support image input and must pass the CaP-X VLM gate. Do not run M2 until `audit_qwen_groq_vision.py` passes and the model is registered in `VLM_MODELS` if needed.

For M3/M4, prefer a strong VLM for visual differencing so that a weak visual captioner does not become the main source of failure. The current planned default is `google/gemini-3.1-pro-preview`, subject to a one-request connectivity/image-input smoke test through the configured proxy before execution.

## Execution order

Do not launch all 24 immediately.

1. Vision preflight: Qwen direct-image smoke for M2.
2. VDM preflight: one image request with the chosen visual-differencing model.
3. Cube Stack six-tier smoke: S2, S3, M1, M2, M3, M4 at seed 1.
4. If all six paths are technically valid, run the remaining 18 seed-1 conditions.
5. Analyze failures before deciding which seeds/tasks to add.

## Failure-analysis record

For each completed trial retain:

- task / tier / seed / model
- success and reward
- generated program for every turn
- stdout / stderr
- visual or VDM feedback used on each turn
- video
- number of code blocks / regenerations
- oracle/reference program when available
- program-structure metrics: LOC, functions, branches, loops, nesting depth, API calls
- explicit state checks, verification, retries, and recovery logic
- infrastructure/provider failures separately from physical/program failures

The per-case reasoning template is:

Task
-> generated code
-> actual execution behavior
-> failure symptom
-> root cause
-> what is missing relative to correct/oracle logic?
-> control-flow complexity: branch / loop / retry / state check / verification / recovery

Do not assign a root cause solely from the final reward. Use generated code, logs, state/visual evidence, and oracle behavior together.

## Frontier-model escalation

After Qwen screening, do not rerun every condition with an expensive model. Select up to roughly 10-15 representative Qwen failures, covering different tasks/root-cause categories, plus about 5 Qwen-success controls.

Rerun those cases with a stronger current GPT or Claude model. This separates:

- capability-sensitive failures: Qwen fails, frontier model succeeds;
- persistent failures: both Qwen and frontier model fail.

Persistent failures are the highest-priority cases for the final analysis, especially if they repeatedly involve missing verification, recovery, coordinate reasoning, or non-trivial control flow.
