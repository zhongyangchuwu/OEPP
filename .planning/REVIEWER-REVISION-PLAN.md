# Reviewer Revision Execution Plan

**Updated:** 2026-07-30
**Consumer:** follow-on experiment execution and paper revision.

## Scope and priority

| Priority | Track | Reviewer need | Deliverable and acceptance gate |
|---|---|---|---|
| S | Qwen3.5 hosted MLLM | R1/R2/R3 request recent MLLM evidence | SiliconFlow `Qwen/Qwen3.5-397B-A17B` under the audited Table V `T=4`, 3+3-image protocol. A 2 Base + 2 Novel pilot must show image transport, exact returned model, parser behavior, and zero API failures before an approved 855 Base + 1,297 Novel full run. Report `paper_compatible` and `strict` separately. |
| A | P3IV and KEPP baselines | R1 asks stronger procedure-planning baselines | Independently audit the public P3IV and KEPP repositories, map each model's inputs and supervision to OEPP without changing the OEPP test protocol, then implement only reproducible, validation-selected Base / Novel evaluations. If a method cannot map without task-defining changes, record the exact blocker rather than a substitute method. |
| B | Alternate OEPP split | R1 asks reliability beyond 43 events | Recover the pre-existing alternate event split, validate it against the current JSON schema and transferability rule, freeze its provenance and action-pool derivation, then rerun the frozen final baseline matrix: MLP, Transformer, PDPP, P3IV, and KEPP. Do not report an alternate split until every included baseline has the same split definition and evaluation protocol. |

## Dependency order

```text
Qwen3.5 pilot → Qwen3.5 full audit
P3IV / KEPP feasibility → frozen baseline matrix
alternate-split recovery + validation → rerun frozen matrix
```

P3IV and KEPP feasibility may proceed while Qwen3.5 runs. Alternate-split full experiments wait until the baseline matrix is frozen; the server being unavailable blocks only server-bound recovery/training, not the current hosted Qwen3.5 experiment.

## Invariants

- Preserve Table V Base and Novel as separate denominators; never mix prompt versions, candidate orders, providers, scoring modes, or horizons.
- API keys remain environment-only. Checked-in API configs stay `api_enabled: false` and `max_calls: 0`; every run above ten calls requires a private approval file.
- API, parser, candidate-pool, and missing-prediction failures remain in the denominator and receive zero under `strict`; no semantic repair or omission.
- Select learned-model checkpoints on validation only: max SR, then max Acc, then min MSE, then earlier epoch. Test Base / Novel never select a checkpoint.
- Alternate-split evaluation must use the original-frame/feature provenance and a versioned split manifest; it cannot be assembled from model outputs or manual event reassignment.

## Immediate execution record

The account-visible SiliconFlow models query on 2026-07-30 returned six Qwen3.5 IDs. This track selects the largest available ID, `Qwen/Qwen3.5-397B-A17B`; the 2+2 pilot, rather than model-list metadata, is the multimodal compatibility gate.

## Qwen3.5 execution outcome

The non-thinking 2 Base + 2 Novel pilot satisfied the transport/format gate. The approved sequential full run completed on 2026-07-30: Base 855 / 855 API success and Novel 1,294 / 1,297 API success. The three Novel HTTP 503 failures and all candidate-pool failures remain in the scoring denominators. `paper_compatible` is Base `2.22/29.04/36.29` and Novel `7.71/35.87/59.80`; `strict` is Base `2.22/28.51/35.78` and Novel `7.71/35.74/59.66` (SR/Acc/mIoU). Full audit: `oepp_api_eval/runs/siliconflow_qwen35_397b_full_audit_20260730.json`.
