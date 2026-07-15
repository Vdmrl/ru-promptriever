# FollowIR evaluation audit

Date: 2026-07-15

## Verdict

The FollowIR paired-confidence-interval run performed on 2026-07-14 must not
be reported. Its prediction files are structurally complete, and the paired
statistics implementation is consistent with the FollowIR metrics, but the
model inputs/checkpoints did not reproduce the paper protocol.

This does **not** imply that the paper's stored FollowIR results are wrong.
The original aggregate artifacts in `followir_results.zip` reproduce the
rounded table values exactly. New per-topic predictions are required only
because confidence intervals cannot be reconstructed from those aggregates.

## Problems found

### 1. MTEB prompt-role incompatibility

MTEB 2.10 passes the input role through `prompt_type` (`query` or `document`).
The local wrappers only inspected the older `prompt_name` argument. Therefore
the configured Promptriever prefixes were silently omitted in the 2026-07-14
FollowIR run. The official Promptriever protocol requires:

- query prefix: `query:  ` (two spaces after the colon);
- document prefix: `passage:  ` (two spaces after the colon);
- manually appended EOS before last-token pooling.

The wrappers now normalize both MTEB and local role conventions. The fix was
also applied to the encoder, Qwen3-Embedding, Giga-Embeddings, and generic
instruction wrappers because they had the same latent incompatibility.

### 2. Wrong ru-Promptriever revisions in the CI config

The English FollowIR table and the Russian mFollowIR best checkpoint do not use
the same Ru+En revision. The CI run used the later mFollowIR-selected step-300
weights and a different Ru-only revision instead of the exact weights used for
the English table.

The corrected English-table protocol is pinned in
`evaluation_pipeline/configs/eval_followir_significance.yaml`:

| Run label | Adapter repository | Adapter revision |
|---|---|---|
| `ru-only-paper` | `Vladimirlv/ru-promptriever-qwen3-4b-v2` | `fc758b4b8fa5c963013d2f05b78aa77355f6131b` |
| `ru-en-paper` | `Vladimirlv/ru-promptriever-qwen3-4b-v3` | `25a22d54b2aa415ebd6d1a0e672f68bc1e4586ae` |
| `promptriever-7b-published` | `samaya-ai/promptriever-llama2-7b-v1` | `30b14e3813c0fa45facfd01a594580c3fe5ecf23` |
| `promptriever-8b-paper` | `samaya-ai/promptriever-llama3.1-8b-v1` | `2ead22cfb1b0e0c519c371c63c2ab90ffc511b8a` |

Base-model revisions are pinned in the same config. Qwen inputs have no
query/passage prefixes; Promptriever inputs use the official double-space
prefixes. All four models append EOS explicitly.

### 3. Published baseline label

The values copied from the original Promptriever FollowIR table correspond to
the Llama-2-7B model, not the later Llama-3.1-8B model. The final paper should
label that row as Promptriever-7B. The Llama-3.1-8B rerun may be reported as an
additional baseline, but it is not the model underlying the published values.

## Checks that passed

- MTEB version in the original and CI artifacts: 2.10.5.
- Dataset revisions match for Robust04, Core17, and News21.
- Topic counts are 52, 20, and 32, respectively.
- The paper reports original-condition retrieval quality (`og` MAP@1000 for
  Robust04/Core17 and `og` nDCG@5 for News21).
- p-MRR reconstruction and paired topic alignment match the MTEB aggregate
  metric on the saved CI predictions.
- Paired bootstrap confidence intervals and the paired sign-flip test operate
  on topic-level model differences.

## Recalculation required

Re-run at least these three models after the fix:

1. `ru-only-paper`;
2. `ru-en-paper`;
3. `promptriever-7b-published`.

The Llama-3.1-8B model is optional for the Reviewer 1 response. None of the
2026-07-14 FollowIR significance values should be reused because all four runs
either used the wrong revision or the wrong input formatting.

Before accepting the new run, verify that its aggregate Ru-only and Ru+En
scores closely reproduce the stored paper artifacts:

| Model | Robust04 MAP / p-MRR | Core17 MAP / p-MRR | News21 nDCG@5 / p-MRR |
|---|---:|---:|---:|
| Ru-only | 0.30090 / 0.12141 | 0.34554 / 0.11659 | 0.49282 / 0.05228 |
| Ru+En | 0.30933 / 0.10602 | 0.34621 / 0.11829 | 0.49815 / 0.04072 |

Small floating-point differences are acceptable; a material discrepancy means
the statistics must not be used until the protocol is audited again.

## mFollowIR-RU status

The separate mFollowIR-RU evaluation uses the custom dense-retrieval path,
which explicitly passed `prompt_name="query"` and `prompt_name="passage"` even
before this fix. The step-300 Ru+En checkpoint produced p-MRR 18.54, closely
matching the paper's 18.57, so there is no evidence that the MTEB role bug
affected that result. Nevertheless, both the Ru+En and Promptriever-8B adapter
and base revisions are now pinned in `eval_mfollowir_significance.yaml`; rerun
both models before reporting the new paired confidence intervals.
