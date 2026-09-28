# Perception nuisance-gate contracts

The source diagnostic uses three component-disjoint partitions. It fits all
predeclared detectors on `scorer_fit`, selects their thresholds on
`threshold_calibration`, and reads `pilot_gate` only for the final decision.
The detector family is fixed to ridge-linear, best univariate linear, and best
pairwise-interaction scores. The last member detects simple interactions such
as XOR patterns that a linear score misses.

The reported statistic is the maximum role-recovery score over every detector
and both endpoints: within-target ranking and thresholded balanced accuracy.
The gate uses the largest per-detector upper confidence bound, and the
role-swap test recalculates a single maximum statistic after every
within-donor-component permutation. Taking the maximum of the individual 95%
upper bounds retains at
least 95% coverage for the maximum target; the maximum-statistic permutation
test separately controls the predeclared family comparison. The fixed 0.5
comparison is named structural chance
because each target contributes exactly one row per role; it is not a learned
question prior. Pair-ranking ties receive half credit, so the pair-ranking
inferential interval is the donor-component cluster bootstrap. Wilson intervals
are retained only for the binary thresholded-classification outcomes.

Create the immutable run contract before evaluating any pilot outcome:

```bash
PYTHONPATH=src python scripts/build_perception_nuisance_contract.py \
  --configuration-sha256 CONFIG_SHA256 \
  --pilot-index-sha256 PILOT_SHA256 \
  --media-receipt-sha256 RECEIPT_SHA256 \
  --media-set-sha256 MEDIA_SET_SHA256 \
  --implementation-bundle-sha256 IMPLEMENTATION_SHA256 \
  --output /new/path/nuisance-contract.json
```

The builder rejects malformed hashes and refuses to replace an existing file.
The machine-readable definition is
`schemas/perception_nuisance_contract.schema.json`. The implementation digest
is the canonical SHA-256 of the sorted filename-to-source-digest map recorded
by the runner. The detailed report and compact result bind the configuration,
pilot index, media receipt, media set, implementation bundle, and contract.
The compact decision is recalculated from the detailed report during validation.

Source-pair diagnostics do not measure the final edited bytes. A second,
automated-only check is therefore required after editing. Its index format is
`schemas/perception_post_edit_index.schema.json`. Each row binds one final
output ID to its original target, role, component, and frozen partition. Probe
the final output files with the same strict stream probe, then call
`build_post_edit_feature_records` followed by `evaluate_post_edit_diagnostics`.
This second gate compares target-to-output engineering differences and applies
the same detector family, per-detector intervals aggregated by their maximum,
and maximum-statistic test.
It performs no listening or other human evaluation, and it does not establish
perceptual indistinguishability.
