# ConflictBench: Controlled Acquisition and Abstention for Three-Channel Evidence

MOSEI caches now use schema v2. Earlier v1 caches averaged full video features
and used only the first utterance label; discard them from scientific analyses
and regenerate the cache from CSD sources. The loader now aligns feature rows
with positive temporal overlap to each label interval and averages those rows.
Each utterance has a distinct `video[index]` ID. MOSEI `seen` and `unseen`
output keys designate validation and test folds, not corruption generalization.

ConflictBench is a small, self-contained benchmark for studying decisions when three evidence channels disagree. The channels are numeric stand-ins for audio, video, and text, so the package tests decision logic without distributing media. It provides clean, corrupted, dropped, mixed, high-confidence, and symmetric-ambiguity cases.

The package compares transparent fixed rules, an observation-only threshold policy, optional PyTorch fusion models, and a learned acquisition policy. The acquisition scorer sees only the initial pair, learns a cost-sensitive query benefit from training labels, and chooses its threshold on a disjoint train-only calibration split. It reports coverage, selective accuracy, abstention, query use, per-mechanism results, and a utility that charges for requests and unsafe abstention. Generator mechanism labels are never passed to a policy.

## Quick start

The CREMA-D replay was checked with Python 3.10.20; its optional dependency
versions are pinned in `pyproject.toml`. The `mosei` and `router` extras select
scikit-learn 1.7.2 on Python 3.10 and 1.9.0 on Python 3.11 or later. The locked
MOSEI router study requires scikit-learn 1.9.0, so use Python 3.11 or later for
that configuration. Other optional pipelines may need their own environment.

Matched controls are available in `conflictbench.controls`. At any positive
threshold, active and always-query-selective have identical final actions;
their utility difference is query cost times the avoided request rate. The
generator's mixed and ambiguity mechanisms both hide the binary label by
sign/permutation symmetry, but only ambiguity has a rejection target. This is
a mechanism-specific reward convention, not a universal identifiability test.

Run `python scripts/protocol_stress.py --output stress.json` for the descriptive
five-seed threshold/noise grid. Run `python scripts/mosei_controls.py --cache
data/cmu-mosei/scores-utterance-v2.npz --output controls.json` to select thresholds
by validation utility only and evaluate fixed test folds. Both scripts refuse
to overwrite an existing output. Use a machine suited to the selected workload.

```bash
python -m pip install -e '.[dev,mosei]'
python scripts/run_benchmark.py --config configs/quickstart.json --output quickstart.json
python -m pytest
```

### CREMA-D raw-media replay

CREMA-D media are obtained separately from the [official repository](https://github.com/CheyneyComputerScience/CREMA-D). Its database is licensed under the Open Database License (ODbL 1.0), and rights in individual contents are licensed under the Database Contents License (DbCL 1.0); review the upstream `LICENSE.txt` and complete the [access form requested in its README](https://github.com/CheyneyComputerScience/CREMA-D#access) when using the GitHub repository. These dataset terms are separate from this package's MIT software license. The checked replay used Python 3.10.20, the versions in the `cremad` optional dependency set, an `ffmpeg` executable on `PATH`, and one CUDA GPU for feature extraction. Run media verification and inference in a suitable compute allocation; the full media download is about 3 GB. The source package contains no media or model weights.

Clone commit `1658cd342dff90010aa843eaeebd53610a08b1dc` with Git LFS smudge disabled. Run the pointer phase while WAV and FLV paths still contain LFS pointers, then fetch media and run the verify phase. `DATA_ROOT` must contain the dataset checkout.

```bash
python -m pip install -e '.[cremad]'
mkdir -p "$DATA_ROOT"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git "$DATA_ROOT/cremad"
git -C "$DATA_ROOT/cremad" checkout 1658cd342dff90010aa843eaeebd53610a08b1dc
python scripts/prepare_cremad_sources.py pointer --data-root "$DATA_ROOT" --repository-root "$DATA_ROOT/cremad"
git -C "$DATA_ROOT/cremad" lfs pull
python scripts/prepare_cremad_sources.py verify --data-root "$DATA_ROOT" --repository-root "$DATA_ROOT/cremad"
```

The extractor pins both model revisions and records their identities and the audio positional-convolution weight digest in each output shard. Use `facebook/wav2vec2-base-960h` at `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` with `google/vit-base-patch16-224` at `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` for the one-, four-, and eight-frame checks. Use `facebook/hubert-base-ls960` at `dba3bb02fda4248b6e082697eee756de8fe8aa8a` with `facebook/deit-base-patch16-224` at `fb2c78a54a5637dec350432794f7b93e31f910c9` for the eight-frame swap. For each pair and frame count, run 48 disjoint shards with `scripts/extract_cremad_pretrained_temporal.py --index "$DATA_ROOT/cremad_paired_index.json" --shard SHARD --shards 48 --video-frames FRAMES --audio-model AUDIO_ID --audio-revision AUDIO_REV --video-model VIDEO_ID --video-revision VIDEO_REV --output OUTPUT.npz`. Each shard must have a distinct output path. Then use `scripts/aggregate_cremad_features.py` and `scripts/evaluate_cremad_temporal_policy.py` on the complete 48-shard directory. `scripts/evaluate_cremad_comprehensive.py` joins the two eight-frame directories by clip ID for the four encoder pairings; `scripts/aggregate_cremad_comprehensive.py` summarizes six overlapping actor reallocations. These scripts refuse to overwrite result files.

### Exact scalar campaign

`configs/scalar_campaign_seeds.json` fixes the 19 primary and 64 disjoint
replication seeds used by the paper. The paired runner executes both the base
and leak-free learned-acquisition configurations and refuses to overwrite
existing outputs:

```bash
python scripts/run_scalar_campaign.py \
  --seed-plan configs/scalar_campaign_seeds.json \
  --campaign-config configs/campaign.json \
  --acquisition-config configs/campaign_acquisition.json \
  --output-dir runs/scalar --device auto
python scripts/scalar_inference.py \
  --campaign runs/scalar/campaign/*.json \
  --acquisition runs/scalar/acquisition/*.json \
  --output runs/scalar/summary.json
```

## Matched-budget selector-value study

This study tests whether two available scores can rank examples whose decision
changes after acquiring the third score. The terminal rule is fixed for both
the initial pair and the post-request triple: take the sign of the score sum
when its magnitude is at least 0.35, and otherwise abstain. A standardized
ridge model is fitted on generated training data to predict the reward change
from replacing the pair decision with the triple decision. Its seven input
features use only the observed pair. Training uses clean, invert, swap, and
dropout mechanisms; evaluation uses clean, mixed, burst, and ambiguity cases.

The evaluation compares train-only value ranking, pair uncertainty, matched
random selection, a no-query reference, and an oracle upper bound at exact
request rates of 10%, 25%, 50%, and 63.7%. Matched random uses a seed offset of
500003. The oracle uses evaluation labels and the unavailable score and is not
a deployable selector. The 83 seeds come from
`configs/scalar_campaign_seeds.json`; aggregation requires the full roster and
reports paired seed-bootstrap intervals.

```bash
mkdir -p runs/matched-selector
for seed in $(python3 -c "import json; d=json.load(open('configs/scalar_campaign_seeds.json')); print(*d['primary_seeds'], *d['replication_seeds'])"); do
  python3 scripts/run_matched_selector_value.py \
    --config configs/matched_selector_value.json \
    --seed "$seed" \
    --output "runs/matched-selector/seed_${seed}.json"
done
python3 scripts/aggregate_matched_selector_value.py \
  --seed-plan configs/scalar_campaign_seeds.json \
  --input-dir runs/matched-selector \
  --output runs/matched-selector/aggregate.json
```

The runner writes one seed at a time and refuses to overwrite an existing
output. The aggregator rejects missing or extra seeds, checks that every run
uses the same protocol, and computes intervals from paired seed resamples.

## Controlled pre-query value diagnostics

This synthetic study separates information available to an oracle from a cue
available before acquisition. One candidate always helps by 0.20 and the other
always harms by 0.20; the observable cue identifies the helpful candidate with
probability `(1 + rho) / 2`. At an exact 50% request budget, the cue policy is
compared with matched random selection and an oracle that sees the hidden
candidate. Separate follow-ups add a cost-aware abstention rule and a policy
calibrated on disjoint examples. These checks validate the simulator's
information boundary; they are not natural audio, video, or text results.

The four paper studies each use 24 evaluation seeds. The fixed-budget families
use 400,000 examples per seed; calibrated families also use a separate
calibration simulation. Run and aggregate the no-cost cue family as follows:

```bash
mkdir -p runs/identifiability/cue
for seed in $(seq 101 124); do
  python scripts/run_identifiability_simulation.py \
    --seed "$seed" --samples 400000 --oracle-headroom 0.20 --budget 0.50 \
    --candidate-costs 0,0 --cue-strengths 0,0.2,0.4,0.6,0.8,1.0 \
    --output "runs/identifiability/cue/seed-${seed}.json"
done
python scripts/aggregate_identifiability.py \
  --input-dir runs/identifiability/cue \
  --expected-seeds "$(seq -s, 101 124)" \
  --output runs/identifiability/cue_aggregate.json
```

For the cost-aware family, use seeds 201--224, candidate costs `0.02,0.02`,
and the same runner, sample count, budget, and cue strengths. Aggregate that
family separately with `aggregate_identifiability.py` and expected seeds
`$(seq -s, 201 224)`.

The calibrated family uses seeds 301--324, calibration seeds offset by 100,000,
400,000 calibration examples, candidate costs `0.02,0.10`, and cue strengths
`0,0.4,0.8`. The finite-calibration sensitivity uses seeds 401--424 with the
same settings but 400 calibration examples. For either family, run:

```bash
mkdir -p runs/identifiability/calibrated
for seed in $(seq 301 324); do
  calibration_seed=$((seed + 100000))
  python scripts/run_calibrated_identifiability_simulation.py \
    --seed "$seed" --calibration-seed "$calibration_seed" \
    --samples 400000 --calibration-samples 400000 \
    --oracle-headroom 0.20 --budget 0.50 --candidate-costs 0.02,0.10 \
    --cue-strengths 0,0.4,0.8 \
    --output "runs/identifiability/calibrated/seed-${seed}.json"
done
python scripts/aggregate_calibrated_identifiability.py \
  --input-dir runs/identifiability/calibrated \
  --expected-seeds "$(seq -s, 301 324)" \
  --output runs/identifiability/calibrated_aggregate.json
```

Replace the seed range, calibration sample count, output directory, and expected
seed range together to run the finite-calibration sensitivity. Both aggregators
reject missing, duplicate, or unexpected seed outputs. Each runner refuses to
overwrite an existing result file.

## Temporal factorial campaign

The temporal extension keeps sequence-valued evidence for the learned fusion
baselines while evaluating scalar controls on the time mean. The supplied
campaign specification spans sequence lengths 32, 64, and 128 at three noise
levels with disjoint seed sets for the matched model and control lanes. Resolve
and run one configuration on any compatible machine with the optional PyTorch
dependency:

```bash
python scripts/run_temporal_campaign.py \
  --campaign-spec configs/temporal_factorial_campaign.json \
  --seed-group replication --index 0 --device cpu --output-dir runs/temporal
```

Use `--seed-group primary --device cuda` when CUDA is available. The device is
selected independently from the research seed group. The runner records the
resolved configuration and rejects device fallbacks. After all expected result
files are present, validate and summarize the matched seed sets:

```bash
python scripts/analyze_temporal_campaign.py \
  --campaign-spec configs/temporal_factorial_campaign.json \
  --input-dir runs/temporal \
  --output runs/temporal_aggregate.json \
  --markdown runs/temporal_report.md
```

For the higher-compute robustness extension, use
`configs/temporal_snr_robustness_campaign.json`. It fixes the sequence length
at 64, varies event strength and observation noise over nine cells, and uses a
larger sequence encoder with the locked optimization schedule. Run each
resolved configuration with the appropriate device and aggregate it with the
same analyzer, changing the specification, input directory, and output names
to `temporal_snr_robustness`. The bounded control lane is a deterministic
execution check and is not pooled with the learned-model robustness estimates.

An independent-seed replication is prepared in
`configs/temporal_snr_replication_campaign.json`. It keeps the same nine cells,
data boundary, model, and optimization schedule while changing only the
disjoint seeds. This is a replication of the SNR grid, not a second
hyperparameter sweep. Resolve each element with
`scripts/run_temporal_campaign.py`, then run the analyzer only after every
expected result passes strict validation.

## Acquisition-value pilot

The value study treats the compact utterance-aligned CMU-MOSEI score cache as
a frozen upstream representation. Its supervised projection was fit on the
complete official training fold, so this experiment is an exploratory test of
downstream routing, not an isolated representation-learning result. A strict
confirmation must use a label-free frozen representation or refit the
projection within each task-fit partition.

The runner partitions training video groups into task fitting, router fitting,
and confidence calibration. It fits a separate frozen ridge task head for each
nonempty observation mask and requires every mask to reach the configured
validation-accuracy floor. The linear value router uses candidate-by-observed-
feature interactions, so its preferred missing modality can vary by example.
Two hybrids reuse the exact linear or confidence query set while swapping only
the candidate-choice rule, which separates timing from modality selection.
It evaluates singleton-start selection and the secondary pair-to-third setting
on official validation while leaving test arrays unopened. Exact-budget error
intervals reselect queries inside every video-cluster resample. The paired 90%
selective-risk interval also reselects both the query budget and exact coverage
inside each resample. These intervals condition on the fitted models; the five
split-seed estimates describe fitting variability. Run all locked seeds, verify
each result, then aggregate the exact file set:

```bash
python scripts/audit_mosei_value_cache.py \
  --cache data/cmu-mosei/scores-utterance-v2.npz \
  --expected-sha256 2805d35e9fb9c3f51d4280a439ad49063d8bbdc37b12d4f201045eadf7365cb1 \
  --output runs/mosei-value-cache-audit.json
python scripts/run_mosei_value_pilot.py \
  --config configs/mosei_value_study.json \
  --cache data/cmu-mosei/scores-utterance-v2.npz \
  --mode singleton --seed 11 --output runs/mosei-value-singleton-11.json
python scripts/verify_mosei_value.py runs/mosei-value-singleton-11.json \
  --mode singleton --seed 11 --config configs/mosei_value_study.json
```

Repeat the runner for `pair` and for seeds `23`, `37`, `41`, and `53`. Pass
the ten explicit paths to `scripts/aggregate_mosei_value.py`; the aggregator
rejects missing, duplicate, stale-configuration, or unexpected seeds. At the
locked 50% query budget, the singleton screening gate requires at least `0.005`
absolute error-reduction gain over the calibration-selected feasible baseline,
a positive lower bound from the pooled paired video-cluster interval, and no
descriptive increase in mean harmful-query rate.

For a locked run, pass `--cache` and `--write-attestation` to
`scripts/verify_mosei_value.py`. The verifier deterministically recomputes the
result from the cache, compares the complete object, and binds the accepted
file hash to a small verification record. Check that record again before
aggregation with `--check-attestation`.

### Singleton direct-action extension

The v3 singleton extension adds a direct cost-sensitive action baseline
without changing v2 outputs. It fits task heads on task-fit video groups,
action risk on router-fit groups, and risk calibration plus online query
thresholds on calibration groups. Per-mask task correctness is also calibrated
there, so the post-acquisition answer/abstain decision uses the newly observed
task score without changing the pre-query action. The primary separate-mask
ridge is evaluated with shared mask-aware and deterministic modality-dropout
sensitivities. A
representation-dependence gate compares their locked 50% budget result, and
each head must reach the predeclared `0.65` validation-accuracy floor for every
nonempty modality mask. The normalized 0/1 task-loss matrix assigns abstention
cost `0.25` and acquisition cost `0.05` per modality as a fixed analysis
sensitivity; these values are not measured latency or runtime costs.

The published-comparator audit in
`docs/mosei_published_comparator_compatibility.md` finds that neither A2MT nor
ACO preserves its published objective and action space under the frozen
one-acquisition, explicit-abstention contract. The extension is consequently
benchmark-only.

```bash
python3.11 -m pip install -r requirements-mosei-singleton-v3.txt -e .
for seed in 11 23 37 41 53; do
  python3.11 scripts/run_mosei_singleton_baselines.py \
    --config configs/mosei_singleton_baselines_v3.json \
    --cache data/cmu-mosei/scores-utterance-v2.npz \
    --seed "${seed}" \
    --output "runs/mosei-singleton-baselines-v3-${seed}.json"
  python3.11 scripts/verify_mosei_singleton_baselines.py \
    "runs/mosei-singleton-baselines-v3-${seed}.json" \
    --config configs/mosei_singleton_baselines_v3.json \
    --cache data/cmu-mosei/scores-utterance-v2.npz \
    --seed "${seed}"
done

python3.11 scripts/aggregate_mosei_singleton_baselines.py \
  --config configs/mosei_singleton_baselines_v3.json \
  --input runs/mosei-singleton-baselines-v3-11.json \
  --input runs/mosei-singleton-baselines-v3-23.json \
  --input runs/mosei-singleton-baselines-v3-37.json \
  --input runs/mosei-singleton-baselines-v3-41.json \
  --input runs/mosei-singleton-baselines-v3-53.json \
  --output runs/mosei-singleton-baselines-v3-aggregate.json
python3.11 scripts/verify_mosei_singleton_aggregate.py \
  runs/mosei-singleton-baselines-v3-aggregate.json \
  --config configs/mosei_singleton_baselines_v3.json \
  --input runs/mosei-singleton-baselines-v3-11.json \
  --input runs/mosei-singleton-baselines-v3-23.json \
  --input runs/mosei-singleton-baselines-v3-37.json \
  --input runs/mosei-singleton-baselines-v3-41.json \
  --input runs/mosei-singleton-baselines-v3-53.json
```

The singleton v3 runner requires the router dependencies and enforces the exact
runtime recorded in its configuration: Python 3.11.15, NumPy 2.4.6, and
scikit-learn 1.9.0. The requirements file pins the two Python packages; select
the stated CPython interpreter before installation. The runner fails closed on
any version mismatch. Install the optional `torch` extra to train the learned
fusion and acquisition baselines:

```bash
python3.11 -m pip install -e '.[torch]'
```

## Public MMSA descriptors

The runner also accepts processed descriptor pickles released by the
Multimodal Sentiment Analysis (MMSA) project for CMU-MOSI, CH-SIMS, and
CH-SIMS v2. These inputs contain pre-extracted audio, visual, and text
sequences only; this package does not download raw recordings. Install the
small downloader extra, then fetch one pinned processed file into its own
directory:

```bash
python -m pip install -e '.[mmsa]'
python scripts/download_mmsa.py --dataset mosi --output-dir data/cmu-mosi
python scripts/cache_mmsa.py \
  --feature-path data/cmu-mosi/aligned_50.pkl \
  --dataset CMU-MOSI \
  --output data/cmu-mosi/scores.npz
python scripts/run_benchmark.py \
  --config configs/mmsa_example.json \
  --output mosi.json
```

Use `--dataset sims` for CH-SIMS (`unaligned_39.pkl`). The built-in CMU-MOSI
and CH-SIMS files are pinned to SHA-256 values
`d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9`
and `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`.
CH-SIMS v2 remains selectable only with an independently obtained digest:

```bash
python scripts/download_mmsa.py --dataset sims_v2 \
  --expected-sha256 EXPECTED_64_HEX_SHA256 --output-dir data/ch-sims-v2
```

The helper fails before downloading an unpinned source unless that digest is
provided, verifies it, and writes `download_record.json` beside the input.
The source IDs follow the MMSA repository at commit
[`a94e65d07fa1ae0d44e552390074b29b0898edfd`](https://github.com/thuiar/MMSA/tree/a94e65d07fa1ae0d44e552390074b29b0898edfd).

The loader mean-pools each sequence using its supplied length, removes padded
rows, and fits one standardized positive-minus-negative mean projection per
modality on the training split only. `sentiment_mode` defaults to
`nonnegative` (zero maps to the positive class); `positive` omits zero labels.
Validation and test become the `seen` and `unseen` output keys, respectively,
and their sample IDs plus feature dimensions are retained in the run record.
This is evidence about the published descriptor file and this projection, not
about raw recordings, temporal alignment, or a listener study.

To run on CMU-MOSEI computational sequence descriptors, install the optional
HDF5 reader and obtain the official CSD files through the acquisition helper:

```bash
python -m pip install -e '.[mosei]'
python scripts/download_mosei.py --output-dir data/cmu-mosei
python scripts/run_benchmark.py \
  --config configs/mosei_example.json \
  --output mosei.json
```

For the paper's fixed 40-epoch schedule, use `configs/mosei_paper.json` with
seeds `11`, `23`, `37`, `41`, and `53`.

The helper uses the URLs published by the CMU Multimodal SDK for COVAREP
audio, OpenFace 2 video, timestamped word vectors, sentiment labels, and the
standard train/validation/test video folds. The legacy host may be blocked or
offline; use `--base-url` for a CMU-style approved mirror and keep the same
filenames. An explicitly pinned Hugging Face mirror can be used when the
official host is unavailable:

```bash
python scripts/download_mosei.py \
  --output-dir data/cmu-mosei \
  --hf-repo reeha-parkar/cmu-mosei-comp-seq \
  --hf-revision 5f8d513c34278006d27f98e5609564e6d4b353d7
```

That repository describes itself as an unofficial mirror. The command records
the repository, revision, resolved URLs, pinned fold source, observed SHA-256
values, optional expected SHA-256 values, and per-file status in
`data/cmu-mosei/download_record.json`; retain that record with any run. Supply
`--expected-sha256 NAME=SHA256` once an authoritative digest is available to
make any source mismatch fatal before use.
The release filename `CMU_MOSEI_VisualOpenFace2.csd` is populated from the
mirror's `CMU_MOSEI_OpenFace2.csd` file.
The official fold file lists 2,249/300/678 train/validation/test videos. At the
pinned mirror revision above, two test video IDs (`7l3BNtSE0xc` and
`dZFV0lyedX4`) have no shared labeled CSD entry, so the reproducible observed
test split covers 676 videos, not 676 utterances. The loader records both official and
observed counts plus unmatched IDs in the cache metadata; the cache construction
gate requires this exact discrepancy and fails closed on other drift. The
loader maps validation to the `seen` split and test to `unseen`, averages
frame-level descriptors, and fits one standardized mean-difference projection
per modality on training segments only. Sentiment is binary by default with
zero treated as non-negative; set `sentiment_mode` to `positive` to omit
zero-valued labels.

For a multi-seed campaign, project the CSD files once and point later configs
at the resulting compact cache:

```bash
python scripts/cache_mosei.py \
  --audio data/cmu-mosei/CMU_MOSEI_COVAREP.csd \
  --video data/cmu-mosei/CMU_MOSEI_VisualOpenFace2.csd \
  --text data/cmu-mosei/CMU_MOSEI_TimestampedWordVectors.csd \
  --labels data/cmu-mosei/CMU_MOSEI_Labels.csd \
  --splits data/cmu-mosei/splits.json \
  --output data/cmu-mosei/scores.npz
```

Set `cache_path` in a run config to use that cache. The cache stores only the
three projected scores, labels, IDs, and preprocessing metadata; it does not
replace the source CSD files or alter the train-only fitting rule.

The output JSON is intended for local analysis. It contains no machine-specific paths and does not require a dataset download.

## Protocol

Training uses clean, single-inversion, source-swap, and dropout mechanisms. Evaluation has a seen split and a partially held-out split with a clean control plus mixed corruption, high-confidence bursts, and symmetric ambiguity as novel mechanisms. A policy may abstain on an ambiguous case; the utility makes this choice explicit. Change the configuration to study different noise, sizes, or costs.

The synthetic configuration is a controlled diagnostic package, not evidence
about real recordings or deployed systems. A MOSEI run is evidence about the
released CSD representation and this protocol only; it does not establish
performance for raw recordings, listeners, or deployment.

## Perception Test structural candidate-pool audit

The Perception Test MC-QA audit checks whether repeated question/option groups
have enough answer-balanced videos in both train and validation annotations to
support a later candidate-pool study. It enforces the canonical official
repository and archive URLs, pinned repository commit and object generations,
and the exact archive sizes, SHA-256 values, members, and snapshot counts in
`configs/perception_test_mcqa_audit.json`. The reviewed configuration SHA-256 is
`0ed19f8193f4872897b572553869ac0ca3732a02f63afd4ca18794425000fe3a`.
The audit reads no media or test annotations. It does inspect validation answer
labels to compute descriptive cross-split counts. The later pilot builder does
not use those counts for eligibility, sampling, strata, or donor assignment.

```bash
PYTHONPATH=src python scripts/audit_perception_candidate_pool.py \
  --config configs/perception_test_mcqa_audit.json \
  --train-archive /path/to/mc_question_train_annotations.zip \
  --validation-archive /path/to/mc_question_valid_annotations.zip \
  --output /path/to/perception_candidate_audit.json
```

The deterministic JSON reports two distinct tiers and records every excluded
duplicate-option question by video ID, question ID, and reason. In the pinned
snapshot, the split-local tier with at least two observed answer labels has 252
groups, 5,757 questions, and 1,912 videos in train and 382 groups, 15,380
questions, and 5,185 videos in validation. The cross-split tier requires the
same observed answer vocabulary in both splits, at least two supported answers,
and at least five unique videos for every supported answer in each split. It
contains 59 groups: 29 with two supported answers and 30 with all three. These
groups contain 3,860/9,916 questions and 1,711/4,485 unique videos in
train/validation. The stricter all-three-option subset contains 30 groups,
1,927/4,988 questions, and 1,181/3,038 unique videos. These are structural
candidate-pool counts only; they do not establish that audio or video is
sufficient to answer a question.

## Training-only Perception Test media pilot

The frozen pilot builder authenticates the audit and both annotation archives,
then loads only the training annotations. Its independent eligibility rule
requires at least two observed answers and at least five unique training videos
for each supported answer. The pinned snapshot yields 85 eligible keys: 55 with
two supported answers and 30 with all three. A deterministic assignment selects
100 unique targets and creates exactly two pairs per target: one same-answer
nuisance pair and one opposite-answer candidate pair. Donor-connected
components stay within the `scorer_fit`, `threshold_calibration`, or
`pilot_gate` partition. The builder does not load media or model scores.

```bash
PYTHONPATH=src python scripts/build_perception_media_pilot.py \
  --config configs/perception_test_media_pilot.json \
  --structural-audit /path/to/perception_candidate_audit.json \
  --structural-audit-sha256 b34916cf9a3c91707daaba16fb2f9ae56fb27c0d10aa08cb3c3e4bac63216388 \
  --train-archive /path/to/mc_question_train_annotations.zip \
  --validation-archive /path/to/mc_question_valid_annotations.zip \
  --output /path/to/pilot-index.json
```

The reviewed pilot configuration SHA-256 is
`c86218903f640c6ef4843e7cf010e683dbdb873ea0673571573d4c7259d33d4a`.
Its declared implementation digest is
`fab8c77d0808e10e332a2b04f827f8adb59a1bf72e3bc3f9fdc4160f7dbfab09`.
The output is write-once and binds the exact source digests. After separately
authenticating the official training-video archive, extract only the selected
files and verify that each contains both audio and video streams:

```bash
PYTHONPATH=src python scripts/extract_perception_pilot_media.py \
  --archive /path/to/train_videos.zip \
  --archive-sha256 EXPECTED_SHA256 \
  --pilot-index /path/to/pilot-index.json \
  --pilot-index-sha256 EXPECTED_PILOT_SHA256 \
  --output-root /path/to/selected-media \
  --ffprobe /path/to/ffprobe \
  --receipt /path/to/media-receipt.json
```

Construction success is not a source-sufficiency result. The frozen multimodal
scorer and the separate nuisance-detection gate must both pass before any
validation-media stage is allowed.

The source-sufficiency runner pins Qwen2.5-Omni-3B revision
`f75b40e3da2003cdd6e1829b1f420ca70797c34e` and evaluates every indexed target
and donor under audio-only, video-only, and audiovisual conditions. It preserves
the two donor roles separately, deduplicates repeated target evaluations for
target accuracy, embeds the complete outcome-blind design, and reconstructs
every request identifier during output validation. Thresholds in
`configs/perception_omni_source_gate.json` are engineering continuation gates,
not calibrated performance estimates. A valid negative result is written and
returned successfully; it stops the controlled-conflict branch. Even a positive
source result remains pending until an independently hash-locked nuisance test
passes.

On a CUDA 12.6 supported Linux or Windows environment, install the pinned
PyTorch build from its package index, then install the Qwen extra and its PyPI
dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 'torch==2.11.0+cu126'
python -m pip install -e '.[qwen]'
```

Video preprocessing requests 2 frames per second, bounds each sample to 4--32
frames, and uses 100,352 minimum pixels and 200,704 maximum pixels per frame.
These values are part of the authenticated configuration and are passed
unchanged to both the video-only and audiovisual paths. The realized sampling
rate returned by the decoder is forwarded to the model processor, so capped and
rounded frame counts retain their actual temporal spacing. Changing any value
creates a different configuration and result.

The same frozen cohort also runs target- and donor-oriented question-only
controls and same-question counteranswer-media controls for all three media
conditions. Correct-answer positions are assigned cyclically within each
partition and source stratum, with at most one count of imbalance across A, B,
and C. Target requests are deduplicated only after their two role copies agree.
Every donor control and aligned-over-control margin is evaluated separately for
the same-answer and opposite-answer pair roles, preventing one role from hiding
leakage in the other. Controls cannot change cohort membership after inference.
Continuation requires question-only and shuffled-media source-answer agreement
to remain at or below `0.50`, and aligned-source accuracy to exceed the stronger
control by at least `0.20` in every locked cell. Integer count comparisons are
used at the boundary, so an exact `0.20` margin passes.

The output also reports each exact `pilot_gate` component outcome for every
condition-by-orientation source cell, with donor cells kept separate by pair
role. Component success means that every aligned-source record in that cell is
correct. The corresponding successes out of 20 and Wilson 95% interval are
descriptive uncertainty for this feasibility screen, not stable performance
estimates. A secondary normalized-question split leaves the primary
video-component cohort unchanged. It evaluates only `pilot_gate` question keys
absent from `scorer_fit` and `threshold_calibration`, reports every observed
held-out cell even when coverage is insufficient, requires at least 15 such
components, and fails closed if any held-out-key aligned-over-control margin is
attenuated by more than `0.20` from its primary counterpart. These bounds are
conservative screening cutoffs above three-choice chance (`1/3`).

```bash
PYTHONPATH=src python scripts/run_perception_omni_gate.py \
  --config configs/perception_omni_source_gate.json \
  --config-sha256 59e7d06bfc5b5df032d4deda394f23261460469ff1bff2ac10edd92b230912a7 \
  --pilot-index /path/to/pilot-index.json \
  --pilot-index-sha256 EXPECTED_PILOT_SHA256 \
  --media-root /path/to/selected-media \
  --model-cache /path/to/model-cache \
  --output /path/to/source-gate.json
```

After the run, freeze the exact model-response transcript and record the printed
file SHA-256 outside the result file:

```bash
PYTHONPATH=src python - /path/to/source-gate.json /path/to/source-gate-transcript.json <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from conflictbench.perception_omni_gate import build_inference_transcript

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
output = json.loads(source.read_text(encoding="utf-8"))
transcript = build_inference_transcript(output)
data = (json.dumps(transcript, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
destination.write_bytes(data)
print(hashlib.sha256(data).hexdigest())
PY
```

The independent replay verifier invokes no model. It reauthenticates the locked
configuration, training pilot, every media byte, and the exact source files;
rebuilds option balancing and every request; replays the anchored responses;
and then reconstructs every record, summary, transcript digest, and final
payload:

```bash
PYTHONPATH=src python scripts/verify_perception_omni_gate.py \
  --config configs/perception_omni_source_gate.json \
  --config-sha256 59e7d06bfc5b5df032d4deda394f23261460469ff1bff2ac10edd92b230912a7 \
  --pilot-index /path/to/pilot-index.json \
  --pilot-index-sha256 EXPECTED_PILOT_SHA256 \
  --media-root /path/to/selected-media \
  --source src/conflictbench/perception_omni_gate.py \
  --source scripts/run_perception_omni_gate.py \
  --transcript /path/to/source-gate-transcript.json \
  --transcript-sha256 EXPECTED_TRANSCRIPT_FILE_SHA256 \
  --output /path/to/source-gate.json
```

This replay detects later result rewriting once the transcript file digest has
been anchored. It does not independently establish that the original backend
execution was honest or that the frozen model weights matched the named remote
revision; those remain execution-provenance limitations.

## Automated Perception Test edit-role nuisance gate

The nuisance gate tests whether container and stream engineering measurements
can distinguish same-answer donor pairs from opposite-answer donor pairs. Its
15 fixed inputs cover duration, byte rate, audio/video duration alignment,
resolution, frame rate, sample rate, channel count, codecs, and sample formats.
It never uses question text, answer options, answers, semantic embeddings,
source-answer scores, or source-sufficiency outcomes.

The three predeclared detectors (ridge-linear, univariate linear, and a shallow
pairwise-interaction detector) are fitted only on `scorer_fit`; each decision
threshold is selected only on `threshold_calibration`; and `pilot_gate` is
evaluated once. Every
target--donor component and media file must remain in one partition. The report
contains exact target and component outcomes, a component-resampled interval,
a Wilson score interval for Bernoulli thresholded classification, a
within-component role-swap-with-refit maximum-statistic permutation test, a
structural-chance reference, and an unseen-question-key sensitivity check.
Ranking ties are fractional credits, so ranking uses only the
component-cluster bootstrap interval and never a Wilson interval. The gate
takes the largest upper confidence bound across the predeclared detectors and
endpoints.
The locked configuration SHA-256 is
`61f68aa4ac5b4701b01b35bad1b518914d330178189e6108b8cc7a5373f5ad67`.
The primary role-recovery statistic is the larger of pair-ranking accuracy and
thresholded balanced accuracy, maximized over the detector family; its upper
bound is the largest 95% per-endpoint bound. Taking the maximum preserves at
least nominal coverage for the maximum target because its true maximizing
member is one of the predeclared endpoints. Passing requires that primary bound
to be at most `0.75`,
the unseen-question upper bound to be at most `0.80`, detector lift over the
structural-chance reference to be at most `0.10`, structural-chance balanced
accuracy to be at most `0.60`, and the maximum-statistic role-swap permutation
p-value above `0.05`. These fixed cutoffs
screen out a readily recoverable edit role. They are not an equivalence test;
small or connected pilot samples widen the interval and make the gate more
likely to fail closed.

```bash
IMPLEMENTATION_SHA256="$(PYTHONPATH=src python - <<'PY'
import hashlib
import json
from pathlib import Path

sources = {
    "__init__.py": Path("src/conflictbench/__init__.py"),
    "runner.py": Path("scripts/run_perception_nuisance_gate.py"),
    "perception_nuisance_gate.py": Path("src/conflictbench/perception_nuisance_gate.py"),
    "perception_media_pilot.py": Path("src/conflictbench/perception_media_pilot.py"),
    "perception_candidate_audit.py": Path("src/conflictbench/perception_candidate_audit.py"),
    "perception_omni_gate.py": Path("src/conflictbench/perception_omni_gate.py"),
}
digests = {role: hashlib.sha256(path.read_bytes()).hexdigest()
           for role, path in sources.items()}
payload = json.dumps(dict(sorted(digests.items())), ensure_ascii=False,
                     sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"

PYTHONPATH=src python scripts/run_perception_nuisance_gate.py \
  --config configs/perception_nuisance_gate.json \
  --config-sha256 61f68aa4ac5b4701b01b35bad1b518914d330178189e6108b8cc7a5373f5ad67 \
  --pilot-index /path/to/pilot-index.json \
  --pilot-index-sha256 EXPECTED_PILOT_SHA256 \
  --media-receipt /path/to/media-receipt.json \
  --media-receipt-sha256 EXPECTED_RECEIPT_SHA256 \
  --media-root /path/to/selected-media \
  --media-set-sha256 EXPECTED_MEDIA_SET_SHA256 \
  --implementation-bundle-sha256 "${IMPLEMENTATION_SHA256}" \
  --nuisance-contract /path/to/nuisance-contract.json \
  --nuisance-contract-sha256 EXPECTED_CONTRACT_SHA256 \
  --ffprobe /path/to/ffprobe \
  --report /path/to/nuisance-report.json \
  --result /path/to/nuisance-result.json
```

This gate cannot run from the repository alone: the authenticated pilot index
and its selected training-media construction are not stored here. It rejects
missing, writable, changed, or extra media and will not create a result until
all bytes and prerequisite records match their supplied hashes. Its features
measure prospective swap compatibility from source streams. The separate
automated post-edit gate described in
`docs/perception_nuisance_protocol.md` applies the same controls to final edited
outputs. Neither diagnostic establishes perceptual indistinguishability.

This release does not include a standalone post-edit command. Its executable
path is the public Python API below. The two diagnostics JSON files must be
objects keyed by `target_video_id` and `output_video_id`, respectively, whose
values are normalized records returned by `probe_media_file`. The index must
conform to `schemas/perception_post_edit_index.schema.json`; the output path is
opened exclusively so an existing report is never replaced.

```bash
PYTHONPATH=src python - \
  /path/to/post-edit-index.json \
  /path/to/source-diagnostics.json \
  /path/to/output-diagnostics.json \
  configs/perception_nuisance_gate.json \
  /new/path/post-edit-nuisance-report.json <<'PY'
import json
import pathlib
import sys

from conflictbench.perception_nuisance_gate import evaluate_post_edit_diagnostics

index_path, source_path, output_path, config_path, report_path = map(
    pathlib.Path, sys.argv[1:]
)
load = lambda path: json.loads(path.read_text(encoding="utf-8"))
report = evaluate_post_edit_diagnostics(
    load(index_path), load(source_path), load(output_path), load(config_path)
)
with report_path.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
PY
```

## Package layout

The following paper-facing analysis entrypoints are available without derived result files:

- `scripts/scalar_inference.py`: paired-seed intervals and clean-excluded summaries;
- `scripts/run_temporal_shortcut_check.py` and `scripts/analyze_temporal_shortcut_check.py`: equal-information temporal controls;
- `scripts/mosei_cluster_uncertainty.py`: official-video-cluster intervals for fixed CMU-MOSEI policies;
- `scripts/perception_source_uncertainty.py`: Wilson pair-support intervals and paired component bootstrap intervals for an authenticated source-gate result.

The analysis scripts require their corresponding generated records or licensed
benchmark inputs. Those data and execution records are not redistributed.

- `src/conflictbench/core.py`: generator, policies, metrics, and optional learned models.
- `src/conflictbench/runner.py`: reproducible configuration runner.
- `src/conflictbench/mosei.py`: optional loader for aligned CMU-MOSEI CSD files.
- `src/conflictbench/multibench.py`: loader and cache format for public MMSA descriptor pickles.
- `src/conflictbench/perception_candidate_audit.py`: strict Perception Test annotation audit.
- `src/conflictbench/perception_media_pilot.py`: training-only target and donor construction.
- `src/conflictbench/perception_nuisance_gate.py`: non-semantic edit-role detection and uncertainty.
- `src/conflictbench/perception_omni_gate.py`: frozen source-sufficiency inference and validation.
- `scripts/run_benchmark.py`: command-line entrypoint.
- `scripts/run_scalar_campaign.py`: exact 83-seed scalar campaign runner.
- `scripts/audit_perception_candidate_pool.py`: candidate-pool audit entrypoint.
- `scripts/build_perception_media_pilot.py`: frozen training-only pilot-index builder.
- `scripts/extract_perception_pilot_media.py`: selected-video extractor and stream verifier.
- `scripts/run_perception_nuisance_gate.py`: hash-bound automated nuisance-test entrypoint.
- `scripts/build_perception_nuisance_contract.py`: write-once nuisance-contract builder.
- `scripts/run_perception_omni_gate.py`: Qwen2.5-Omni source-sufficiency entrypoint.
- `scripts/download_mosei.py`: official CSD and fold acquisition helper.
- `scripts/cache_mosei.py`: one-time projected-score cache builder.
- `scripts/download_mmsa.py`: pinned processed-descriptor downloader.
- `scripts/cache_mmsa.py`: one-time MMSA projected-score cache builder.
- `configs/quickstart.json`: a small runnable configuration.
- `configs/mosei_example.json`: path-based MOSEI configuration template.
- `configs/mmsa_example.json`: path-based CMU-MOSI descriptor configuration template.
- `configs/campaign_acquisition.json`: bounded synthetic acquisition configuration.
- `configs/perception_test_mcqa_audit.json`: pinned annotation sources and eligibility rule.
- `configs/perception_test_media_pilot.json`: frozen training-only construction contract.
- `configs/perception_nuisance_gate.json`: frozen nuisance-test features, statistics, and gates.
- `configs/perception_omni_source_gate.json`: frozen source-sufficiency thresholds and model pins.
- `tests/`: checks for generator determinism, action semantics, and runner output.

## License

The ConflictBench software is MIT licensed; see `LICENSE`. Perception Test
annotations are not redistributed here. The audit-derived records use
[*Perception Test: A Diagnostic Benchmark for Multimodal Video Models*](https://openreview.net/forum?id=HYEGXFnPoq)
(Pătrăucean et al., 2023), Copyright 2022 DeepMind Technologies Limited. The
[pinned upstream repository](https://github.com/google-deepmind/perception_test/tree/3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf)
licenses non-software materials under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode); downstream
use must preserve that attribution and license notice.

CMU-MOSEI recordings, CSD files, and MMSA-derived descriptors are not
redistributed. Obtain CMU-MOSEI through the
[CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/4f2eadcd7e7b9e20e83b868cdad385b41830285c)
and processed descriptors through the
[MMSA repository](https://github.com/thuiar/MMSA/tree/a94e65d07fa1ae0d44e552390074b29b0898edfd),
then follow the dataset providers' terms. The MIT licenses in those linked
repositories cover their software and do not grant rights to the underlying
recordings or annotations. When the pinned checkpoint is absent from the
configured cache, the
optional Qwen2.5-Omni runner asks Transformers to fetch its model and processor.
Qwen2.5-Omni-3B is governed by the
[Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/blob/f75b40e3da2003cdd6e1829b1f420ca70797c34e/LICENSE),
which restricts use to non-commercial research; users must review and accept
those terms before enabling the optional runner.
