# Published acquisition comparator compatibility

Checked on 2026-09-17 against the paper sources and the ACO authors' code.

## Frozen benchmark contract

The CMU-MOSEI singleton study starts with exactly one observed modality. Its
first action is `answer`, `abstain`, or acquisition of one of the two missing
modalities. After one acquisition the episode terminates with `answer` or
`abstain`. The task head is frozen before router fitting. Evaluation uses both
offline exact query budgets and thresholds fitted on separate calibration
video groups.

## A2MT

Source: Jannik Kossen et al., [Active Acquisition for Multimodal Temporal
Data](https://arxiv.org/abs/2211.05039), Sections 2, 4, and Appendix D.

The published task makes a binary acquisition decision for each modality at
each time step of a temporal input. Its reward is terminal predictive loss
plus the sum of modality-by-time acquisition costs. The published large-data
method uses a Perceiver IO predictor and an actor-critic policy over repeated
temporal decisions. Its formal action sequence ends in a prediction step; it
does not define the benchmark's explicit abstention action.

A faithful port would require temporal inputs, repeated modality-by-time
decisions, the Perceiver IO policy and predictor, and the published reward.
Replacing those elements with three static projected scores, one acquisition,
and explicit abstention changes the objective, architecture, state, and action
space. The paper does not point to an official implementation. A2MT is
therefore not implemented under this frozen benchmark contract.

## Acquisition Conditioned Oracle

Sources: Michael Valancius, Maxwell Lennon, and Junier Oliva,
[Acquisition Conditioned Oracle for Nongreedy Active Feature
Acquisition](https://proceedings.mlr.press/v235/valancius24a.html), Sections
3.1--3.5, and the authors' AACO code at commit
[`3b2316661651699d11e904e9c5911c175e8b2fdc`](https://github.com/lupalab/aaco/tree/3b2316661651699d11e904e9c5911c175e8b2fdc),
especially
[`src/aaco_rollout.py`](https://github.com/lupalab/aaco/blob/3b2316661651699d11e904e9c5911c175e8b2fdc/src/aaco_rollout.py)
and
[`src/classifier.py`](https://github.com/lupalab/aaco/blob/3b2316661651699d11e904e9c5911c175e8b2fdc/src/classifier.py).

ACO acts sequentially. At each state it either acquires one unobserved feature
or terminates with a prediction. Its defining objective evaluates conditional
expected loss over future feature subsets; the approximate method estimates
those expectations with nearest neighbours and searches candidate subsets.
The authors' code performs iterative rollouts and stops when its selected
subset adds no feature. It has no explicit abstention action.

Under the one-acquisition cap, restricting ACO to singleton future subsets
turns its published non-greedy objective into a greedy expected-loss rule.
Keeping multi-feature future subsets while forcing termination after the first
action evaluates only a truncated prefix of the published policy. Allowing the
second acquisition changes the frozen benchmark action space. Adding
abstention also changes the terminal objective. ACO is therefore not
implemented under this frozen benchmark contract.

## Decision

Neither method preserves its published objective and action space here. The
v3 study stays benchmark-only. Its observed-only linear action-risk model is
an internal cost-sensitive baseline and is not presented as A2MT, ACO, or a
published-method reproduction. A future sequential benchmark may revisit ACO;
a future temporal raw-input benchmark may revisit A2MT.
