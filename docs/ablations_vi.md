# Running Ablations

The fixed-weight identity A–D suite is documented in
[Identity balance ablations](identity_balance_ablation.md). It uses the selected
coarse/fine fusion weights 0.4/0.4 and does not rerun the older sweeps below.

First, update the paths in `configs/methods/proposed.yaml`. Run all commands from
the repository root in the proposed environment. All suites reuse the existing
visual cache.

## 1. Train and evaluate on validation

```bash
python tools/methods/run.py run --config configs/methods/proposed.yaml --train --splits val
```

With `C=24`, two available positives and `train.sampling.hard_fraction=0`, each
query receives 2 positives, 11 negatives containing all required identities,
and 11 random negatives. If a pool contains too few images,
random sampling fills the remaining slots. Candidates are unique. The query
image and all other Full Positives are excluded from the negative pool. Negatives
that conflict with positive labels from equivalent training queries are also
excluded. The original annotations and validation/test protocols remain unchanged.

The main YAML enables mining (`hard_fraction=0.3`). To run without mining, add
`--set train.sampling.hard_fraction=0 output.dir=runs/proposed_no_mining` to the
first command. To explicitly run with mining:

```bash
python tools/methods/run.py run --config configs/methods/proposed.yaml --train --splits val \
  --set train.sampling.hard_fraction=0.3 output.dir=runs/proposed_mined
```

After 1 warmup epoch, sampling uses 2 positives, 11 identity negatives, 6 coarse-mined
negatives, and 5 random negatives. Queries with only one positive use 11 identity,
6 mined and 6 random negatives. Each mined pool contains up to 100 highly ranked
negatives, sampled uniformly. Pools are refreshed after epochs 1, 3, 5, ... for
use in the following epoch. Mining runs only on train, skips fine reranking, and
retains only the required portion of each ranking. Actual sampling sources are
logged; random sampling may also happen to select images with the required identities.

Each epoch draws `N_train` queries with replacement, with case probability
proportional to `sqrt(N_case)`. All suites inherit the frozen BERT, 384-dimensional
model and dropout 0.1. Epoch checkpoints use
`0.5 * overall Full-mAP + 0.5 * macro-case Full-mAP` on validation; macro averages
the nonempty cases. These changes require retraining while reusing the visual cache.

## 2. Coarse retrieval and beta selection

```bash
python tools/methods/run.py ablate --config configs/ablations/coarse.yaml
```

All variants use the same checkpoint: identity-only, state-only, raw fusion,
z-score fusion, and a beta sweep. By default, beta is selected using
**CandidateRecall@500** on the complete validation split. The grid includes
beta=0. If you change the main Top-M budget, update the selection metric and
`candidate_ks` accordingly before inspecting the results.

Results are saved under `runs/ablations/coarse/`: `summary.csv`, `selection.json`,
`selected.yaml`, and the rankings/metrics for each variant.

- CandidateRecall@K: the fraction of Full Positives retained in the coarse Top-K.
- CandidateHit@K: the fraction of queries with at least one Full Positive in Top-K.
- Full-mAP: mean AP over the complete ranking, using Full Positive labels.
- ID-mAP: mean AP over the same ranking, using images containing all required
  identities as positives.
- R@K: a query-level hit rate, rather than the fraction of all positives retrieved.

## 3. Contributions of the state score and fine reranking

```bash
python tools/methods/run.py ablate --config configs/ablations/retrieval.yaml
```

The suite reads `selected.yaml` and keeps the checkpoint and beta selected on
validation fixed:

- `fine_identity_only` versus `fine_top500`: measures the contribution of state
  scoring to the shortlist, with fine reranking and M=500 in both variants. If
  the selected beta is 0, the two rankings should be identical; this is a valid
  result.
- `coarse_only` versus `fine_top500`: measures the contribution of fine reranking
  with the same shortlist.
- `fine_top100/500/1000`: measures the effect of the M budget with **fixed beta**.
  This experiment does not find a separate optimal beta for each M.

## 4. Contributions of the losses and sampler

```bash
python tools/methods/run.py ablate --config configs/ablations/loss.yaml
python tools/methods/run.py ablate --config configs/ablations/sampling.yaml
```

Each variant trains a new checkpoint using the same seed, training schedule, and
frozen cache. `loss.yaml` uses **identity-only retrieval for every variant**.
Here, `full` means the complete set of losses. Comparing it with
`without_state_loss` measures the contribution of auxiliary supervision to shared
components. Scores from untrained state projections are not used.

`sampling.yaml` compares natural versus balanced case draws, one versus up to two
positives, and random, identity/random, and identity/mined/random negatives.
`identity_mined_random` is the full new sampler. `natural_cases` and `one_positive`
each change one setting; the negative-source variants hold both those settings
fixed. All runs use `C=24` and the same state mask. This is a sampler ablation within
the updated pipeline; it does not reproduce the old state loss. The random
variant may produce fewer state pairs, so inspect the diagnostics as well.

## 5. Finalize settings before testing

```bash
python tools/methods/run.py run --config runs/ablations/coarse/selected.yaml \
  --splits test --set retrieval.rerank=true
```

Do not tune on test. New selections check both checkpoint and validation-data
fingerprints. If validation data or the checkpoint changes, rerun the sweep.
Older checkpoints remain readable if they contain a state branch, but retraining
is required to evaluate the updated supervision. The current test split contains
only 14 queries and should be expanded before drawing research conclusions.

To add an ablation, add an experiment with `name` and `overrides` to the appropriate
suite. `stage: inference` changes retrieval settings only; `stage: train` trains
a separate checkpoint. Sweeps take the Cartesian product of the value lists,
without requiring a registry or framework.
