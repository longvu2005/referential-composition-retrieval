# Editing the research code

The YAML describes the experiment. Functions implement the experiment directly;
there is no registry, config inheritance, trainer framework or compatibility
wrapper for the old import paths.

## Where to make changes

| Change | Start here |
| --- | --- |
| Model architecture | `src/rcr/proposed/nn/model.py`, then the component in `nn/` |
| Loss terms and weights | `src/rcr/proposed/losses.py`, `configs/proposed.yaml` |
| Identity, semantic and coarse scores | `src/rcr/proposed/scores.py` |
| Training loop and optimizer | `src/rcr/proposed/train.py` |
| Candidate sampling and mining | `src/rcr/proposed/sampling.py` |
| Batch construction and prefetch | `src/rcr/proposed/batch.py` |
| Shortlist and reranking | `src/rcr/proposed/ranking.py` |
| Checkpoint loading and saved rankings | `src/rcr/proposed/retrieve.py` |
| Experiments and calibration | `src/rcr/proposed/experiments.py`, `configs/ablations/`, `configs/calibration.yaml` |
| DINO features/detections | `src/rcr/proposed/cache/dino.py` |
| FAFA crop features | `src/rcr/proposed/cache/fafa.py` |
| Cache stage selection and encoder provenance | `src/rcr/proposed/cache/build.py` |
| Read cached features | `src/rcr/proposed/cache/store.py` |
| Baseline implementations | `src/rcr/baselines/` |
| Split loading and gallery membership | `src/rcr/common/data.py` |
| Metrics and benchmark protocol | `src/rcr/evaluation/` |
| Annotation and dataset construction | `src/rcr/dataset/`, `scripts/data/`, `labelstudio/` |

`common/io.py` holds JSONL, image-path and result I/O. `common/runtime.py` holds
device/seed setup, flat YAML overrides and evaluation coordination. These are
small functions; the numerical model does not depend on scripts or experiments.

For training, read `scripts/run.py` → `proposed/train.py` → `losses.py` and
`nn/model.py`. For inference, read `proposed/retrieve.py` → `ranking.py` →
`scores.py` and the model. Cache extraction has a separate path through
`cache/build.py` → `cache/dino.py` or the native FAFA worker.

## Checks at the boundaries

- Dataset construction/audit validates full annotations. Loading retains ID
  uniqueness, disjoint query splits and query/seed membership checks.
- Building/reusing a cache compares gallery order, encoder configuration and
  source ID. Interrupted builds keep their resume manifest. Completed caches
  are skipped without scanning every feature file.
- Train/retrieve validate cache IDs, feature dimensions and checkpoint
  architecture. They do not inspect the encoder environment or FAFA YAML.
  Feature files are checked as they are loaded; a missing file raises its normal
  I/O error rather than triggering a rebuild.
- Numerical checks stop nonfinite losses/scores. Validation remains responsible
  for model/calibration selection; test results do not select settings.

No full model initialization hash is computed. Seeds, configs, cache provenance,
checkpoint hashes and split fingerprints still identify the experiment.

## Paths after the cleanup

| Previous path | Current path |
| --- | --- |
| `tools/methods/run.py` | `scripts/run.py` |
| `tools/methods/cache_fafa.py` | `scripts/cache_fafa.py` |
| `tools/dataset/` | `scripts/data/` |
| `scripts/phase*.bash` | `scripts/data/phase*.bash` |
| `configs/methods/*.yaml` | `configs/*.yaml` |
| `configs/calibration/joint.yaml` | `configs/calibration.yaml` |
| `rcr.methods.proposed` | `rcr.proposed` (components in `nn/` and `cache/`) |
| `rcr.methods.baselines` | `rcr.baselines` |
| `rcr.methods.common`, `rcr.utils` | `rcr.common` |
| `tests/methods/` | `tests/{proposed,baselines,common}/`, `tests/test_core.py` |

Update external notebooks/custom YAML paths to match. Previously saved run
configs may still point to the old FAFA YAML: adjust that path before using them
to build a cache. Existing feature files and checkpoint state dictionaries keep
their formats. Runtime `cache/`, `runs/`, `weights/`, dataset files and numerical
defaults are unchanged.
