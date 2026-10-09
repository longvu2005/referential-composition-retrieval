# Proposed v2 controls

`configs/ablations/shortlist.yaml` compares three explicit inference policies
using one frozen trained checkpoint and the same split gallery:

| Policy | Scoring |
| --- | --- |
| `coarse` | Grounded identity maximum, no transport or reasoner |
| `shortlist500` | Exact v2 fine score in coarse top-500, then coarse tail |
| `full` | Exact v2 fine score for every eligible gallery image |

```bash
python tools/run.py ablate --config configs/ablations/shortlist.yaml --splits val
```

Every output has complete-gallery rankings, self-exclusion, the same reviewed
positives and the unchanged AP denominator. CandidateRecall@M/CandidateHit@M
measure the coarse shortlist. `run.json` gives timing/memory where measurable.
The default M=500 is a starting value, not a measured optimum. To study other M,
add explicit entries and set `candidate_ks` no larger than that entry's M.
Choose settings on validation and freeze them before evaluating test.

Old state/fusion calibration and shared/dual or binding-term removal configs
were removed because those branches no longer exist. The v2 suite does not
silently load old checkpoints or pretend these controls are equivalent.
