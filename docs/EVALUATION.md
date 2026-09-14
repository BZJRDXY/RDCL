# Paper Evaluation Settings

RDCL uses two distinct Top-p settings:

- **Top-3% evaluation cutoff:** used to report atom- and residue-level Recall and enrichment factor (EF) in the paper tables.
- **Top-15% rationale selection:** used by the model for rationale selection and by inference to mark high-confidence rationale edges.

For one complex with `N` ranked candidates, the nominal Top-3% rule selects

```text
k = ceil(0.03 * N)
```

candidates. Because `k` must be an integer, EF uses the actual selected fraction:

```text
Recall_i = selected true positives / all true positives
EF_i = Recall_i / (k_i / N_i)
```

Recall and EF are computed for each complex and then averaged separately. Therefore, mean EF is not generally equal to mean Recall divided by `0.03`.

The implementation is `recall_and_ef_at_top_p` in `rdcl/training/metrics.py`. Complexes without positive labels are excluded from both macro averages, matching the existing Recall implementation.
