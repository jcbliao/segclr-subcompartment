# Same-subcompartment pair filter

`pair_filter.py` runs an exact comparison against the existing historical-v117
pair audit. It uses the five independent six-class classifiers and the completed
historical prediction artifacts described in [README.md](README.md).

The analysis retains a candidate pair if its two endpoints have the same argmax
subcompartment class. Each fold is evaluated independently. All six classes are
eligible; no confidence threshold, majority vote, or fold average is applied.

The common cohort consists of 1,924 reference neurons. Another 165 neurons in the
original audit lack complete historical skeleton coverage and are explicitly
excluded from every geometry. Glial reference cells are excluded from this
neuronal summary, but glial material merged into historical neuronal roots still
contributes to the original pair-error proxy.

Three geometries are compared at the original 150 µm cutoff:

* L2 endpoints with chunkedgraph path distances and L2 predictions.
* The same L2 endpoints with skeleton path distances and L2 predictions.
* Distinct occupied skeleton vertices with native skeleton predictions.

The graph itself is unchanged. Filtering endpoints does not delete paths through
vertices with different labels. Distinct L2 endpoints collapsed onto one skeleton
vertex remain distinct and retain their own labels; identical endpoints are
excluded. Native skeleton endpoints use the original supervoxel-count-weighted
membership aggregation.

For a reference neuron, endpoint membership is its fraction of the endpoint's
supervoxels. An unordered pair contributes correct weight `p_i * p_j` and
incorrect weight `p_i * (1-p_j) + p_j * (1-p_i)`. This preserves the existing
reference-membership error proxy; outside material is not necessarily assigned
to another independently verified neuron.

All-pair, uniform-distance, and exponential-distance policies use the original
root/bin weights, fixed **before** subcompartment filtering. Thus the result
answers what happens when pairs are removed from the existing distribution.
It does not rebalance the surviving pairs into a new distribution. Pooled error
is `sum(incorrect) / sum(correct + incorrect)` over reference neurons. These are
per-neuron exposures, not deduplicated unique cohort pairs. Retention is reported
using the same denominator and weights. Per-neuron rates and class summaries are
also saved; zero retained denominators yield null rates.

The run lives at:

```
/orcd/scratch/orcd/013/jcbliao/segclr_pair_audit/v117_subcompartment_filter_20261108
```

It contains `plan.json`, `cells.json`, `excluded_cells.json`, frozen source and
hashes, Slurm job records, root label arrays, per-neuron bin histograms,
`cell_scores.csv`, `pooled_scores.csv`, `cohort_comparison.csv`, and `summary.json`.
Only `summary.json` with `complete: true` establishes a finished cohort result.

Every neuron must reproduce the existing unfiltered correct/incorrect histogram
in all three geometries before its filtered result is accepted. Synthetic tests
compare the implementation to exhaustive pair enumeration, including fractional
membership, disconnected vertices, collapsed endpoints, and compact label dtype
overflow. Membership and geometry reads use existing local caches only; an
uncached API request fails explicitly.

The cluster workflow has separate `prepare`, `labels`, `worker`, and `summarize`
commands. Label extraction verifies score-file hashes and exact native-node
coverage. CPU scoring workers claim neurons from a shared filesystem pool and
write atomic completion receipts. `summarize` refuses to declare success while
any selected neuron is missing.
