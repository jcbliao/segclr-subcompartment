# SegCLR subcompartment dataset curation

Target classes:

1. soma
2. dendrite
3. axon
4. astrocytic process

## Agreed labeling rules

- Manifest neuronal root IDs accept the Casey coarse classification as
  sufficient for this dataset.
- Manifest neuronal root IDs will supply dendrite labels. Their remaining
  neuronal processes can supply axon labels subject to the spatial exclusions.
- All thalamocortical cells contribute axon only; they never contribute
  dendrite labels.
- For every neuronal cell, only the largest connected component of its axon
  label is eligible to contribute axon training nodes. Connected components
  are computed on the source skeleton adjacency before applying the spatial
  node exclusions below. All smaller axon components are discarded.
- All manifest astrocytes contribute astrocytic-process labels only; they never
  contribute axon or dendrite labels.
- Axon, dendrite, and astrocytic-process training nodes must be more than
  10 micrometers from the nucleus point.
- As a global class-boundary rule, both endpoints of every skeleton edge whose
  endpoints have different class labels are excluded from training, regardless
  of the edge length. This rule applies to every class pair and to both distance
  variants.
- Transition regions between axon and dendrite are excluded using two dataset
  variants that will be compared during training:
  1. **Geodesic:** exclude axon and dendrite nodes whose shortest path along
     the weighted skeleton is within 5 micrometers of the opposite class.
  2. **Geometric:** exclude axon and dendrite nodes whose 3D Euclidean distance
     is within 5 micrometers of the opposite class.
- Soma samples are the segmentation-verified soma-center SegCLR embeddings.
- Use all applicable manifest cells; manual selection is no longer required.
- Microglia, oligodendrocytes, and OPCs supply none of these four classes.
- Exclude root `864691136364187874`, which has no soma-labelled synapses.

Distance thresholds must be applied in physical units using nanometers from
the source coordinates: `10_000 nm` and `5_000 nm`. Geodesic edge weights are
the Euclidean lengths of skeleton edges in nanometers. Both transition variants
must use identical selected cells, labels, splits, and all other filters so the
distance metric is the only experimental difference.

## Inventories

- `casey_coarse_types.csv`: coarse Casey types and candidate counts.
- `casey_coarse_candidates.csv`: candidates ordered by type, then decreasing
  Casey confidence, with root ID and existing metadata.
- `thalamocortical_roots.csv`: automatically axon-only roots.
- `astrocyte_candidates.csv`: candidates for manual astrocyte selection.
- `selected_dendrite_roots.txt`: one manually accepted neuronal root ID per line.
- `selected_astrocyte_roots.txt`: one manually selected astrocyte root ID per line.

Regenerate the inventories with `python3 curation/build_inventory.py`.
