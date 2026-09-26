# Architecture

```text
S1 / S2 / S3 TSV
      |
      v
validation + normalization
      |
      v
multi-channel candidate generation
  | char TF-IDF
  | exact / high-confidence rules
  | optional dense retrieval
      |
      v
candidate union + deduplication
      |
      v
pairwise feature engineering
      |
      v
LightGBM pair scorer
      |
      v
precision-oriented threshold + grouping by S1
      |
      +--> candidate_pairs.tsv
      +--> matching_results.tsv
```

## Training
Ground truth is expanded into positive S1-to-S2/S3 pairs. Candidate-derived nonmatches
form hard negatives. Validation splits must be by Source 1 entity to avoid leakage.

## Inference invariant
Every predicted match must exist in the final candidate set supplied to the scorer.

## Compliance
The system is offline. External business databases, geocoding, registries, search engines,
or other identity-enrichment services are not pipeline inputs.
