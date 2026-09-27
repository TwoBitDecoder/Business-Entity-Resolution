# Residual retrieval

The exact-rule benchmark retrieved only about 30% of labelled positive pairs,
so classifier work is deferred.

Run:

```bash
python -m business_entity_resolution.residual_profiler
```

This profiles the positives missed by exact blocking using three cheap,
country-scoped diagnostics:

- first six compact-name characters;
- first four compact-name characters;
- numeric tokens extracted from addresses.

These are diagnostic rules, not the final production blocker. Their purpose is
to measure whether the residual positives retain cheap lexical structure and
to quantify the candidate explosion caused by coarse keys.

The output is `artifacts/residual_profile.json`.

If prefix candidate volume is excessive (expected for common prefixes), the
next implementation should use partitioned sparse character n-gram top-K
retrieval rather than materializing prefix cross-products.
