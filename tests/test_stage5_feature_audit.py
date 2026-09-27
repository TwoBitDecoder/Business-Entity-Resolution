from business_entity_resolution.stage5_feature_audit import NUMERIC_FEATURES

def test_feature_audit_numeric_columns_are_unique():
    assert len(NUMERIC_FEATURES) == len(set(NUMERIC_FEATURES))
    assert "name_similarity" in NUMERIC_FEATURES
    assert "address_number_jaccard" in NUMERIC_FEATURES


def test_benchmark_loader_keeps_feature_columns(tmp_path):
    import polars as pl
    from business_entity_resolution.r5_hybrid_benchmark import _load

    path = tmp_path / "records.parquet"
    pl.DataFrame({
        "entity_id": ["x"],
        "name_norm": ["alpha company"],
        "name_compact": ["alphacompany"],
        "address_norm": ["1 main road"],
        "unused": ["drop me"],
    }).write_parquet(path)
    got = _load(path, 1)
    assert got.columns == ["entity_id", "name_norm", "name_compact", "address_norm"]
