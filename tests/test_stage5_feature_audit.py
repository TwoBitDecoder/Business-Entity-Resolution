from business_entity_resolution.stage5_feature_audit import NUMERIC_FEATURES

def test_feature_audit_numeric_columns_are_unique():
    assert len(NUMERIC_FEATURES) == len(set(NUMERIC_FEATURES))
    assert "name_similarity" in NUMERIC_FEATURES
    assert "address_number_jaccard" in NUMERIC_FEATURES
