import polars as pl

from business_entity_resolution.r5_residual_analysis import exact_pair_flags


def test_exact_pair_flags_require_nonempty_equal_values():
    q = pl.DataFrame({
        "entity_id": ["q1", "q2"],
        "name_norm": ["alpha ltd", ""],
        "name_compact": ["alphaltd", ""],
        "address_norm": ["10 market road", ""],
    })
    t = pl.DataFrame({
        "entity_id": ["t1"],
        "name_norm": ["alpha ltd"],
        "name_compact": ["alphaltd"],
        "address_norm": ["10 market road"],
    })
    out = exact_pair_flags(q, t)
    q1 = out.filter(pl.col("s1_id") == "q1").row(0, named=True)
    q2 = out.filter(pl.col("s1_id") == "q2").row(0, named=True)
    assert q1["exact_name"] and q1["exact_compact_name"] and q1["exact_address"]
    assert not q2["exact_name"]
    assert not q2["exact_compact_name"]
    assert not q2["exact_address"]
