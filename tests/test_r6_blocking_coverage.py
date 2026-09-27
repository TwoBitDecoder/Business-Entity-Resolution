import polars as pl

from business_entity_resolution.r6_blocking_coverage import _with_keys


def test_blocking_keys_are_deterministic():
    df = pl.DataFrame({
        "entity_id": ["x"],
        "name_compact": ["alphaco"],
        "address_numbers": ["12|34"],
    })
    got = _with_keys(df).row(0, named=True)
    assert got["name_prefix2"] == "al"
    assert got["name_prefix3"] == "alp"
    assert got["name_suffix2"] == "co"
    assert got["name_suffix3"] == "aco"
    assert got["address_first_number"] == "12"
