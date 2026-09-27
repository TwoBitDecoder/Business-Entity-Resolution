from business_entity_resolution.r8_source_split_retrieval import _retrieve_one_source
import polars as pl

def test_source_split_retrieval_respects_local_bound():
    q = pl.DataFrame({
        "entity_id": ["q1"],
        "name_compact": ["alpha"],
        "address_norm": ["1 main"],
    })
    t = pl.DataFrame({
        "entity_id": ["a", "b"],
        "name_compact": ["alpha", "beta"],
        "address_norm": ["1 main", "2 road"],
    })
    got, seconds = _retrieve_one_source(q, t, name_k=1, address_k=1)
    assert got.height <= 2
    assert seconds >= 0
