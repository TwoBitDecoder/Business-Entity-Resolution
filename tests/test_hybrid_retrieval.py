import polars as pl

from business_entity_resolution.hybrid_retrieval import retrieve_hybrid_topk


def test_hybrid_adds_address_candidate_and_stays_bounded():
    q = pl.DataFrame({
        "entity_id": ["q1"],
        "name_compact": ["acmetrading"],
        "address_norm": ["12 mg road bengaluru"],
    })
    t = pl.DataFrame({
        "entity_id": ["namehit", "addresshit", "other"],
        "name_compact": ["acmetradng", "totallydifferent", "othercompany"],
        "address_norm": ["99 elsewhere", "12 mg road bengaluru", "77 nowhere"],
    })
    result = retrieve_hybrid_topk(q, t, name_k=1, address_k=1, final_k=2, n_jobs=1)
    assert result.height <= 2
    assert set(result["target_id"]) == {"namehit", "addresshit"}


def test_hybrid_deduplicates_same_target():
    q = pl.DataFrame({
        "entity_id": ["q1"],
        "name_compact": ["acme"],
        "address_norm": ["12 main road"],
    })
    t = pl.DataFrame({
        "entity_id": ["t1"],
        "name_compact": ["acme"],
        "address_norm": ["12 main road"],
    })
    result = retrieve_hybrid_topk(q, t, name_k=1, address_k=1, final_k=2, n_jobs=1)
    assert result.height == 1
    assert result["target_id"].to_list() == ["t1"]
