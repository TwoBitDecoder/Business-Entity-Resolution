import polars as pl
import pytest

from business_entity_resolution.hybrid_candidates import (
    combine_fuzzy_candidates,
    validate_hybrid_bound,
)


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema=["s1_id", "target_id", "name_similarity"],
        orient="row",
    )


def test_hybrid_union_deduplicates_and_preserves_signal_scores():
    name = _frame([("q1", "a", 0.9), ("q1", "b", 0.8)])
    address = _frame([("q1", "a", 0.7), ("q1", "c", 0.95)])
    out = combine_fuzzy_candidates(name, address, final_top_k=30)
    assert out.height == 3
    a = out.filter(pl.col("target_id") == "a").row(0, named=True)
    assert a["name_similarity"] == pytest.approx(0.9)
    assert a["address_similarity"] == pytest.approx(0.7)
    assert a["from_name"] is True
    assert a["from_address"] is True
    validate_hybrid_bound(out, 30)


def test_hybrid_prunes_deterministically_to_final_bound():
    name = _frame([("q", f"n{i}", 1.0 - i / 100) for i in range(20)])
    address = _frame([("q", f"a{i}", 0.99 - i / 100) for i in range(10)])
    out = combine_fuzzy_candidates(name, address, final_top_k=7)
    assert out.height == 7
    validate_hybrid_bound(out, 7)


def test_hybrid_overlap_does_not_consume_two_slots():
    name = _frame([("q", "same", 0.9), ("q", "n", 0.8)])
    address = _frame([("q", "same", 0.95), ("q", "a", 0.7)])
    out = combine_fuzzy_candidates(name, address, final_top_k=3)
    assert out.height == 3
    assert out.select(["s1_id", "target_id"]).n_unique() == 3


def test_hybrid_rejects_invalid_bound():
    empty = _frame([])
    with pytest.raises(ValueError, match="final_top_k"):
        combine_fuzzy_candidates(empty, empty, final_top_k=0)
