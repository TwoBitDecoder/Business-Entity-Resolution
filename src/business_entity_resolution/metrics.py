"""Entity-level decision and exact macro F-beta evaluation."""
from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def fbeta_for_entity(truth: Iterable[str], predicted: Iterable[str], beta: float = 0.5) -> float:
    truth_set, pred_set = set(truth), set(predicted)
    if not truth_set:
        return 1.0 if not pred_set else 0.0
    tp = len(truth_set & pred_set)
    if not pred_set or tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(truth_set)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_fbeta(rows: Iterable[tuple[Iterable[str], Iterable[str]]], beta: float = 0.5) -> float:
    scores = [fbeta_for_entity(t, p, beta) for t, p in rows]
    if not scores:
        raise ValueError("Cannot score an empty evaluation set")
    return sum(scores) / len(scores)


def entity_macro_fbeta(
    s1_ids: Iterable[str],
    truth_pairs: pl.DataFrame,
    scored_pairs: pl.DataFrame,
    *,
    threshold: float,
    beta: float = 0.5,
) -> float:
    """Score every supplied S1 exactly once, including true singletons."""
    ids = list(dict.fromkeys(s1_ids))
    if not ids:
        raise ValueError("Cannot score an empty S1 evaluation set")
    required_truth={"s1_id","target_id"}
    required_score={"s1_id","target_id","match_probability"}
    if missing:=required_truth-set(truth_pairs.columns):
        raise ValueError(f"truth_pairs missing columns: {sorted(missing)}")
    if missing:=required_score-set(scored_pairs.columns):
        raise ValueError(f"scored_pairs missing columns: {sorted(missing)}")

    id_set=set(ids)
    truth={x: set() for x in ids}
    pred={x: set() for x in ids}
    for s1,target in truth_pairs.select("s1_id","target_id").iter_rows():
        if s1 in id_set:
            truth[s1].add(target)
    for s1,target,p in scored_pairs.select("s1_id","target_id","match_probability").iter_rows():
        if s1 in id_set and p >= threshold:
            pred[s1].add(target)
    return macro_fbeta(((truth[x],pred[x]) for x in ids),beta=beta)
