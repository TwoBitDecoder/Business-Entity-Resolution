from collections.abc import Iterable


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
