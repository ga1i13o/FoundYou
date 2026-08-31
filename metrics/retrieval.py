"""Retrieval metrics for ILIAS and PerMIR."""
import numpy as np
from typing import Dict, List


def evaluate_retrieval(
    query_ids: List[str],
    retrieval_results: Dict[str, List[int]],
    db_ids: np.ndarray,
    gt: Dict[str, set],
    valid_query_ids: set = None,
    k_values: List[int] = [1, 5, 10, 20, 50, 100]
) -> Dict:
    """Compute mAP and Recall@K from ranked database indices.

    Args:
        query_ids: Query IDs to evaluate.
        retrieval_results: Ranked database indices for each query.
        db_ids: Database identifiers indexed by ``retrieval_results``.
        gt: Per-query or per-instance sets of positive database IDs.
        valid_query_ids: Optional subset of query IDs to evaluate.
        k_values: Recall cutoffs.

    Returns:
        A dictionary containing ``mAP``, ``recalls``, ``aps``, and
        ``num_queries``.
    """
    # Normalize instance-level GT and database indices to per-query image IDs.
    query_ground_truth = {}
    query_retrieval_ids = {}

    for query_id in query_ids:
        if valid_query_ids is not None and query_id not in valid_query_ids:
            continue

        if query_id in gt:
            gt_entry = gt[query_id]
        else:
            instance = query_id.split('/')[0]
            if instance not in gt:
                continue
            gt_entry = gt[instance]

        if query_id in retrieval_results:
            retrieved_db_indices = retrieval_results[query_id]
            query_ground_truth[query_id] = gt_entry
            query_retrieval_ids[query_id] = [db_ids[idx] for idx in retrieved_db_indices]

    recalls_count = {k: 0 for k in k_values}
    aps = []

    for query_id, positive_ids in query_ground_truth.items():
        # Empty-positive queries remain in all metric denominators with AP 0.
        if len(positive_ids) == 0:
            aps.append(0.0)
            continue

        retrieved_ids = query_retrieval_ids[query_id]
        num_correct = 0
        sum_precisions = 0.0
        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id in positive_ids:
                num_correct += 1
                sum_precisions += num_correct / rank

        num_retrieved = len(retrieved_ids)
        denominator = min(num_retrieved, len(positive_ids)) if num_retrieved > 0 else len(positive_ids)
        aps.append(sum_precisions / denominator if denominator > 0 else 0.0)

        for k in k_values:
            if k <= num_retrieved and any(
                positive_id in retrieved_ids[:k] for positive_id in positive_ids
            ):
                recalls_count[k] += 1

    num_queries = len(aps)
    return {
        'mAP': np.mean(aps) if aps else 0.0,
        'recalls': {
            k: count / num_queries if num_queries > 0 else 0.0
            for k, count in recalls_count.items()
        },
        'aps': aps,
        'num_queries': num_queries,
    }


def print_retrieval_metrics(results: Dict, title: str = "Retrieval Results"):
    """
    Print retrieval metrics in a formatted way.
    
    Args:
        results: Dictionary from evaluate_retrieval()
        title: Title to display
    """
    print("\n" + "="*50)
    print(f"{title}:")
    print("="*50)
    print(f"Queries evaluated: {results['num_queries']}")
    print(f"mAP: {results['mAP']:.4f}")
    print("-"*50)
    for k in sorted(results['recalls'].keys()):
        print(f"Recall@{k:3d}: {results['recalls'][k]:.4f}")
    print("="*50 + "\n")
