"""
Metrics package for retrieval and segmentation evaluation.

- metrics.retrieval: Standard mAP@N and Recall@K for ILIAS and PerMIR
- metrics.segmentation: IoU / Jaccard Index for segmentation evaluation
"""

from metrics.retrieval import (
    evaluate_retrieval,
    print_retrieval_metrics,
)
