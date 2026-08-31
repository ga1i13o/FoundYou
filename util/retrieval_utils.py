
from typing import Dict, List, Tuple

import h5py


def load_features_hdf5(file_path):
    """Load features and IDs from an HDF5 file."""
    if not h5py.is_hdf5(file_path):
        raise ValueError(f"File {file_path} is not a valid HDF5 file.")

    with h5py.File(file_path, 'r') as f:
        features = f['features'][:]
        ids = f['index'][:].astype(str)
    return features, ids


def filter_supports_from_rankings(
    retrieval_candidates: Dict[str, List[int]],
    db_ids,
    shot_episodes: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """Remove each episode's complete support pool from its gallery ranking."""
    if not shot_episodes:
        return retrieval_candidates

    filtered = {}
    for query_id, ranking in retrieval_candidates.items():
        reserved_ids = set(shot_episodes.get(query_id, []))
        if reserved_ids:
            ranking = [db_idx for db_idx in ranking if db_ids[db_idx] not in reserved_ids]
        filtered[query_id] = ranking
    return filtered


def exclude_supports_from_gt(
    gt: Dict,
    shot_episodes: Dict[str, List[str]],
) -> Dict:
    """Build per-query GT entries with the complete support pool excluded."""
    if not shot_episodes:
        return gt

    filtered_gt = dict(gt)
    for query_id, support_ids in shot_episodes.items():
        if query_id in gt:
            gt_entry = gt[query_id]
        else:
            instance_id = query_id.split('/')[0]
            if instance_id not in gt:
                continue
            gt_entry = gt[instance_id]

        reserved_ids = set(support_ids)
        filtered_gt[query_id] = set(gt_entry) - reserved_ids
    return filtered_gt


def collect_candidate_paths(
    query_ids_from_file: List[str],
    query_id_to_dataset_idx: Dict,
    retrieval_candidates: Dict,
    db_ids,
    dataset,
    top_n: int,
) -> Tuple[List[str], Dict[str, Tuple]]:
    """
    Pre-collect all unique candidate image paths across queries.

    Args:
        query_ids_from_file: ordered list of query IDs.
        query_id_to_dataset_idx: maps valid query_id -> dataset index.
        retrieval_candidates: maps query_id -> list of retrieved db indices.
        db_ids: array of database image IDs.
        dataset: dataset object with .get_candidate_paths().
        top_n: number of top candidates per query.

    Returns:
        (all_candidate_paths, query_to_local_info)
        where all_candidate_paths is a deduplicated list of image paths, and
        query_to_local_info maps query_id -> (retrieved_db_indices, candidate_image_ids, global_indices).
    """
    all_candidate_paths: List[str] = []
    path_to_global_idx: Dict[str, int] = {}
    query_to_local_info: Dict[str, Tuple] = {}

    for query_id in query_ids_from_file:
        if query_id not in query_id_to_dataset_idx:
            continue
        retrieved_db_indices = retrieval_candidates[query_id][:top_n]
        candidate_image_ids = [db_ids[db_idx] for db_idx in retrieved_db_indices]
        candidate_paths = dataset.get_candidate_paths(candidate_image_ids, query_id)

        global_indices = []
        for p in candidate_paths:
            if p not in path_to_global_idx:
                path_to_global_idx[p] = len(all_candidate_paths)
                all_candidate_paths.append(p)
            global_indices.append(path_to_global_idx[p])

        query_to_local_info[query_id] = (retrieved_db_indices, candidate_image_ids, global_indices)

    return all_candidate_paths, query_to_local_info
