import os
import sys
import argparse
import numpy as np
import pickle as pk
import h5py
import faiss
import time
import gc
from os.path import join
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from metrics.retrieval import evaluate_retrieval, print_retrieval_metrics
from util.retrieval_utils import load_features_hdf5


"""
Script to perform k-NN search using precomputed ILIAS features for image retrieval.

python scripts/retrieval_precomputed_feats.py \
    --model_name vit_large_patch16_siglip_384.webli \
    --total_distractors 100 \
    --k 1000 \
    --batch_search 10
"""

ROOT_FEATURES_PATH = 'data/ilias/download/ilias/features/{model_name}'

# Backward-compatible alias
load_features = load_features_hdf5


def load_database_features(positive_hdf5, distractor_hdf5=None, total_distractors=0):
    """
    Load database features from positive and distractor HDF5 files.
    
    Args:
        positive_hdf5: Path to positive features HDF5 file
        distractor_hdf5: Template path for distractor files (with {idx} placeholder)
        total_distractors: Number of distractor files to load
        
    Returns:
        db_features: Concatenated database features
        db_ids: Concatenated database IDs
    """
    print("\n> Loading database features...")
    
    # Load positive features
    print(f"  Loading positives from {positive_hdf5}")
    positive_feat, positive_ids = load_features(positive_hdf5)
    print(f"  Loaded {len(positive_ids)} positives")
    
    # Initialize lists with positives
    db_features_list = [positive_feat]
    db_ids_list = [positive_ids]
    
    # Load distractor features
    if distractor_hdf5 is not None and total_distractors > 0:
        print(f"  Loading {total_distractors} distractor files...")
        for i in tqdm(range(total_distractors), desc="Loading distractors"):
            distractor_file = distractor_hdf5.format(idx=f"{i:05d}")
            if not os.path.exists(distractor_file):
                print(f"Warning: Distractor file {distractor_file} not found, skipping.")
                continue
            feats, ids = load_features(distractor_file)
            db_features_list.append(feats)
            db_ids_list.append(ids)
    
    # Concatenate all features
    db_features = np.concatenate(db_features_list, axis=0)
    db_ids = np.concatenate(db_ids_list, axis=0)
    
    print(f"  Total database size: {len(db_ids)} items")
    return db_features, db_ids


def perform_search(query_features, db_features, k):
    """
    Perform k-NN search using FAISS.
    
    Args:
        query_features: Query feature matrix [N_queries, D]
        db_features: Database feature matrix [N_db, D]
        k: Number of nearest neighbors
        
    Returns:
        ranks: Indices of top-K candidates for each query [N_queries, K]
    """
    print(f"\n> Performing k-NN search for {query_features.shape[0]} queries in database of {db_features.shape[0]} items...")
    
    # Normalize features
    query_features = query_features.astype(np.float32)
    db_features = db_features.astype(np.float32)
    
    faiss.normalize_L2(query_features)
    faiss.normalize_L2(db_features)
    
    # Perform search
    try:
        gpu_res = faiss.StandardGpuResources()
        _, ranks = faiss.knn_gpu(
            gpu_res, 
            query_features, 
            db_features, 
            k, 
            metric=faiss.METRIC_INNER_PRODUCT
        )
    except Exception as e:
        print(f"GPU search failed: {e}. Falling back to CPU.")
    
    _, ranks = faiss.knn(
        query_features, 
        db_features, 
        k, 
        metric=faiss.METRIC_INNER_PRODUCT
    )
    
    return ranks


def load_database_batch(positive_hdf5, distractor_hdf5, batch_start, batch_end, total_distractors):
    """
    Load a batch of database features.
    
    Args:
        positive_hdf5: Path to positive features HDF5 file
        distractor_hdf5: Template path for distractor files
        batch_start: Starting distractor file index (0-indexed)
        batch_end: Ending distractor file index (exclusive)
        total_distractors: Total number of distractor files available
        
    Returns:
        batch_features: Features for this batch
        batch_ids: IDs for this batch
        batch_offset: Global offset for indices in this batch
    """
    db_features_list = []
    db_ids_list = []
    batch_offset = 0
    
    # Include positives only in first batch
    if batch_start == 0:
        print(f"  Loading positives from {positive_hdf5}")
        positive_feat, positive_ids = load_features(positive_hdf5)
        db_features_list.append(positive_feat)
        db_ids_list.append(positive_ids)
        print(f"  Loaded {len(positive_ids)} positives")
        batch_offset = 0  # First batch starts at index 0
    else:
        # Calculate offset from positives + previous distractor files
        with h5py.File(positive_hdf5, 'r') as f:
            positive_count = len(f['index'][:])
        batch_offset = positive_count + batch_start * 1_000_000  # Each distractor file has ~1M features
    
    # Load distractor files for this batch
    actual_end = min(batch_end, total_distractors)
    print(f"  Loading distractor files {batch_start} to {actual_end-1}...")
    for i in range(batch_start, actual_end):
        distractor_file = distractor_hdf5.format(idx=f"{i:05d}")
        if not os.path.exists(distractor_file):
            print(f"Warning: Distractor file {distractor_file} not found, skipping.")
            continue
        feats, ids = load_features(distractor_file)
        db_features_list.append(feats)
        db_ids_list.append(ids)
    
    if not db_features_list:
        return None, None, batch_offset
    
    batch_features = np.concatenate(db_features_list, axis=0)
    batch_ids = np.concatenate(db_ids_list, axis=0)
    
    return batch_features, batch_ids, batch_offset


def perform_batched_search(query_features, positive_hdf5, distractor_hdf5, total_distractors, k, batch_size_millions):
    """
    Perform k-NN search in batches to reduce memory usage.
    
    Args:
        query_features: Query feature matrix [N_queries, D]
        positive_hdf5: Path to positive features HDF5 file
        distractor_hdf5: Template path for distractor files
        total_distractors: Total number of distractor files
        k: Number of nearest neighbors
        batch_size_millions: Number of millions of features per batch
        
    Returns:
        global_ranks: Global indices of top-K candidates for each query [N_queries, K]
        db_ids: All database IDs (accumulated across batches)
    """
    print(f"\n> Performing batched k-NN search with batch size = {batch_size_millions}M features")
    
    num_queries = query_features.shape[0]
    
    # Initialize: track top-k distances and indices for each query
    # Use large initial distances (will be replaced)
    global_top_distances = np.full((num_queries, k), np.inf, dtype=np.float32)
    global_top_indices = np.full((num_queries, k), -1, dtype=np.int64)
    
    # Normalize query features once
    query_features = query_features.astype(np.float32)
    faiss.normalize_L2(query_features)
    
    # Accumulate all db_ids for final output
    all_db_ids = []
    
    # Process in batches
    num_batches = (total_distractors + batch_size_millions - 1) // batch_size_millions + 1  # +1 for positives
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size_millions
        batch_end = batch_start + batch_size_millions
        
        # Load batch features
        batch_features, batch_ids, batch_offset = load_database_batch(
            positive_hdf5, distractor_hdf5, batch_start, batch_end, total_distractors
        )
        
        if batch_features is None:
            print(f"  Batch {batch_idx}: No features loaded, skipping.")
            continue
        
        all_db_ids.append(batch_ids)
        
        print(f"  Batch {batch_idx}: Loaded {len(batch_ids)} features (offset={batch_offset})")
        
        # Normalize batch features
        batch_features = batch_features.astype(np.float32)
        faiss.normalize_L2(batch_features)
        
        # Perform search on this batch
        try:
            gpu_res = faiss.StandardGpuResources()
            batch_distances, batch_ranks = faiss.knn_gpu(
                gpu_res, 
                query_features, 
                batch_features, 
                k, 
                metric=faiss.METRIC_INNER_PRODUCT
            )
        except Exception as e:
            print(f"  GPU search failed: {e}. Falling back to CPU for this batch.")
            batch_distances, batch_ranks = faiss.knn(
                query_features, 
                batch_features, 
                k, 
                metric=faiss.METRIC_INNER_PRODUCT
            )
        
        # Convert to distances (FAISS returns similarities for INNER_PRODUCT)
        # Higher similarity = lower distance, so negate
        batch_distances = -batch_distances
        
        # Adjust indices to global indices
        batch_ranks_global = batch_ranks + batch_offset
        
        # Merge with global top-k
        for query_idx in range(num_queries):
            # Combine current global top-k with new batch top-k
            combined_distances = np.concatenate([global_top_distances[query_idx], batch_distances[query_idx]])
            combined_indices = np.concatenate([global_top_indices[query_idx], batch_ranks_global[query_idx]])
            
            # Sort by distance and keep top-k
            sorted_indices = np.argsort(combined_distances)[:k]
            global_top_distances[query_idx] = combined_distances[sorted_indices]
            global_top_indices[query_idx] = combined_indices[sorted_indices]
        
        # Free memory
        del batch_features, batch_distances, batch_ranks, batch_ranks_global
        gc.collect()
        
        print(f"  Batch {batch_idx}: Merged results, memory freed")
    
    # Concatenate all db_ids
    db_ids = np.concatenate(all_db_ids, axis=0)
    
    print(f"\n> Batched search completed. Total database size: {len(db_ids)} items")
    
    return global_top_indices, db_ids


def main(args):
    """
    Main function to perform k-NN search with precomputed ILIAS features.
    
    This function:
      - Loads precomputed query, positive, and distractor features from HDF5 files
      - Performs FAISS k-NN search
      - Saves top-K candidate indices for each query (for reranking)
    """
    if args.k > 1000:
        print("\nWarning: FAISS GPU does not support k values >1000. Switching to CPU.\n")
    root_path = ROOT_FEATURES_PATH.format(model_name=args.model_name)
    query_hdf5 = join(root_path, 'features_image_queries.hdf5')
    positive_hdf5 = join(root_path, 'features_positives.hdf5')
    distractor_hdf5 = join(root_path, 'features_distractors_{idx}.hdf5')
    output_file = f"retrieval_candidates_{args.model_name}_{args.total_distractors}M_top{args.k}{args.output_suffix}.pkl"

    # Load query features
    print(f"\n> Loading query features from {query_hdf5}")
    query_features, query_ids = load_features(query_hdf5)
    print(f"  Loaded {len(query_ids)} queries with dimension {query_features.shape[1]}")
    
    # Perform k-NN search (batched or full)
    start_time = time.time()
    
    if args.batch_search is not None:
        # Batched search to reduce memory usage
        ranks, db_ids = perform_batched_search(
            query_features,
            positive_hdf5,
            distractor_hdf5,
            args.total_distractors,
            args.k,
            args.batch_search
        )
    else:
        # Full search (load all features at once)
        db_features, db_ids = load_database_features(
            positive_hdf5,
            distractor_hdf5,
            args.total_distractors
        )
        ranks = perform_search(query_features, db_features, args.k)
        # Free memory after search
        del db_features
        gc.collect()
    
    search_time = time.time() - start_time
    print(f"\n> k-NN search completed in {search_time:.2f} seconds ({search_time/60:.2f} minutes)")
    
    # Build retrieval candidates dictionary
    # Format: {query_id: [list of top-K database indices]}
    retrieval_candidates = {}
    for i, query_id in enumerate(query_ids):
        retrieval_candidates[query_id] = ranks[i].tolist()
    
    # Save results  
    print(f"\n> Saving retrieval candidates to {output_file}")
    with open(output_file, 'wb') as f:
        results = {
            'retrieval_candidates': retrieval_candidates,
            'query_ids': query_ids,
            'db_ids': db_ids,
            'k': args.k
        }
        pk.dump(results, f)
    
    print(f"\n> Done! Retrieved top-{args.k} candidates for {len(query_ids)} queries.")
    print(f"  Results saved to: {output_file}")
    
    # Evaluate retrieval results if ground truth is available
    if args.evaluate:
        print("\n> Evaluating retrieval results...")
        try:
            # Load ground truth from ILIAS dataset structure
            from datasets import build_dataset
            
            # Create minimal args for dataset loading
            class MinimalArgs:
                def __init__(self):
                    self.data_root = 'data'
                    self.img_size = 1024
                    self.candidates_dir = None
            
            dataset_args = MinimalArgs()
            dataset = build_dataset('ilias', args=dataset_args)
            
            # Evaluate using unified function
            eval_results = evaluate_retrieval(
                query_ids=query_ids,
                retrieval_results=retrieval_candidates,
                db_ids=db_ids,
                gt=dataset.gt,
                valid_query_ids=set(dataset.get_query_id_mapping().keys())
            )
            print_retrieval_metrics(eval_results, title="ILIAS Retrieval Results")
            
        except Exception as e:
            print(f"Warning: Could not evaluate results: {e}")
            print("Skipping evaluation step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-NN search with precomputed ILIAS features for retrieval")
    parser.add_argument("--model_name", type=str, default='vit_large_patch16_siglip_384.webli', help="Model to use")
    parser.add_argument("--total_distractors", type=int, default=0, help="Number of distractor HDF5 files to load (e.g., 100 for ILIAS)")
    parser.add_argument("--k", type=int, default=100, help="Number of top candidates to retrieve for each query")
    parser.add_argument("--output_suffix", type=str, default='', help="Path to output pickle file for retrieval candidates")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate retrieval results using ground truth")
    parser.add_argument("--batch_search", type=int, default=None, help="Batch size in millions for memory-efficient search (e.g., 10 for 10M per batch). If None, loads all features at once.")
    args = parser.parse_args()
    main(args)
