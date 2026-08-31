#!/usr/bin/env python3
"""
Script to extract top-K retrieval candidates to a directory structure.

For each query, creates a sub-folder containing:
- Symlinks to positives from the ILIAS core database
- Extracted images for distractors from YFCC100M tar files

Usage:
python scripts/extract_ilias_candidates.py \
    --retrieval_pkl retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000.pkl \
    --data_root data/ilias/download/ilias \
    --yfcc_root data/ilias/download/ilias/yfcc100m \
    --output_dir ilias_candidates_siglip_100M_top1k \
    --top_k 100
"""

import os
import argparse
import pickle as pk
import tarfile
import io
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image


def load_retrieval_results(retrieval_pkl):
    """Load retrieval results from pickle file."""
    print(f"Loading retrieval results from {retrieval_pkl}")
    with open(retrieval_pkl, 'rb') as f:
        data = pk.load(f)
    
    retrieval_candidates = data['retrieval_candidates']  # dict: query_id -> list of top-K indices
    query_ids = data['query_ids']  # array of query IDs (paths)
    db_ids = data['db_ids']  # array of database IDs (paths)
    k = data.get('k', len(list(retrieval_candidates.values())[0]))
    
    print(f"Loaded {len(retrieval_candidates)} queries with top-{k} candidates")
    print(f"Database size: {len(db_ids)}")
    
    return retrieval_candidates, query_ids, db_ids, k


def collect_tar_extractions(retrieval_candidates, query_ids, db_ids, top_k=None):
    """
    Collect all images that need to be extracted from each tar file.
    
    Returns:
        query_to_candidates: dict mapping query_id -> list of (rank, db_id, is_distractor)
        tar_to_images: dict mapping tar_file_path -> list of (image_hash, query_id, rank)
    """
    print("\nCollecting extraction requirements...")
    
    query_to_candidates = {}
    tar_to_images = defaultdict(list)
    
    for query_id in tqdm(query_ids, desc="Processing queries"):
        if query_id not in retrieval_candidates:
            continue
        
        candidate_indices = retrieval_candidates[query_id]
        if top_k is not None:
            candidate_indices = candidate_indices[:top_k]
        
        query_to_candidates[query_id] = []
        
        for rank, db_idx in enumerate(candidate_indices):
            db_id = db_ids[db_idx]
            
            # Check if this is a distractor (tar file reference)
            if '.tar/' in db_id:
                # Distractor format: "yfcc100m-NNNNNN.tar/hash"
                tar_filename, image_hash = db_id.split('.tar/', 1)
                tar_file = f"{tar_filename}.tar"
                
                # Record which tar file and image we need
                tar_to_images[tar_file].append((image_hash, query_id, rank))
                query_to_candidates[query_id].append((rank, db_id, True))  # is_distractor=True
            else:
                # Positive from ILIAS core
                query_to_candidates[query_id].append((rank, db_id, False))  # is_distractor=False
    
    print(f"Found {len(tar_to_images)} tar files to process")
    total_extractions = sum(len(images) for images in tar_to_images.values())
    print(f"Total distractor images to extract: {total_extractions}")
    
    return query_to_candidates, tar_to_images


def create_output_structure(output_dir, query_to_candidates, data_root, use_symlink=False):
    """
    Create output directory structure and symlinks/copies for positives.
    
    Args:
        use_symlink: If True, create symlinks for positives; if False, copy files
    
    Returns:
        query_dirs: dict mapping query_id -> output directory path
    """
    print("\nCreating output directory structure and symlinks...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    query_dirs = {}
    
    for query_id, candidates in tqdm(query_to_candidates.items(), desc="Creating directories"):
        # Create query directory
        # query_id format: "instance_name_000/query/Q000_00.jpg"
        query_name = query_id.replace('/', '_').replace('.jpg', '')
        query_dir = output_dir / query_name
        query_dir.mkdir(parents=True, exist_ok=True)
        query_dirs[query_id] = query_dir
        
        # Create symlinks for positives
        for rank, db_id, is_distractor in candidates:
            if not is_distractor:
                # Positive from ILIAS core - create symlink
                # db_id format: "instance_name_000/pos/P000_00.jpg"
                source_path = Path(data_root) / 'ilias_core' / db_id
                
                if not source_path.exists():
                    print(f"Warning: Source file not found: {source_path}")
                    continue
                
                # Create symlink or copy without rank prefix
                target_name = db_id.replace('/', '_')
                target_path = query_dir / target_name
                
                # Create symlink or copy based on use_symlink flag
                try:
                    if not target_path.exists():
                        if use_symlink:
                            os.symlink(source_path, target_path)
                        else:
                            import shutil
                            shutil.copy2(source_path, target_path)
                except Exception as e:
                    action = "symlink" if use_symlink else "copy"
                    print(f"Warning: Failed to {action} {target_path}: {e}")
    
    return query_dirs


def extract_distractors_from_tars(tar_to_images, query_dirs, yfcc_root):
    """
    Extract distractor images from tar files.
    Opens each tar file only once and extracts all required images.
    Skips images that already exist in the destination.
    """
    print("\nExtracting distractor images from tar files...")
    
    yfcc_root = Path(yfcc_root)
    
    # First pass: filter out images that already exist
    filtered_tar_to_images = defaultdict(list)
    total_images_before = 0
    total_images_after = 0
    
    for tar_file, images_to_extract in tar_to_images.items():
        total_images_before += len(images_to_extract)
        
        for image_hash, query_id, rank in images_to_extract:
            query_dir = query_dirs[query_id]
            output_name = f"{tar_file.replace('.tar', '')}_{image_hash}.jpg"
            output_path = query_dir / output_name
            
            # Only add to extraction list if destination doesn't exist
            if not output_path.exists():
                filtered_tar_to_images[tar_file].append((image_hash, query_id, rank))
                total_images_after += 1
    
    # Print statistics
    already_exist = total_images_before - total_images_after
    num_tars = len(filtered_tar_to_images)
    
    print(f"\n{'='*70}")
    print(f"Extraction Statistics:")
    print(f"  Total images needed: {total_images_before}")
    print(f"  Already exist (skipping): {already_exist}")
    print(f"  Will extract: {total_images_after} images from {num_tars} different tar files")
    print(f"{'='*70}\n")
    
    if total_images_after == 0:
        print("All images already exist. Nothing to extract.")
        return
    
    for tar_file, images_to_extract in tqdm(filtered_tar_to_images.items(), desc="Processing tar files"):
        tar_path = yfcc_root / tar_file
        
        if not tar_path.exists():
            print(f"Warning: Tar file not found: {tar_path}")
            continue
        
        # Open tar file once
        try:
            with tarfile.open(tar_path, 'r') as tar:
                # Build a mapping of hash -> member for fast lookup
                hash_to_member = {}
                for member in tar.getmembers():
                    # Extract hash from member name (last component)
                    # Member name might be something like "images/abc123..."
                    member_hash = member.name.split('/')[-1]
                    hash_to_member[member_hash] = member
                
                # Extract all required images
                for image_hash, query_id, rank in images_to_extract:
                    # Find the member
                    member = None
                    for hash_key, tar_member in hash_to_member.items():
                        if image_hash in hash_key:
                            member = tar_member
                            break
                    
                    if member is None:
                        print(f"Warning: Image {image_hash} not found in {tar_file}")
                        continue
                    
                    # Extract image
                    try:
                        f = tar.extractfile(member)
                        img = Image.open(io.BytesIO(f.read()))
                        
                        # Save to query directory
                        query_dir = query_dirs[query_id]
                        output_name = f"{tar_file.replace('.tar', '')}_{image_hash}.jpg"
                        output_path = query_dir / output_name
                        
                        img.save(output_path, 'JPEG')
                    except Exception as e:
                        print(f"Warning: Failed to extract {image_hash} from {tar_file}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error opening tar file {tar_path}: {e}")
            continue


def main(args):
    """
    Main function to extract top-K candidates for each query.
    
    Process:
    1. Load retrieval results from pickle file
    2. Collect all tar extraction requirements (which images from which tar files)
    3. Create output directory structure with query sub-folders
    4. Create symlinks for positives from ILIAS core
    5. Extract distractors from tar files (opening each tar only once)
    """
    # Load retrieval results
    retrieval_candidates, query_ids, db_ids, k = load_retrieval_results(args.retrieval_pkl)
    
    # Determine actual top_k to extract
    top_k = args.top_k if args.top_k is not None else k
    print(f"\nExtracting top-{top_k} candidates for each query")
    
    # Collect tar extraction requirements
    query_to_candidates, tar_to_images = collect_tar_extractions(
        retrieval_candidates, query_ids, db_ids, top_k
    )
    
    # Create output structure and symlinks/copies for positives
    query_dirs = create_output_structure(
        args.output_dir, query_to_candidates, args.data_root, use_symlink=args.symlink
    )
    
    # Extract distractors from tar files
    if args.yfcc_root and tar_to_images:
        extract_distractors_from_tars(tar_to_images, query_dirs, args.yfcc_root)
    elif tar_to_images:
        print("\nWarning: yfcc_root not provided, skipping distractor extraction")
    
    print(f"\n✓ Done! Extracted candidates to: {args.output_dir}")
    print(f"  Created {len(query_dirs)} query directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract top-K retrieval candidates to directory structure")
    parser.add_argument("--retrieval_pkl", type=str, required=True, help="Path to retrieval results pickle file")
    parser.add_argument("--data_root", type=str, required=True, help="Path to ILIAS data root (containing ilias_core/)")
    parser.add_argument("--yfcc_root", type=str, default=None, help="Path to YFCC100M tar files directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted candidates")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top candidates to extract (default: all from pickle)")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks for positives instead of copying files")
    
    args = parser.parse_args()
    main(args)
