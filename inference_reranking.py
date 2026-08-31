import datetime
import os
from os.path import join
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import argparse
import numpy as np

import opts
from util.commons import setup_logging, make_deterministic, load_checkpoint
from metrics.retrieval import evaluate_retrieval, print_retrieval_metrics
from models.foundyou import build_foundyou, FoundYou
from models.model_utils import compile_image_encoder
from util.promptable_utils import build_prompt_dict
from datasets import build_dataset
from datasets.transform_utils import ImageFolderDataset, IndexSampler, load_shot_episodes
from util.retrieval_utils import (
    collect_candidate_paths,
    exclude_supports_from_gt,
    filter_supports_from_rankings,
)

"""
PerMIR
python inference_reranking.py \
    --dataset_file permir \
    --checkpoint ./pretrain/foundyou.pth

ILIAS:
Note: Requires a pre-extracted candidates directory created by scripts/extract_ilias_candidates.py
python inference_reranking.py --dataset_file ilias \
    --candidates_dir ilias_candidates_siglip_100M_top1k \
    --retrieval_file retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000.pkl \
    --checkpoint ./pretrain/foundyou.pth \
    --top_n 20
"""

def main(args: argparse.Namespace):
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)

    dataset = build_dataset(args.dataset_file, args=args)
    if args.shots > 0 and dataset.benchmark != 'ilias':
        raise ValueError("N-shot reranking is supported only for ILIAS")
    if dataset.benchmark == 'permir':
        args.top_n = len(dataset.positive_ids)
    print(args)

    # Build model
    model = build_foundyou(args.model_config)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters())

    load_checkpoint(args.checkpoint, model)

    compile_image_encoder(model, args.top_n, args.batch_size, device)

    print(f"number of params: {n_parameters}")
    print('Start reranking inference')

    metrics = rerank(model, dataset, args)
    return metrics


def rerank(model: FoundYou, dataset, args: argparse.Namespace):
    """
    Rerank top-N candidates using FoundYou.
    For each query, loads top_n candidates and computes similarity scores.
    """
    print(f'Reranking on {args.dataset_file}')

    if dataset.benchmark == 'permir':
        query_ids_from_file = dataset.image_query_ids
        db_ids = dataset.positive_ids
        full_gallery = list(range(len(db_ids)))
        retrieval_candidates = {
            query_id: full_gallery.copy() for query_id in query_ids_from_file
        }
        has_initial_retrieval = False
        print(f"Full-gallery mode: reranking all {len(db_ids)} candidates per query")
    else:
        if not args.retrieval_file:
            raise ValueError("--retrieval_file is required unless --dataset_file is permir")
        print(f"Loading retrieval candidates from: {args.retrieval_file}")
        with open(args.retrieval_file, 'rb') as f:
            data = pickle.load(f)
        retrieval_candidates = data['retrieval_candidates']
        query_ids_from_file = data['query_ids']
        db_ids = data['db_ids']
        has_initial_retrieval = True

    shot_episodes = {}
    if args.shots > 0:
        shot_episodes = load_shot_episodes(args.shots_file)
        retrieval_candidates = filter_supports_from_rankings(
            retrieval_candidates=retrieval_candidates,
            db_ids=db_ids,
            shot_episodes=shot_episodes,
        )
        gt = exclude_supports_from_gt(dataset.gt, shot_episodes)
    else:
        gt = dataset.gt

    print(f"Loaded {len(retrieval_candidates)} queries with candidates")
    print(f"Database size: {len(db_ids)}")
    
    # Get ground truth and query mapping from dataset
    query_id_to_dataset_idx = dataset.get_query_id_mapping()
    valid_query_ids = set(query_id_to_dataset_idx.keys())
    
    model.eval()
    device = args.device
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)

    print(f"Number of folders with positives: {len(gt)}")
    
    initial_results = None
    if has_initial_retrieval:
        print("\nComputing initial retrieval metrics...")
        initial_results = evaluate_retrieval(
            query_ids=query_ids_from_file,
            retrieval_results=retrieval_candidates,
            db_ids=db_ids,
            gt=gt,
            valid_query_ids=valid_query_ids,
        )
        print_retrieval_metrics(initial_results, title="Initial Retrieval Results (Before Reranking)")
    
    # Rerank candidates
    reranked_results = {}

    # ---- Build a single persistent DataLoader for all candidate images ----
    print("Pre-collecting all candidate paths ...")
    all_candidate_paths, query_to_local_info = collect_candidate_paths(
        query_ids_from_file=query_ids_from_file,
        query_id_to_dataset_idx=query_id_to_dataset_idx,
        retrieval_candidates=retrieval_candidates,
        db_ids=db_ids,
        dataset=dataset,
        top_n=args.top_n,
    )
    print(f"Total unique candidate images: {len(all_candidate_paths)}")

    global_dataset = ImageFolderDataset(all_candidate_paths, transform=dataset.transform)
    sampler = IndexSampler()
    candidate_loader = DataLoader(
        global_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0, # persit. workers allowed if num_workers > 0
    )
    
    pbar = tqdm(query_ids_from_file, desc='Reranking queries')    
    for query_id in pbar:
        if query_id not in query_id_to_dataset_idx:
            continue
        
        query_dataset_idx = query_id_to_dataset_idx[query_id]
        # Load query without cropping
        query_sample = dataset[query_dataset_idx]
        query_img = query_sample['im_t'].to(device)  # [C, H, W]

        query_prompt = build_prompt_dict(query_sample['box'], 'box', device)
        reference_images = [query_img]
        reference_prompts = [query_prompt]

        # Add the n-shots references to memory.  The
        # complete episode pool has already been removed from the gallery/GT.
        support_ids = shot_episodes.get(query_id, [])[:max(0, args.shots - 1)]
        if support_ids:
            for support_sample in dataset.load_support_samples(support_ids, query_id):
                reference_images.append(support_sample['im_t'].to(device))
                reference_prompts.append(build_prompt_dict(support_sample['box'], 'box', device))

        with torch.no_grad(), autocast_ctx:
            reference_context = model.encode_references(reference_images, reference_prompts)

        # Get candidate database indices (top_n from retrieval)
        retrieved_db_indices, candidate_image_ids, global_indices = query_to_local_info[query_id]
        sampler.set_indices(global_indices)

        # Compute scores for each candidate
        candidate_scores = []
        
        for batch in candidate_loader:
            batch = batch.to(device)  # [B, C, H, W]
            with torch.no_grad(), autocast_ctx:
                scores = model.score_candidates(batch, reference_context)
            candidate_scores.extend(scores.cpu().tolist())
        
        # Sort candidates by score (descending)
        candidate_scores = np.array(candidate_scores)
        sorted_indices = np.argsort(-candidate_scores)  # Descending order
        
        # Reorder candidate image IDs
        reranked_image_ids = [candidate_image_ids[i] for i in sorted_indices]
        
        # Map back to original db_ids indices for remaining candidates
        reranked_db_indices = []
        for img_id in reranked_image_ids:
            # Find the original db_idx
            for db_idx in retrieved_db_indices:
                if db_ids[db_idx] == img_id:
                    reranked_db_indices.append(db_idx)
                    break
        
        # Add remaining candidates (not in top_n) at the end in original order
        remaining_db_indices = retrieval_candidates[query_id][args.top_n:]
        reranked_db_indices.extend(remaining_db_indices)
        
        reranked_results[query_id] = reranked_db_indices
    
    # Evaluate reranked results
    metrics_file = join(args.output_dir, f'reranked_metrics.pkl')
    final_results = evaluate_retrieval(
        query_ids=query_ids_from_file,
        retrieval_results=reranked_results,
        db_ids=db_ids,
        gt=gt,
        valid_query_ids=valid_query_ids,
    )
    print_retrieval_metrics(final_results, title="Reranking Results (After FoundYou Reranking)")
    metrics = {
        'recalls': final_results['recalls'],
        'mAP': final_results['mAP'],
    }
    if initial_results is not None:
        metrics['initial_mAP'] = initial_results['mAP']
        metrics['initial_recalls'] = initial_results['recalls']
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved metrics to: {metrics_file}")
    return metrics


if __name__ == "__main__":
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('FoundYou reranking script', parents=[opts.get_args_parser()])
    parser.add_argument('--model_config', default='configs/retrieval.yaml', help='FoundYou model configuration.')
    parser.add_argument('--candidates_dir', type=str, help='Path to pre-extracted candidates directory')
    parser.add_argument("--retrieval_file", type=str, default="", help="Retrieval shortlist; required except for PerMIR full-gallery evaluation")

    parser.add_argument("--top_n", type=int, default=10, help="Rerank top-n candidates.")
    parser.add_argument('--shots', type=int, default=0, help=(
            'Total number of query/reference images. 0 runs regular inference; 1 uses '
            'the controlled episode gallery with only the original query; N adds N-1 supports.'
        ),
    )
    parser.add_argument('--shots_file', type=str, help='Episode JSON used when --shots is greater than zero.',
        default='data/ilias/download/ilias/retrieval_shots_retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000_3shots_seed42.json',
    )

    args = parser.parse_args()
    name_exp = args.name_exp+"_"+str(datetime.datetime.now().strftime("%m%d_%H%M"))+"_"+str(random.randint(0,1000))
    args.output_dir = join(args.output_dir, name_exp)
    os.makedirs(args.output_dir)

    main(args)
