import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import argparse

import opts
from util.commons import setup_logging, make_deterministic, load_checkpoint
from util.promptable_utils import build_prompt_dict
from models.foundyou import build_foundyou, FoundYou
from metrics.segmentation import BinarySegmentationEvaluator, BinarySegmentationMeter
from datasets import build_dataset

"""
python inference_pers_seg.py --dataset_file perseg \
    --checkpoint ./pretrain/foundyou.pth

python inference_pers_seg.py --dataset_file permis \
    --checkpoint ./pretrain/foundyou.pth

"""

def main(args: argparse.Namespace) -> dict:
    setup_logging(args.output_dir, console="info", rank=0)
    make_deterministic(args.seed)
    print(args)
    model = build_foundyou(args.model_config)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    load_checkpoint(args.checkpoint, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    metrics = eval_permis(model, args)
    return metrics


def eval_permis(model: FoundYou, args: argparse.Namespace) -> float:
    """
    Evaluate FoundYou on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    # load data
    ds = build_dataset(args.dataset_file, args=args)
    print(f'Evaluating {args.dataset_file}')
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    device = torch.device(args.device)
    evaluator = BinarySegmentationEvaluator()
    meter = BinarySegmentationMeter()
    pbar = tqdm(dataloader, ncols=100, desc='runn avg.', file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']
        class_name = batch['class_name'][0]

        reference_images = [image.to(device) for image in support_imgs[0]]
        reference_prompts = []
        for support_idx in range(support_masks.shape[1]):
            support_mask = support_masks[0:1, support_idx:support_idx + 1]
            reference_prompts.append(build_prompt_dict(support_mask, args.prompt, device))

        with torch.no_grad():
            reference_context = model.encode_references(reference_images, reference_prompts)
            pred_masks = model.segment_candidates(query_img.to(device), reference_context)

        pred_mask = (pred_masks.sigmoid() > 0.5).cpu()
        gt_mask = query_mask.cpu() > 0

        intersection, union, bound_inter, bound_union = evaluator.classify_prediction(pred_mask, gt_mask)
        meter.update(class_name, intersection.item(), union.item(), bound_inter.item(), bound_union.item())

        if (idx + 1) % 50 == 0:
            current_miou, _ = meter.compute()
            pbar.set_description(f"runn. avg = {current_miou * 100:.1f}")

    # Compute overall metrics
    mIoU, mbIoU = meter.compute()
        
    # Print only overall metrics to console
    print(f"mIoU = {mIoU * 100:.2f}%")
    print(f"mbIoU = {mbIoU * 100:.2f}%")
    return {
        'mIoU': mIoU,
        'mbIoU': mbIoU,
    }


if __name__ == "__main__":
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('FoundYou evaluation script', parents=[opts.get_args_parser()])
    parser.add_argument(
        '--model-config', '--model_config',
        dest='model_config', default='configs/pers_seg.yaml',
        help='FoundYou model configuration.',
    )
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    main(args)
