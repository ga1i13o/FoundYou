"""
Segmentation evaluation metrics (mIoU, Boundary IoU).
"""
import cv2
import numpy as np
import torch


class BinarySegmentationMeter:
    """Accumulates binary foreground/background statistics by evaluation group."""

    def __init__(self):
        self.groups = {}
        self.boundary_iou_sum = 0.0
        self.boundary_iou_count = 0

    def update(self, group, intersection, union, bound_inter, bound_union):
        if group not in self.groups:
            self.groups[group] = {
                "intersection": 0.0,
                "union": 0.0,
            }
        self.groups[group]["intersection"] += float(intersection)
        self.groups[group]["union"] += float(union)
        boundary_iou = float(bound_inter) / (float(bound_union) + 1e-10)
        self.boundary_iou_sum += boundary_iou
        self.boundary_iou_count += 1

    def compute(self):
        ious = []

        for group, stats in self.groups.items():
            ious.append(stats["intersection"] / (stats["union"] + 1e-10))

        if len(ious) == 0:
            return 0.0, 0.0

        miou = sum(ious) / len(ious)
        mbiou = self.boundary_iou_sum / max(self.boundary_iou_count, 1)
        return miou, mbiou


class BinarySegmentationEvaluator:
    """Computes IoU and Boundary IoU statistics for binary masks."""

    @staticmethod
    def classify_prediction(pred_mask, gt_mask, void_pixels=None, dilation_ratio=0.02):
        pred = torch.as_tensor(pred_mask).bool()
        gt = torch.as_tensor(gt_mask).bool()

        assert pred.shape == gt.shape, \
            f'Prediction({pred.shape}) and ground truth({gt.shape}) dimensions do not match.'

        if void_pixels is not None:
            valid = ~torch.as_tensor(void_pixels).bool()
            assert valid.shape == gt.shape, \
                f'Ground truth({gt.shape}) and void pixels({valid.shape}) dimensions do not match.'
        else:
            valid = torch.ones_like(gt, dtype=torch.bool)

        intersection = (pred & gt & valid).sum()
        union = ((pred | gt) & valid).sum()
        bound_inter, bound_union = BinarySegmentationEvaluator.classify_boundary_prediction(
            pred, gt, valid, dilation_ratio=dilation_ratio,
        )

        return intersection, union, bound_inter, bound_union

    @staticmethod
    def classify_boundary_prediction(pred, gt, valid, dilation_ratio=0.02):
        pred_np = pred.cpu().numpy().astype(np.uint8)
        gt_np = gt.cpu().numpy().astype(np.uint8)
        valid_np = valid.cpu().numpy().astype(bool)

        bound_inter = 0
        bound_union = 0

        for idx in np.ndindex(pred_np.shape[:-2]):
            pred_mask = pred_np[idx]
            gt_mask = gt_np[idx]
            valid_mask = valid_np[idx]

            h, w = gt_mask.shape
            img_diag = np.sqrt(h ** 2 + w ** 2)
            dilation = max(1, int(round(dilation_ratio * img_diag)))

            gt_boundary = _mask_to_boundary(gt_mask, dilation)
            pred_boundary = _mask_to_boundary(pred_mask, dilation)

            bound_inter += np.sum(((gt_boundary * pred_boundary) > 0) & valid_mask)
            bound_union += np.sum(((gt_boundary + pred_boundary) > 0) & valid_mask)

        device = pred.device
        return (
            torch.tensor(bound_inter, device=device),
            torch.tensor(bound_union, device=device),
        )


def _mask_to_boundary(mask, dilation):
    """Convert a binary mask to its boundary mask.

    Follows the official Boundary IoU implementation (Cheng et al., CVPR 2021).
    Pads the mask so that object regions touching the image border are also
    treated as boundary.

    Arguments:
        mask      (ndarray): 2-D binary mask (uint8).
        dilation  (int):     number of erosion iterations (boundary width).

    Return:
        boundary (ndarray): binary boundary mask of the same shape.
    """
    h, w = mask.shape
    # Pad so edges touching the image border are counted as boundary
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    return mask - mask_erode
