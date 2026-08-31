from typing import Dict, Tuple
import torch

from datasets.transform_utils import mask_to_bbox
from util.commons import rescale_points, resize_mask


def build_prompt_dict(
    prompt: torch.Tensor, prompt_type: str, device: torch.device
) -> Dict:
    """Build one model prompt from a mask or a ``[x1, y1, x2, y2]`` box."""
    if prompt_type == 'box' and prompt.ndim == 1 and prompt.numel() == 4:
        prompt_inputs = {
            'point_coords': prompt.reshape(1, 2, 2).float().to(device),
            'point_labels': torch.tensor([[2, 3]], dtype=torch.int32, device=device),
        }
    elif prompt_type in ('point', 'box'):
        prompt_inputs = build_prompt_inputs(prompt, prompt_type, device)
    elif prompt_type == 'mask':
        prompt_inputs = prompt.to(device)
    else:
        raise NotImplementedError(f"Unsupported prompt type: {prompt_type}")

    return {
        'prompt_type': prompt_type,
        'prompt': prompt_inputs,
    }


def rescale_prompt(frame_prompt, prompt_type:str, orig_scale: Tuple[int, int], dest_scale: int):
    if prompt_type in ['point', 'box']:
        frame_prompt['point_coords'] = rescale_points(frame_prompt['point_coords'], orig_scale, (dest_scale, dest_scale))
    elif prompt_type == 'mask':
        frame_prompt = resize_mask(frame_prompt, dest_scale)
    else:
        raise NotImplementedError()
    
    return frame_prompt


def build_prompt_inputs(frame_gt: torch.Tensor, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Build prompt inputs (points/box) from a binary GT mask at image size.

    Args:
        frame_gt: Boolean tensor [1,1,IMG,IMG].
        prompt:   "point" | "box".
    Returns:
        Dict with keys:
        - "point_coords": [1, P, 2]
        - "point_labels": [1, P]
    """
    if prompt == 'point':
        support_mask = get_point_mask(frame_gt)

        if support_mask.sum().item() > 0:
            pos = torch.nonzero(support_mask[0, 0] == 1, as_tuple=False)[:, [1, 0]]  # (y,x)->(x,y)
            point_coords = pos.unsqueeze(0).float()  # [1, P, 2]
            point_labels = torch.ones(pos.shape[0], dtype=torch.int32).unsqueeze(0)  # [1, P]
        else:
            point_coords = torch.zeros(1, 1, 2)
            point_labels = torch.ones(1, 1, dtype=torch.int32)

        return {"point_coords": point_coords.to(device), "point_labels": point_labels.to(device)}

    # box
    box = mask_to_bbox(frame_gt[0, 0])
    if box is None:
        box = torch.zeros(4, dtype=torch.float32)
    box_coords = box.reshape(1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
    return {"point_coords": box_coords.to(device), "point_labels": box_labels.to(device)}


def get_point_mask(mask, max_points=20):
    """
    Returns:
        Point_mask: up to 20 randomly selected foreground points.
        If a mask is empty, it's Point_mask will be all zero.
    """
    h, w = mask.shape[-2:]
    if mask.sum().item() == 0:
        return torch.zeros(1, 1, h, w).to(mask.device)
    max_points = min(max_points, mask.sum().item())
    num_points = max_points

    view_mask = mask[0,0].view(-1)
    non_zero_idx = view_mask.nonzero()[:, 0]  # get non-zero index of mask
    selected_idx = torch.randperm(len(non_zero_idx))[:num_points]  # select id
    non_zero_idx = non_zero_idx[selected_idx]  # select non-zero index
    rand_mask = torch.zeros(view_mask.shape).to(mask.device)  # init rand mask
    rand_mask[non_zero_idx] = 1  # get one place to zero
    rand_mask = rand_mask.reshape(h, w).unsqueeze(0).unsqueeze(0)
    return rand_mask
