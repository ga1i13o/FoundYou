from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import einops
import torch
from torch import Tensor


def compile_image_encoder(model, top_n: int, batch_size: int, device: torch.device):
    """Compile and warm up the image encoder for inference batch sizes."""
    print('Compiling backbone trunk with torch.compile (reduce-overhead)...')
    model.sam.image_encoder.trunk = torch.compile(
        model.sam.image_encoder.trunk, mode='reduce-overhead'
    )

    remainder = top_n % batch_size
    warmup_batch_sizes = sorted(
        {1, batch_size} | ({remainder} if remainder else set())
    )
    height, width = 1024, 1024
    print(f"Warmup: capturing graphs for batch sizes {warmup_batch_sizes} ...")
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)
    for warmup_batch_size in warmup_batch_sizes:
        dummy = torch.randn(
            warmup_batch_size, 1, 3, height, width, device=device
        )
        with torch.no_grad(), autocast_ctx:
            model._encode_images(dummy)
    torch.cuda.synchronize()
    del dummy
    print("Warmup complete.")



@dataclass
class DecoderOutput:
    """
    Container for decoder outputs (logits, masks, and auxiliary tensors).
    Fields default to None; set by the decoder as available.
    """
    low_res_masks: Optional[Tensor] = None
    high_res_masks: Optional[Tensor] = None
    obj_ptr: Optional[Tensor] = None
    pix_feat_with_mem: Optional[Tensor] = None
    masks: Optional[Tensor] = None
    object_score_logits: Optional[Tensor] = None
    # Additional fields for multimask output
    low_res_multimasks: Optional[Tensor] = None  # [B, 3, H, W] when multimask_output=True
    high_res_multimasks: Optional[Tensor] = None  # [B, 3, H, W] when multimask_output=True
    ious: Optional[Tensor] = None  # [B, 3] IoU scores for each mask

    def __post_init__(self):
        # Use multimasks if available, otherwise use single mask
        if self.low_res_multimasks is not None:
            self.masks = self.low_res_multimasks
        else:
            self.masks = self.low_res_masks


@dataclass
class BackboneOutput:
    """
    Container for backbone features across a clip.

    Attributes:
        orig_size: List of original sizes [(H, W)] for each frame (length B*T).
        vision_feats: List of feature tensors per FPN level.
        vision_pos_embeds: List of positional encodings per level.
        feat_sizes: List of (H, W) per FPN level.
    """
    orig_size: List[Tuple[int, int]]
    vision_feats: List[Tensor]
    vision_pos_embeds: List[Tensor]
    feat_sizes: List[Tuple[int, int]]

    def get_current_feats(self, idx: int) -> List[Tensor]:
        """
        Slice features for the flat frame index `idx` (0 <= idx < B*T).

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        return [x[:, idx:idx + 1, :] for x in self.vision_feats]

    def get_current_pos_embeds(self, idx: int) -> List[Tensor]:
        """
        Slice positional encodings for the flat frame index `idx`.

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        return [x[:, idx:idx + 1, :] for x in self.vision_pos_embeds]

    def get_current_feats_x16(self, idx: int) -> Tensor:
        """
        Slice features for the flat frame index `idx` (0 <= idx < B*T).

        Returns:
            List of tensors, one per level, sliced as [:, idx:idx+1, :].
        """
        vision_feats_16 = self.vision_feats[-1][:, idx:idx + 1, :]
        vision_feats_16 = einops.rearrange(vision_feats_16, '(h w) b c -> b c h w', h=self.feat_sizes[-1][0])
        return vision_feats_16

    def get_high_res_features(self, current_vision_feats: List[Tensor]) -> List[Tensor]:
        """
        Reformat high-resolution backbone features for the decoder.
        Operates on all levels except the last.

        Args:
            current_vision_feats: list of per-level tensors for the current frame.

        Returns:
            List of tensors shaped [B, C, H, W] for the high-resolution levels.
        """
        high_res = []
        for x, s in zip(current_vision_feats[:-1], self.feat_sizes[:-1]):
            # [HW, B, C] -> [B, C, H, W]
            x_perm = x.permute(1, 2, 0).contiguous()
            x_bchw = x_perm.view(x.size(1), x.size(2), *s)
            high_res.append(x_bchw)
        return high_res


@dataclass
class ReferenceContext:
    """Opaque reference state used to score retrieval candidates."""

    memory_bank: Dict[int, Dict[str, Tensor]]
    position_embedding: Tensor
    feature_size: Tuple[int, int]
