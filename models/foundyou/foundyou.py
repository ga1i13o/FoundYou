from typing import Any, Dict, List, Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.sam2.modeling.sam2_utils import preprocess_images
from models.sam2.modeling.sam2_base import SAM2Base 
from models.model_utils import BackboneOutput, DecoderOutput, ReferenceContext
from util.promptable_utils import rescale_prompt


class FoundYou(nn.Module):
    def __init__(self, sam: SAM2Base):
        super().__init__()
        self.sam = sam

    def encode_references(
        self,
        images: Sequence[torch.Tensor],
        prompts: Sequence[Dict[str, Any]],
    ) -> ReferenceContext:
        """Encode prompted ``[C, H, W]`` images into a retrieval context."""
        if len(images) == 0:
            raise ValueError("At least one reference image is required")
        if len(images) != len(prompts):
            raise ValueError("Reference images and prompts must have the same length")

        memory_bank = {}
        position_embedding = None
        feature_size = None

        for memory_idx, (image, prompt_spec) in enumerate(zip(images, prompts)):
            if image.ndim != 3:
                raise ValueError("Each reference image must have shape [C, H, W]")

            orig_size = tuple(image.shape[-2:])
            reference_feats = self._encode_images(image[None, None])
            prompt_type = prompt_spec['prompt_type']
            prompt = prompt_spec['prompt']
            frame_prompt = dict(prompt) if isinstance(prompt, dict) else prompt
            frame_prompt = rescale_prompt(frame_prompt, prompt_type, orig_size, self.sam.image_size)

            if prompt_type == 'mask':
                decoder_out = self.sam._use_mask_as_output(reference_feats, frame_prompt, 0)
            else:
                decoder_out = self._decode_reference(reference_feats, frame_prompt)

            memory_bank[memory_idx] = self._compute_memory_bank_dict(decoder_out, reference_feats, 0)
            if position_embedding is None:
                position_embedding = reference_feats.vision_pos_embeds[-1]
                feature_size = reference_feats.feat_sizes[-1]

        return ReferenceContext(memory_bank, position_embedding, feature_size)

    def score_candidates(
        self, images: torch.Tensor, context: ReferenceContext
    ) -> torch.Tensor:
        """Score a batch of ``[B, C, H, W]`` candidate images."""
        if images.ndim != 4:
            raise ValueError("Candidate images must have shape [B, C, H, W]")
        candidate_feats = self._encode_images(images[:, None]).vision_feats[-1]
        conditioned_feats = self._condition_candidates(candidate_feats, context)
        decoder_out = self.sam._forward_sam_heads_transformer_only(conditioned_feats)
        return decoder_out.object_score_logits.squeeze(-1)

    def segment_candidates(
        self, images: torch.Tensor, context: ReferenceContext
    ) -> torch.Tensor:
        """Segment a batch of candidates independently using fixed reference memory."""
        if images.ndim != 4:
            raise ValueError("Candidate images must have shape [B, C, H, W]")

        candidate_output = self._encode_images(images[:, None])
        candidate_feats = candidate_output.vision_feats
        conditioned_feats = self._condition_candidates(candidate_feats[-1], context)
        decoder_out = self.sam._forward_sam_heads(
            backbone_features=conditioned_feats,
            point_inputs=None,
            high_res_features=candidate_output.get_high_res_features(candidate_feats),
            multimask_output=True,
        )
        masks = F.interpolate(
            decoder_out.low_res_masks,
            size=images.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return masks[:, 0]

    def _encode_images(self, samples: torch.Tensor) -> BackboneOutput:
        """Extract multiscale backbone features from ``[B, T, C, H, W]`` images."""
        samples, _, _, orig_size = preprocess_images(samples, self.sam.image_size)
        return self._forward_backbone(samples, orig_size)

    def _condition_candidates(
        self, feats: torch.Tensor, context: ReferenceContext
    ) -> torch.Tensor:
        """Condition batched candidate features only on the fixed references."""
        B = feats.shape[1]
        pos = context.position_embedding.expand(-1, B, -1)

        num_memories = len(context.memory_bank)
        if num_memories == 0:
            raise RuntimeError("Candidate decoding requires at least one encoded reference")
        if set(context.memory_bank) != set(range(num_memories)):
            raise RuntimeError("Reference memory indices must be contiguous and start at zero")
        
        return self.sam._prepare_memory_conditioned_features(
            frame_idx=num_memories,
            current_vision_feats=[feats],
            current_vision_pos_embeds=[pos],
            feat_sizes=[context.feature_size],
            num_frames=num_memories + 1,
            memory_bank=context.memory_bank,
            expand_memory=True
        )

    def _decode_reference(
        self, backbone_out: BackboneOutput, prompt_input: Dict[str, torch.Tensor]
    ) -> DecoderOutput:
        """Decode the single prompted image used to build a reference memory."""
        current_vision_feats = backbone_out.get_current_feats(0)
        high_res_features = backbone_out.get_high_res_features(current_vision_feats)
        decoder_features = current_vision_feats[-1] + self.sam.no_mem_embed
        height, width = backbone_out.feat_sizes[-1]
        decoder_features = decoder_features.permute(1, 2, 0).reshape(1, -1, height, width)
        return self.sam._forward_sam_heads(
            backbone_features=decoder_features,
            point_inputs=prompt_input,
            high_res_features=high_res_features,
            multimask_output=False,
        )

    def _compute_memory_bank_dict(
        self, decoder_out: DecoderOutput, backbone_out: BackboneOutput, idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode current prediction into memory for later frames.

        Args:
            decoder_out: decoder output with high_res/low_res masks.
            backbone_out:  backbone features.
            idx: absolute idx.

        Returns:
            Memory entry dict.
        """
        current_vision_feats = backbone_out.get_current_feats(idx)
        feat_sizes = backbone_out.feat_sizes

        mem_feats, mem_pos = self.sam._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=decoder_out.high_res_masks,
            is_mask_from_pts=False,
        )
        return {
            "maskmem_features": mem_feats,
            "maskmem_pos_enc": mem_pos,
            "pred_masks": decoder_out.low_res_masks,
            "obj_ptr": decoder_out.obj_ptr,
        }

    def _forward_backbone(
        self, samples: torch.Tensor, orig_size: List[Tuple[int, int]]
    ) -> BackboneOutput:
        """
        Run SAM2 image encoder and prepare backbone features for decoding.

        Args:
            samples:  Tensor [B*T, C, H, W] after preprocessing.
            orig_size:   list of original frame sizes.

        Returns:
            BackboneOutput.
        """
        vis = self.sam.image_encoder.trunk(samples)
        feats, pos = self.sam.image_encoder.neck(vis)

        # discard lowest resolution
        feats, pos = feats[:-1], pos[:-1]

        feats[0] = self.sam.sam_mask_decoder.conv_s0(feats[0])
        feats[1] = self.sam.sam_mask_decoder.conv_s1(feats[1])

        bb = {
            "vision_features": feats[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": feats,
        }
        vision_feats, vision_pos, sizes = self.sam._prepare_backbone_features(bb)
        return BackboneOutput(orig_size, vision_feats, vision_pos, sizes)
