__all__ = ['foundyou', 'adapter']


import py3_wget
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

from models.foundyou.foundyou import FoundYou
from util.path_utils import SAM2_PATHS_CONFIG, SAM2_WEIGHTS_URL


def build_foundyou(config_path: str) -> FoundYou:
    model_config = OmegaConf.load(config_path)
    sam2_version = model_config.sam2_version
    adaptformer_stages = list(model_config.adaptformer_stages)
    channel_factor = model_config.channel_factor
    obj_score = model_config.obj_score

    assert sam2_version in SAM2_PATHS_CONFIG.keys(), f'wrong argument sam2_version: {sam2_version}'
    
    sam2_weights, sam2_config = SAM2_PATHS_CONFIG[sam2_version]
    if not os.path.isfile(sam2_weights):
        print(f"Downloading SAM2-{sam2_version}")
        py3_wget.download_file(SAM2_WEIGHTS_URL[sam2_version], sam2_weights)

    with initialize(version_base=None, config_path=".", job_name="test_app"):
        cfg = compose(config_name=sam2_config, overrides=[
            f"++model.image_encoder.trunk.adaptformer_stages={adaptformer_stages}",
            f"++model.image_encoder.trunk.adapt_dim={channel_factor}",
            f"++model.use_retrieval_decoder={True}",
        ])

        OmegaConf.resolve(cfg)
        cfg.model.pred_obj_scores = obj_score
        cfg.model.pred_obj_scores_mlp = obj_score
        cfg.model.fixed_no_obj_ptr = obj_score
        sam = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(sam2_weights, map_location="cpu", weights_only=False)["model"]
    sam.load_state_dict(state_dict, strict=False)

    decoder1_state = sam.sam_mask_decoder.state_dict()
    sam.retrieval_decoder.load_state_dict(decoder1_state)

    model = FoundYou(sam=sam)

    return model
