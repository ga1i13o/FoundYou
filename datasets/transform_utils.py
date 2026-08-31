import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from PIL import ImageFile

# Avoid crashing for truncated (corrupted) images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_shot_episodes(shots_file: str) -> Dict[str, List[str]]:
    """Load a reranking episode manifest."""
    with open(shots_file, "r") as f:
        payload = json.load(f)
    episodes = payload['episodes']
    print(f"Loaded support episodes for {len(episodes)} queries from {shots_file}")
    return episodes


def load_box(
    box_file: str,
    original_size: Tuple[int, int],
    transformed_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load an ``x y width height`` box and return scaled ``x1 y1 x2 y2`` coordinates."""
    with open(box_file, 'r') as file:
        line = file.readline().strip()
    if not line:
        raise ValueError(f'No bounding box found in {box_file}')

    x, y, width, height = map(float, line.split())
    original_height, original_width = original_size
    if transformed_size is None:
        transformed_size = original_size
    transformed_height, transformed_width = transformed_size
    scale_x = transformed_width / original_width
    scale_y = transformed_height / original_height

    x_scaled = int(x * scale_x)
    y_scaled = int(y * scale_y)
    x_min = max(0, x_scaled)
    y_min = max(0, y_scaled)
    x_max = min(transformed_width - 1, x_scaled + int(width * scale_x))
    y_max = min(transformed_height - 1, y_scaled + int(height * scale_y))
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


class ImageFolderDataset(Dataset):
    """Simple dataset for loading images from a list of paths with transforms.
    
    Useful for efficient loading of candidate images in reranking with DataLoader.
    """
    def __init__(self, image_paths: List[str], transform=None):
        """
        Args:
            image_paths: List of absolute paths to images
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image


class IndexSampler(Sampler):
    """Resettable sampler that yields indices set by the main process.

    Useful with persistent_workers=True to reuse DataLoader workers
    across different subsets of a global dataset.
    """
    def __init__(self):
        self.indices = []

    def set_indices(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    

def mask_to_bbox(mask):
    """
    Derive a bounding box from a boolean mask.
    
    Args:
        mask: Boolean numpy array or torch tensor of shape (H, W)
        
    Returns:
        bbox: Tensor [x_min, y_min, x_max, y_max] or None if mask is empty
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(bool)
    
    # Find rows and columns where mask is True
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None  # Empty mask
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
