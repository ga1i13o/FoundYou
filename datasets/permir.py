r""" PerMiR few-shot segmentation dataset """
import glob
import os
from os.path import join
import numpy as np
from glob import glob
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List

from .transform_utils import mask_to_bbox


class DatasetPerMIR(Dataset):
    def __init__(self, datapath, transform):
        self.benchmark = 'permir'

        self.base_path = join(datapath, 'PerMIRS')
        self.transform = transform
        
        # Load ILIAS-style metadata
        image_query_ids = torch.load(join(self.base_path, 'permir_image_query_ids.pt'), weights_only=False)
        positive_ids = torch.load(join(self.base_path, 'permir_positive_ids.pt'), weights_only=False)
        gt = torch.load(join(self.base_path, 'permir_gt.pt'), weights_only=False)
        
        self.image_query_ids = image_query_ids  # List of query paths (strings)
        self.positive_ids = positive_ids  # List of positive paths (strings)
        self.gt = gt  # Dict: folder (instance) -> set of positive paths
        
        # Build all_paths list: queries + positives (database)
        self.query_paths = [join(self.base_path, q) for q in self.image_query_ids]
        self.db_paths = [join(self.base_path, p) for p in self.positive_ids]
        self.all_paths = self.query_paths + self.db_paths
        
        # Create ID mappings
        self.path_to_id = {path: idx for idx, path in enumerate(self.all_paths)}
        self.id_to_path = {idx: path for path, idx in self.path_to_id.items()}
        
        # Query and database indices in self.all_paths
        self.query_ids = list(range(len(self.query_paths)))
        self.database_ids = list(range(len(self.query_paths), len(self.all_paths)))
        
        # Build pos_ids dict for backward compatibility with index-based code
        # query_local_idx -> set of positive dataset indices
        self.pos_ids = {}
        for query_local_idx, query_path in enumerate(self.query_paths):
            query_id = self.image_query_ids[query_local_idx]
            folder = query_id.rsplit('/', 1)[0] if '/' in query_id else ''
            
            # Get all positive indices for this folder
            positive_indices = set()
            if folder in self.gt:
                for pos_id in self.gt[folder]:
                    pos_path = join(self.base_path, pos_id)
                    if pos_path in self.path_to_id:
                        positive_indices.add(self.path_to_id[pos_path])
            
            self.pos_ids[query_local_idx] = positive_indices
        
        self.n_queries = len(self.query_paths)
        
        print(f"PerMIR dataset loaded:")
        print(f"  Queries: {len(self.query_paths)}")
        print(f"  Positives (database): {len(self.db_paths)}")
        print(f"  Total images: {len(self.all_paths)}")
        print(f"  Folders with positives: {len(self.gt)}")
    
    def get_query_id_mapping(self):
        """Build mapping from query IDs (relative paths) to dataset indices.
        
        Returns:
            dict: Maps query_id (e.g., '95/0.jpg') to dataset index
        """
        query_id_to_dataset_idx = {}
        for query_local_idx, query_dataset_idx in enumerate(self.query_ids):
            query_path = self.all_paths[query_dataset_idx]
            # Extract relative path from full path
            query_rel_path = os.path.relpath(query_path, self.base_path)
            query_id_to_dataset_idx[query_rel_path] = query_dataset_idx
        return query_id_to_dataset_idx
    
    def get_candidate_paths(self, candidate_image_ids: List[str], query_id: str = None) -> List[str]:
        """Build absolute paths for candidate images.
        
        Args:
            candidate_image_ids: List of image IDs (relative paths)
            query_id: Query ID (not used for PerMIR, kept for API compatibility with ILIAS)
        
        Returns:
            List of absolute paths to candidate images
        """
        candidate_paths = []
        for img_id in candidate_image_ids:
            img_path = os.path.join(self.base_path, img_id)
            candidate_paths.append(img_path)
        return candidate_paths

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        im_t, mask, bbox, class_sample = self.load_frame(path)
        batch = {
            'im_t': im_t, 
            'mask': mask, 
            'box': bbox,
            'im_path': path,
            'class_id': torch.tensor(class_sample)
        }        
        return batch

    def load_frame(self, path):
        im_t = self.transform(Image.open(path).convert('RGB'))
        mask_file = join(os.path.dirname(path), "masks.npz.npy")
        masks_np = np.load(mask_file, allow_pickle=True)
        im_id = int(os.path.basename(path).split('.')[0])
        class_sample = int(os.path.basename(os.path.dirname(path)))
        mask = self.read_mask(np.array(list(masks_np[im_id].values()))[0])
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), im_t.size()[-2:], mode='nearest').squeeze()
        bbox = mask_to_bbox(mask)
        return im_t, mask, bbox, class_sample

    def read_mask(self, mask):
        mask = torch.tensor(mask)
        mask[mask == True] = 1
        mask[mask == False] = 0
        return mask
