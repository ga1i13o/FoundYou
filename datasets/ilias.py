"""ILIAS dataset for instance retrieval with reranking"""
import os
from os.path import join
import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from typing import List

from .transform_utils import load_box


class DatasetILIAS(Dataset):
    def __init__(self, datapath, transform, candidates_dir=None):
        """
        Args:
            datapath: Path to ILIAS dataset directory
            transform: Image transforms
            candidates_dir: Path to pre-extracted candidates directory (required for distractor loading)
        """
        self.benchmark = 'ilias'
        self.datapath = join(datapath, 'ilias/download/ilias')
        self.transform = transform
        self.candidates_dir = candidates_dir
        
        # Load query IDs from image_query_ids.txt
        image_query_info = np.loadtxt(
            join(self.datapath, 'image_ids', 'image_query_ids.txt'), 
            dtype=str, 
            delimiter=','
        )
        self.image_query_ids = sorted(image_query_info[:, 0].tolist())
        
        # Load positive IDs from positive_ids.txt
        positive_info = np.loadtxt(
            join(self.datapath, 'image_ids', 'positive_ids.txt'),
            dtype=str,
            delimiter=','
        )
        self.positive_ids = sorted(positive_info[:, 0].tolist())
        
        # Build ground truth: instance -> set of positive image paths
        self.gt = {}
        for pos_id in self.positive_ids:
            instance = pos_id.split('/')[0]
            if instance not in self.gt:
                self.gt[instance] = set()
            self.gt[instance].add(pos_id)
        
        # Build all_paths list: queries + positives (database)
        self.query_paths = [join(self.datapath, 'ilias_core', q) for q in self.image_query_ids]
        self.db_paths = [join(self.datapath, 'ilias_core', p) for p in self.positive_ids]
        self.all_paths = self.query_paths + self.db_paths
        
        # Create ID mappings
        self.path_to_id = {path: idx for idx, path in enumerate(self.all_paths)}
        self.id_to_path = {idx: path for path, idx in self.path_to_id.items()}
        
        # Query and database indices in self.all_paths
        self.query_ids = list(range(len(self.query_paths)))
        self.database_ids = list(range(len(self.query_paths), len(self.all_paths)))
        
        # Build pos_ids dict: query_local_idx -> set of positive dataset indices
        self.pos_ids = {}
        for query_local_idx, query_path in enumerate(self.query_paths):
            query_id = self.image_query_ids[query_local_idx]
            instance = query_id.split('/')[0]
            
            # Get all positive indices for this instance
            positive_indices = set()
            if instance in self.gt:
                for pos_id in self.gt[instance]:
                    pos_path = join(self.datapath, 'ilias_core', pos_id)
                    if pos_path in self.path_to_id:
                        positive_indices.add(self.path_to_id[pos_path])
            
            self.pos_ids[query_local_idx] = positive_indices
        
        self.n_queries = len(self.query_paths)
        
        print(f"ILIAS dataset loaded:")
        print(f"  Queries: {len(self.query_paths)}")
        print(f"  Positives (database): {len(self.db_paths)}")
        print(f"  Total images: {len(self.all_paths)}")
        print(f"  Instances with positives: {len([p for p in self.pos_ids.values() if len(p) > 0])}")
    
    def get_query_id_mapping(self):
        """Build mapping from query IDs (relative paths) to dataset indices.
        
        Returns:
            dict: Maps query_id (e.g., 'instance/query/Q000_00.jpg') to dataset index
        """
        query_id_to_dataset_idx = {}
        for query_local_idx, query_dataset_idx in enumerate(self.query_ids):
            query_path = self.all_paths[query_dataset_idx]
            # Extract relative path: instance/query/Q000_00.jpg
            query_rel_path = '/'.join(query_path.split('/')[-3:])
            query_id_to_dataset_idx[query_rel_path] = query_dataset_idx
        return query_id_to_dataset_idx
    
    def get_candidate_paths(self, candidate_image_ids: List[str], query_id: str) -> List[str]:
        """Build absolute paths for candidate images (positives and distractors).
        
        Args:
            candidate_image_ids: List of image IDs (either ILIAS positive paths or distractor IDs)
            query_id: Query ID for constructing distractor paths (e.g., 'instance/query/Q000_00.jpg')
        
        Returns:
            List of absolute paths to candidate images
        """
        candidate_paths = []
        for img_id in candidate_image_ids:
            if '.tar/' in img_id:
                # Distractor from extracted directory
                tar_filename, image_hash = img_id.split('.tar/', 1)
                query_folder = query_id.replace('/', '_').replace('.jpg', '')
                img_path = os.path.join(self.candidates_dir, query_folder, f"{tar_filename}_{image_hash}.jpg")
            else:
                # Positive from ILIAS core
                img_path = os.path.join(self.datapath, 'ilias_core', img_id)
            candidate_paths.append(img_path)
        return candidate_paths

    def load_support_samples(self, support_ids: List[str], query_id: str) -> List[dict]:
        """Load N-shot reference images and their image-coordinate boxes."""
        samples = []
        for path in self.get_candidate_paths(support_ids, query_id):
            with Image.open(path) as image:
                im_t = self.transform(image.convert('RGB'))
            height, width = im_t.shape[-2:]
            # Support shots have no box annotation, so prompt the full image.
            box = torch.tensor([0, 0, width - 1, height - 1], dtype=torch.float32)
            samples.append({'im_t': im_t, 'box': box, 'im_path': path})
        return samples

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, idx):
        path = self.all_paths[idx]
        im_t, mask, bbox = self.load_frame(path)
        batch = {
            'im_t': im_t,
            'mask': mask,
            'box': bbox,
            'im_path': path,
        }
        return batch

    def load_frame(self, path, query_id=None):
        """Load image and bounding box for ILIAS
        
        Args:
            path: Image path or distractor ID
            query_id: Query ID (required for loading distractors from candidates_dir)
        """
        # Check if this is a distractor (YFCC100M image)
        # Distractors have format: "yfcc100m-NNNNNN.tar/hash"
        if '.tar/' in path:
            # This is a distractor ID - load from pre-extracted directory
            img = self._load_from_extracted_dir(path, query_id)
            im_t = self.transform(img.convert('RGB'))
            # Distractors don't have bboxes - use full image
            H, W = im_t.shape[-2:]
            bbox = torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32)
            mask = torch.ones((H, W), dtype=torch.float32)
            return im_t, mask, bbox
        
        # Regular ILIAS core image (query or positive)
        # Handle relative paths by prepending datapath
        if not path.startswith(self.datapath):
            path = join(self.datapath, 'ilias_core', path)
        
        with Image.open(path) as orig_img:
            original_size = (orig_img.height, orig_img.width)
            im_t = self.transform(orig_img.convert('RGB'))
        
        # Load bounding box from *_bbox.txt file
        bbox_file = path.replace('.jpg', '_bbox.txt')
        if not bbox_file.startswith(self.datapath):
            bbox_file = join(self.datapath, 'ilias_core', bbox_file)
        bbox = None
        mask = None
        
        if os.path.exists(bbox_file):
            bbox = load_box(bbox_file, original_size, im_t.shape[-2:])
            x_min, y_min, x_max, y_max = (int(value) for value in bbox)
            mask = torch.zeros(im_t.shape[-2:], dtype=torch.float32)
            mask[y_min:y_max + 1, x_min:x_max + 1] = 1.0
        
        # If no bbox, use full image
        if bbox is None:
            H, W = im_t.shape[-2:]
            bbox = torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32)
            mask = torch.ones((H, W), dtype=torch.float32)
        
        return im_t, mask, bbox

    def _load_from_extracted_dir(self, distractor_id, query_id=None):
        """
        Load distractor image from pre-extracted candidates directory.
        Uses direct path construction instead of glob for O(1) lookup.
        
        Args:
            distractor_id: e.g., "yfcc100m-000003.tar/0f0902d93f44f128eeb571c6282df69"
            query_id: Query ID (e.g., "instance_name/query/Q000_00.jpg") to locate query folder
        
        Returns:
            PIL.Image
        """
        if self.candidates_dir is None:
            raise ValueError("candidates_dir must be specified for distractor loading")
        
        if query_id is None:
            raise ValueError("query_id is required when loading distractors from candidates_dir")
        
        # Parse format: yfcc100m-{shard:06d}.tar/{hash}
        tar_filename, image_hash = distractor_id.split('.tar/', 1)
        
        # Construct the query folder name used by scripts/extract_ilias_candidates.py.
        query_folder = query_id.replace('/', '_').replace('.jpg', '')
        
        # Direct path construction - O(1) instead of glob
        img_path = os.path.join(self.candidates_dir, query_folder, f"{tar_filename}_{image_hash}.jpg")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Extracted image not found: {img_path}")
        
        return Image.open(img_path)
