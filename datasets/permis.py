r""" PerMIS few-shot semantic segmentation dataset """
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class DatasetPerMis(Dataset):
    def __init__(self, datapath, transform):
        self.benchmark = 'permis'
        self.base_path = datapath
        self.transform = transform
        
        # Build episode metadata instead of loading all data
        self.episodes_metadata = self.build_episodes_metadata()
        print(f"Number of episodes: {len(self.episodes_metadata)}")

    def __len__(self):
        return len(self.episodes_metadata)

    def __getitem__(self, idx):
        episode_meta = self.episodes_metadata[idx]
        
        # Load query and support images/masks
        query_img, query_mask, support_imgs, support_masks = self.load_episode(episode_meta)
        
        # Transform query image
        query_img = self.transform(query_img)
        
        # Interpolate query mask to match transformed image size
        if query_mask.shape[-2:] != query_img.shape[-2:]:
            query_mask = F.interpolate(query_mask.unsqueeze(0).float(), size=query_img.shape[-2:], mode='nearest').squeeze(0)
        query_mask = query_mask.squeeze(0) if query_mask.dim() == 3 else query_mask
        
        # Transform support images
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        
        # Interpolate support masks
        support_masks_tmp = []
        for smask in support_masks:
            if smask.shape[-2:] != support_imgs.shape[-2:]:
                smask = F.interpolate(smask.unsqueeze(0).float(), size=support_imgs.shape[-2:], mode='nearest').squeeze(0)
            smask = smask.squeeze(0) if smask.dim() == 3 else smask
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)
        
        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': episode_meta['query_name'],
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': episode_meta['support_names'],
            'class_name': episode_meta['class_name']
        }
        
        return batch

    def build_episodes_metadata(self):
        """Build metadata for all episodes without loading images into memory"""
        episodes_metadata = []
        
        vid_ids = sorted([d for d in os.listdir(self.base_path) 
                         if os.path.isdir(os.path.join(self.base_path, d))])
        
        for vid_id in tqdm(vid_ids, desc=f"Building episodes metadata"):
            vid_path = os.path.join(self.base_path, vid_id)
            masks_path = os.path.join(vid_path, "masks.npz.npy")
            
            if not os.path.exists(masks_path):
                continue
            
            # Create two episodes per video (frame 0->1 and frame 0->2)
            for query_frame_idx in [1, 2]:
                episode_meta = {
                    'vid_id': vid_id,
                    'support_frame_idx': 0,
                    'query_frame_idx': query_frame_idx,
                    'masks_path': masks_path,
                    'support_names': [os.path.join(vid_path, '0.jpg')],
                    'query_name': os.path.join(vid_path, f'{query_frame_idx}.jpg'),
                    'class_name': vid_id
                }
                episodes_metadata.append(episode_meta)
        
        return episodes_metadata

    def load_episode(self, episode_meta):
        """Load images and masks for an episode"""
        vid_path = os.path.dirname(episode_meta['query_name'])
        
        # Load masks from npz file
        masks_np = np.load(episode_meta['masks_path'], allow_pickle=True)
        
        # Load support (frame 0)
        support_frame_path = os.path.join(vid_path, f"{episode_meta['support_frame_idx']}.jpg")
        support_img = Image.open(support_frame_path).convert("RGB")
        support_mask = torch.tensor(list(masks_np[episode_meta['support_frame_idx']].values())[0]).int()
        support_mask = support_mask.unsqueeze(0) if support_mask.dim() == 2 else support_mask
        
        # Load query
        query_frame_path = episode_meta['query_name']
        query_img = Image.open(query_frame_path).convert("RGB")
        query_mask = torch.tensor(list(masks_np[episode_meta['query_frame_idx']].values())[0]).int()
        query_mask = query_mask.unsqueeze(0) if query_mask.dim() == 2 else query_mask
        
        return query_img, query_mask, [support_img], [support_mask]
