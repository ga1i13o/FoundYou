r"""PerSeg few-shot segmentation dataset."""
import os
from os.path import join

import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetPerSeg(Dataset):
    def __init__(self, datapath, transform):
        self.benchmark = 'perseg'
        self.base_path = join(datapath, 'PerSeg')
        self.transform = transform
        self.class_ids = sorted(os.listdir(join(self.base_path, 'Images')))
        self.img_metadata = self.build_img_metadata()
        print(f"Number of images: {len(self.img_metadata)}")

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names = self.sample_episode(idx)
        query_img, query_mask = self.load_frame(query_name)
        support_frames = [self.load_frame(path) for path in support_names]

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'support_imgs': torch.stack([image for image, _ in support_frames]),
            'support_masks': torch.stack([mask for _, mask in support_frames]),
            'support_names': support_names,
            'class_name': os.path.basename(os.path.dirname(query_name)),
        }
        return batch

    def load_frame(self, image_path):
        with Image.open(image_path) as image:
            im_t = self.transform(image.convert('RGB'))

        mask_path = image_path.replace('Images', 'Annotations').replace('.jpg', '.png')
        mask = self.read_mask(mask_path, im_t.shape[-2:])
        return im_t, mask

    @staticmethod
    def read_mask(mask_path, size):
        with Image.open(mask_path) as mask:
            mask = transforms.functional.resize(mask.convert('L'), size, interpolation=transforms.InterpolationMode.NEAREST)
            mask = transforms.functional.pil_to_tensor(mask)[0]
        return (mask > 0).float()

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        support_names = [join(os.path.dirname(query_name), '00.jpg')]
        return query_name, support_names

    def build_img_metadata(self):
        img_metadata = []
        for class_id in self.class_ids:
            if class_id.startswith('.'):
                continue
            class_path = join(self.base_path, 'Images', class_id)
            if not os.path.isdir(class_path):
                continue
            for image_name in sorted(os.listdir(class_path)):
                stem, extension = os.path.splitext(image_name)
                if extension.lower() == '.jpg' and stem != '00':
                    img_metadata.append(join(class_path, image_name))
        return img_metadata
