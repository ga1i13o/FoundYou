from os.path import join

from torchvision import transforms

from .ilias import DatasetILIAS
from .permir import DatasetPerMIR
from .permis import DatasetPerMis
from .perseg import DatasetPerSeg


def build_dataset(dataset_file: str, args=None, transform=None):
    """Build an inference dataset with the shared image transform."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    if dataset_file == 'perseg':
        return DatasetPerSeg(datapath=args.data_root, transform=transform)
    if dataset_file == 'permis':
        return DatasetPerMis(datapath=join(args.data_root, 'PerMIRS'), transform=transform)
    if dataset_file == 'permir':
        return DatasetPerMIR(datapath=args.data_root, transform=transform)
    if dataset_file == 'ilias':
        return DatasetILIAS(
            datapath=args.data_root,
            transform=transform,
            candidates_dir=getattr(args, 'candidates_dir', None),
        )
    raise ValueError(f'dataset {dataset_file} not supported')
