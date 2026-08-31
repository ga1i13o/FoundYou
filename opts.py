import argparse


def get_args_parser() -> argparse.ArgumentParser:
    """Shared argument parser for FoundYou inference."""
    parser = argparse.ArgumentParser("FoundYou inference", add_help=False)

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    parser.add_argument("--checkpoint", type=str, help="Path to the FoundYou checkpoint.")

    # Experiment I/O
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--name_exp", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--img_size", type=int, default=518, help="Image size for resizing.")
    parser.add_argument("--batch_size", type=int, default=16, help="Global batch size (may be split per GPU).")

    # Data
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for datasets.")
    parser.add_argument("--dataset_file", type=str, default="permir", help="Dataset name. Use 'multi' for training the generalist model.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers.")

    # Prompting
    parser.add_argument("--prompt", type=str, default="mask", choices=["mask", "box", "point"], help="Prompt type for support frames; 'multi' samples a type at random.")

    return parser
