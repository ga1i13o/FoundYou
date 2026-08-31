#!/bin/bash

# Script to download ILIAS precomputed features
# Usage: ./scripts/download_ilias_features.sh /path/to/ilias/directory

if [ $# -eq 0 ]; then
    echo "Error: Please provide the ILIAS directory as an argument"
    echo "Usage: $0 /path/to/ilias/directory"
    exit 1
fi

ILIAS_DIR=$1

# Check if directory exists
if [ ! -d "$ILIAS_DIR" ]; then
    echo "Error: Directory $ILIAS_DIR does not exist"
    exit 1
fi

echo "Changing to directory: $ILIAS_DIR"
cd "$ILIAS_DIR" || exit 1

# Create features directory if it doesn't exist
mkdir -p features
cd features || exit 1

echo "Downloading ILIAS SigLIP features..."
wget -r -np -nH --cut-dirs=2 -R "index.html*" -c -nc \
    https://vrg.fel.cvut.cz/ilias_data/features/vit_large_patch16_siglip_384.webli/

echo "Download complete!"
