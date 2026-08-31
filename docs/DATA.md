# Data Setup

Run all commands from the repository root. By default, FoundYou expects datasets under `data/`; inference scripts accept `--data_root` if the datasets are stored elsewhere. The ILIAS preprocessing commands below assume the default location.

The expected layout is:

```text
FoundYou/
└── data/
    ├── PerMIRS/
    │    ├── 0/
    │    │   ├── 0.jpg
    │    │   ├── 1.jpg
    │    │   ├── 2.jpg
    │    │   └── masks.npz.npy
    │    └── ...
    ├── ilias/
    │   └── download/
    │       └── ilias/
    │           ├── features/
    │           ├── ilias_core/
    │           ├── image_ids/
    │           └── yfcc100m/
    └── PerSeg/
        ├── Annotations/
        └── Images/
```

If the datasets live on another disk, you can symlink that location to `data/`.

## PerMIS and PerMIR

The PerMIS segmentation and PerMIR retrieval benchmarks use the same **PerMIRS** data sampled from [BURST](https://github.com/Ali2500/BURST-benchmark). We used the dataset generation pipeline released in [PDM / Where's Waldo](https://github.com/dvirsamuel/PDM), and we provide the formatted result via GDrive.

### 1. Download PerMIRS

```bash
mkdir -p data
gdown https://drive.google.com/file/d/1VMXkWLHEJ2AdXSBRSRHYL4elLYbr5TwM/view?usp=sharing -O data/PerMIRS.zip
unzip data/PerMIRS.zip -d data && rm data/PerMIRS.zip
```

This produces 216 instance folders under `data/PerMIRS/`. Each folder contains three frames and a `masks.npz.npy` file. 
The folder also contains a query/gallery split used by the retrieval evaluation, in the following metadata files:

- `permir_image_query_ids.pt`
- `permir_positive_ids.pt`
- `permir_gt.pt`

If you use a different dataset root, pass it with `--data_root`.

## PerSeg

Download and extract PerSeg:

```bash
mkdir -p data
gdown "https://drive.google.com/file/d/18TbrwhZtAPY5dlaoEqkPa5h08G9Rjcio/view?usp=sharing" -O data/PerSeg.zip
unzip data/PerSeg.zip -d data && rm data/PerSeg.zip
mv "data/data 3" data/PerSeg
```

After extraction, images and annotations should be located at `data/PerSeg/Images/` and `data/PerSeg/Annotations/`, respectively.

## ILIAS

ILIAS contains:

- `ilias_core`: 1,232 queries and their positives, totaling 5,947 images;
- `yfcc100m`: approximately 99 M distractor images stored in tar shards.

Preparing the full benchmark requires four steps: downloading ILIAS, downloading the precomputed SigLIP features, retrieving the top candidates, and extracting the images that FoundYou will rerank.

### 1. Download ILIAS

Download `ilias_core`, the image-ID files, and all YFCC100M shards directly to the path expected by the dataset loader:

```bash
python scripts/download_ilias.py data/ilias/download/ilias
```

> **Storage requirement:** the full ILIAS download requires approximately **6 TB**.

The downloader accepts three additional positional arguments: the maximum number of YFCC100M shards, the number of retries, and the number of parallel workers. For example, the following command downloads one YFCC100M shard for a small-scale test:

```bash
python scripts/download_ilias.py data/ilias/download/ilias 1 1 8
```

The default shard limit is `0`, meaning unlimited. Omit the extra arguments, as in the full command above, to download the complete benchmark.

To run inference in the few-shot retrieval setting, download the following JSON:

```bash
gdown https://drive.google.com/file/d/1Zbt1gjlouCdspfOJfmOvq13Qib84feAH/view?usp=sharing -O data/ilias/download/ilias/retrieval_shots_retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000_3shots_seed42.json
```

### 2. Download precomputed SigLIP features

Download the query, positive, and distractor features used for the initial retrieval:

```bash
./scripts/download_ilias_features.sh data/ilias/download/ilias
```

The features are stored under `data/ilias/download/ilias/features/vit_large_patch16_siglip_384.webli/`.

### 3. Retrieve the top-1,000 candidates

The retrieval script requires `h5py` and FAISS. Running the full search over 100 M 1024-dimensional features at once requires more than 500 GB of RAM. Setting `--batch_search 10` processes ten 1 M-feature shards at a time and reduces peak memory usage to approximately 100 GB.

```bash
python scripts/retrieval_precomputed_feats.py \
  --model_name vit_large_patch16_siglip_384.webli \
  --total_distractors 100 \
  --k 1000 \
  --batch_search 10
```

This creates:

```text
retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000.pkl
```

Add `--evaluate` to compute the retrieval metrics after the search.

### 4. Extract candidates for reranking

FoundYou loads distractors from image files rather than directly from the YFCC100M tar shards. Extract the top candidates for every query with:

```bash
python scripts/extract_ilias_candidates.py \
  --retrieval_pkl retrieval_candidates_vit_large_patch16_siglip_384.webli_100M_top1000.pkl \
  --data_root data/ilias/download/ilias \
  --yfcc_root data/ilias/download/ilias/yfcc100m \
  --output_dir ilias_candidates_siglip_100M_top1k \
  --top_k 100
```

Set `--top_k` to at least the `--top_n` value used during FoundYou reranking. The resulting `candidates_output/` directory and retrieval pickle can then be passed to `inference_reranking.py` as described in the main README.
