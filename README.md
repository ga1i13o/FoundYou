<div align="center">

# FoundYou: A Unified Model for Personalized Segmentation and Retrieval - Demo

[Gabriele Trivigno*](https://scholar.google.com/citations?user=JXf_iToAAAAJ&hl=en) ·
[Marcos Alfaro*](https://scholar.google.com/citations?user=QXWedBkAAAAJ&hl=en) ·
[Claudia Cuttano*](https://scholar.google.it/citations?user=W7lNKNsAAAAJ&hl=en) ·
[Gabriele Berton](https://scholar.google.com/citations?user=pc_rMSMAAAAJ&hl=en) ·
[Luis Paya](https://scholar.google.com/citations?user=pFlS_NQAAAAJ&hl=en) ·
[Carlo Masone](https://scholar.google.it/citations?user=cM3Iz_4AAAAJ&hl=en)
</div>
This repository provides access to the interactive demo for large-scale personalized retrieval and segmentation.

The demo allows you to upload a reference image, optionally specify the target object with a point or bounding box, and search for visually identical instances in a large image collection. Results are ranked by instance-level similarity, and the top match is shown together with the predicted segmentation mask.

The interface is designed to demonstrate instance-level search in realistic conditions, where objects must be identified across different scenes, viewpoints, and cluttered environments.

## Demo

👉 **Launch the demo:**  
https://4fd43f2da70b4be13d.gradio.live/

## Features

- Upload a reference image
- Optional point / box prompt
- Ranking of candidate matches
- Visualization of top results
- Segmentation of the best match

## Notes

The demo runs on a remote server.  
Loading time may depend on server availability.
