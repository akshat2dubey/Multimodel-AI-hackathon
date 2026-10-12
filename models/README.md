# Jalnetra — model weights

Checkpoints are **not** tracked here: they are large binaries, and the one that
matters is trained from data the repository does not redistribute.

| File | What it is |
|---|---|
| `flood_unet.pt` | The trained flood-segmentation model. **Not in the repo.** Produce it with `python scripts/train_segmentation.py --dataset <kuro_siwo\|sen1floods11> --output models/flood_unet.pt` on a real, labelled dataset (CUDA strongly recommended). |
| `flood_unet_smoke.pt` | Written **only** by `train_segmentation.py --smoke-test`: a few epochs on generated synthetic tiles, to prove the training/inference plumbing runs. It is not a result and detects nothing on real scenes. |

**Nothing here is required to run the pipeline.** With no checkpoint,
[`scripts/inference.py`](../scripts/inference.py) falls back to the log-ratio
change-detection baseline (after-minus-before VV backscatter drop), which still
produces a real flood mask from real imagery, and
`results/change_detection.json` records which method was used.

Training data (cite whichever you train on):

- **Kuro Siwo** — Bountos, N., et al. (2024). "Kuro Siwo: A Large-Scale Dataset
  for Flood Segmentation in Satellite Imagery." *NeurIPS 2024*.
  <https://github.com/Orion-AI-Lab/KuroSiwo> (MIT / CC BY).
- **Sen1Floods11** — Bonafilia, D., et al. (2020). "Sen1Floods11: A Georeferenced
  Cropped Sentinel-1 Dataset for Flood Detection." *CVPR Workshops 2020*.
  <https://github.com/cloudtostreet/Sen1Floods11> (CC BY 4.0).
