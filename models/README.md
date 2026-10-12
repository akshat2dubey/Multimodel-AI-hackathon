# Jalnetra — model weights

Checkpoints are **not** tracked here: they are large binaries, and the one that
matters is trained from data the repository does not redistribute.

| File | What it is |
|---|---|
| `flood_unet.pt` | The trained flood-segmentation model. **Not in the repo.** Produce it with `python scripts/train_segmentation.py --dataset <kuro_siwo\|sen1floods11> --output models/flood_unet.pt` on a real, labelled dataset (CUDA strongly recommended). |
| `flood_unet_smoke.pt` | Written **only** by `train_segmentation.py --smoke-test`: a few epochs on generated synthetic tiles, to prove the training/inference plumbing runs. It is not a result and detects nothing on real scenes. |

## Loading rule (why a `.sha256` file sits next to each checkpoint)

A `.pt` file is a pickle, so loading one can run code. Therefore:

1. `scripts/safe_model.py` is the **only** place this project deserialises a
   checkpoint, always with `torch.load(..., weights_only=True)` — a pickled
   object is refused, not executed.
2. `train_segmentation.py` writes `<checkpoint>.pt.sha256` beside the weights it
   saves. `inference.py` verifies that digest and **refuses to load a checkpoint
   whose bytes do not match** (e.g. weights swapped after the model was
   evaluated, or a truncated download).
3. A checkpoint with *no* sidecar is also refused, because an unverifiable file
   is not the file that was benchmarked. The explicit escape hatch is
   `inference.py --trust-unverified-model`, for weights you produced yourself.

So when you train on Colab/Kaggle, download **both** files:
`flood_unet.pt` **and** `flood_unet.pt.sha256`. No model path is ever accepted
from the dashboard — the web UI can only run the CLI's default location.

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
