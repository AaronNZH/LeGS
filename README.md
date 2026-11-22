# LeGS: Learnable Density Control for 3D Gaussian Splatting

**Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting** · ICML 2026

Zhenhua Ning, Xin Li, Jun Yu, Guangming Lu, Yaowei Wang, Wenjie Pei

[Paper (PMLR)](https://proceedings.mlr.press/v306/ning26e.html) · [arXiv](https://arxiv.org/abs/2605.00408) · [Results](#results) · [Quick Start](#quick-start) · [Method](#method) · [Citation](#citation)

LeGS learns which Gaussians to keep, clone, split, or prune during scene optimization. It uses rendering sensitivity to reward density decisions, improving novel-view quality over vanilla 3DGS on three benchmarks while using fewer Gaussians.

[![Figure 1: LeGS and FastGS compared on thin bicycle-wheel details.](https://arxiv.org/html/2605.00408v2/teaser.png)](https://arxiv.org/html/2605.00408v2/teaser.png)

*Figure 1. FastGS leaves artifacts around the wheel where its fixed rule allocates too few Gaussians. LeGS learns where to add detail. [Figure 1 in the paper](https://arxiv.org/pdf/2605.00408v2#page=1).*

## Results

### Qualitative comparison

[![Published Figure 3: Train and Playroom, with RGB renders above Gaussian visualizations for each scene. Left to right: ground truth, LeGS, FastGS, Perceptual-GS, and 3DGS-MCMC.](https://arxiv.org/html/2605.00408v2/qualitative_results.png)](https://arxiv.org/html/2605.00408v2/qualitative_results.png)

*Train and Playroom. The top row for each scene shows RGB renders; the second shows Gaussian visualizations. Baselines: FastGS, Perceptual-GS, and 3DGS-MCMC. [Open Figure 3](https://arxiv.org/html/2605.00408v2/qualitative_results.png) · [Figure 3 in the paper](https://arxiv.org/pdf/2605.00408v2#page=7).*

### Quantitative comparison

Selected rows from [Table 1](https://arxiv.org/html/2605.00408v2#S3.T1), reported for novel-view synthesis. **#G** is the final number of Gaussians in millions.

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #G (M) ↓ |
| :-- | :-- | --: | --: | --: | --: |
| Mip-NeRF 360 | Vanilla 3DGS | 27.59 | 0.815 | 0.215 | 2.74 |
| Mip-NeRF 360 | **LeGS** | **28.30** | **0.837** | **0.184** | **2.17** |
| Tanks & Temples | Vanilla 3DGS | 23.75 | 0.854 | 0.169 | 1.68 |
| Tanks & Temples | **LeGS** | **24.74** | **0.871** | **0.142** | **1.63** |
| Deep Blending | Vanilla 3DGS | 29.80 | 0.907 | 0.238 | 2.48 |
| Deep Blending | **LeGS** | **30.35** | **0.914** | **0.227** | **1.05** |

## Quick Start

### 1. Clone

```bash
git clone --recursive https://github.com/AaronNZH/LeGS.git
cd LeGS
```

### 2. Set up the environment

Create the Conda environment, then build the CUDA extensions from the repository root. The environment targets Python 3.7 and PyTorch 1.12.1 with CUDA 11.6. Use an NVIDIA driver compatible with CUDA 11.6 and a CUDA 11.6 development toolkit (including `nvcc`) with a supported host C++ compiler. Conda's `cudatoolkit` runtime alone does not provide the compiler. If needed, set `CUDA_HOME` to the development toolkit path.

```bash
conda env create --file environment.yml
conda activate fastgs
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda); assert sys.version_info[:2] == (3, 7); assert torch.__version__.split('+')[0] == '1.12.1' and torch.version.cuda == '11.6', 'Expected PyTorch 1.12.1 with CUDA 11.6'"
"${CUDA_HOME:+$CUDA_HOME/bin/}nvcc" --version | grep -F 'release 11.6,'
python -m pip install --no-index --only-binary=:all: --find-links 'https://data.pyg.org/whl/torch-1.12.1+cu116.html' 'torch-scatter==2.1.0+pt112cu116'
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization_fastgs submodules/simple-knn submodules/fused-ssim
python -c "from torch_scatter import scatter_add; import diff_gaussian_rasterization_fastgs, simple_knn._C, fused_ssim"
```

On Windows, use a shell with the supported Visual Studio build tools initialized and set `DISTUTILS_USE_SDK=1` before building the extensions (in Command Prompt, `set DISTUTILS_USE_SDK=1`).

### 3. Prepare the datasets

Download [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and the [Tanks & Temples and Deep Blending SfM datasets](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

<details>
<summary>Example dataset layout and training options</summary>

```text
datasets/
├── mipnerf360/
│   ├── bicycle/
│   └── …
├── db/
│   ├── playroom/
│   └── …
└── tanksandtemples/
    ├── truck/
    └── …
```

After setup, run `python train.py --help` for the available arguments and defaults. Common settings include `-s` (scene path), `-m` (output path), `-i` (image directory), `--eval` (hold out test views), and `--iterations` (training steps). The default image loader resizes inputs wider than 1,600 pixels; use `-r 1` to keep their original size, or `--data_device cpu` to keep input images in host memory.

</details>

### 4. Train and evaluate

Edit the dataset paths in `train.sh`, then run it with a fresh output directory. The launcher defaults to `output/rl` and does not overwrite earlier runs:

```bash
bash train.sh
# To start another run without moving the first:
LEGS_OUTPUT_ROOT=output/rl-rerun bash train.sh
```

The full-suite report requires valid 30,000-iteration test metrics and training statistics for every scene. It reports missing or invalid scenes instead of averaging them as zeros. To report an existing run separately, use `python report_results.py`, or prefix it with the same `LEGS_OUTPUT_ROOT` value used for that run.

## Method

The policy chooses **maintain, clone, split, or prune** for each Gaussian. A sensitivity-based reward measures how changing a Gaussian affects rendering error. The paper derives a closed-form score from the alpha-compositing terms, avoiding a separate render for each Gaussian. This release uses the [FastGS](https://github.com/fastgs/FastGS) renderer.

The trained Gaussians use the standard 3DGS representation and can be explored in the [SIBR viewer](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) or [SuperSplat](https://superspl.at/editor).

<details>
<summary>Research poster</summary>

![LeGS research poster](assets/poster.png)

</details>

## Citation

```bibtex
@inproceedings{pmlr-v306-ning26e,
  title     = {Beyond Heuristics: Learnable Density Control for 3{D} {G}aussian Splatting},
  author    = {Ning, Zhenhua and Li, Xin and Yu, Jun and Lu, Guangming and Wang, Yaowei and Pei, Wenjie},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {93422--93438},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/ning26e.html}
}
```

## Acknowledgments

This release builds on [FastGS](https://github.com/fastgs/FastGS). See the [repository license](LICENSE) and the applicable licenses for upstream components.
