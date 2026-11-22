<div align="center">
<h1>Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting</h1> 
<h2>ICML 2026 <a href="https://arxiv.org/abs/2605.00408">[arXiv]</a> </h2> 

</div>

<p align="center">
    <img src="assets/poster.png" width="800px"/>
</p>

## 🚀 What is LeGS?

LeGS is a **learnable framework** to density control for 3D Gaussian Splatting training:
- **📑 Learnable Density Control**: Control the scene density using an RL network to optimize rendering quality.
- **⚡ High fidelity**:  Significantly improves rendering quality with comparable training time.
- **🎯 Easy Integration**: Seamlessly integrates with various backbones (Vanilla 3DGS, FastGS, etc.)

## 📢 Latest Updates
#### 🔥 **[2026.05.01]** Code Released - Get Started Now! 🚀

## 🚀 Quick Start

### 📥 Clone the Repository

```bash
git clone https://github.com/AaronNZH/LeGS.git --recursive
cd LeGS
```

### ⚙️ Environment Setup

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

### 📂 Dataset Organization

Organize your datasets in the following structure:

```bash
datasets/
├── mipnerf360/
│   ├── bicycle/
│   ├── flowers/
│   └── ...
├── db/
│   ├── playroom/
│   └── ...
└── tanksandtemples/
    ├── truck/
    └── ...
```

The MipNeRF360 scenes are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). You can find our SfM data sets for Tanks&Temples and Deep Blending [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). 

## 🎯 Training & Evaluation

Edit the dataset paths in `train.sh`, then run it with a fresh output directory. The launcher defaults to `output/rl` and does not overwrite earlier runs:

```bash
bash train.sh
# To start another run without moving the first:
LEGS_OUTPUT_ROOT=output/rl-rerun bash train.sh
```

The full-suite report requires valid 30,000-iteration test metrics and training statistics for every scene. It reports missing or invalid scenes instead of averaging them as zeros. To report an existing run separately, use `python report_results.py`, or prefix it with the same `LEGS_OUTPUT_ROOT` value used for that run.

<details>
<summary><span style="font-weight: bold;">📋 Advanced: Command Line Arguments for train.py</span></summary>
  #### --grad_abs_thresh 
  Absolute gradient (same as Abs-GS) threshold for density control.
  #### --grad_thresh
  Gradient(same as vanilla 3DGS) threshold for density control.
  #### --highfeature_lr
  Learning rate for high-order SH coefficients (features_rest).
  #### --lowfeature_lr
  Learning rate for low-order SH coefficients (features_dc).
  #### --dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified.
  #### --mult 
  Multiplier for the compact box to control the tile number of each splat
  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

Note that similar to MipNeRF360 and vanilla 3DGS, we target images at resolutions in the 1-1.6K pixel range. For convenience, arbitrary-size inputs can be passed and will be automatically resized if their width exceeds 1600 pixels. We recommend to keep this behavior, but you may force training to use your higher-resolution images by setting ```-r 1```.

## 🎬 Interactive Viewers

Our 3DGS representation is identical to vanilla 3DGS, so you can use the official [SIBR viewer](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#interactive-viewers) for interactive visualization. For a quick start without local setup, try the web-based [Supersplat](https://superspl.at/editor).


## 🙏 Acknowledgements

This project is built upon [FastGS](https://github.com/fastgs/FastGS/tree/main). We extend our gratitude to all the authors for their outstanding contributions and excellent repositories!

**License**: Please adhere to the licenses of FastGS.


## Citation
If you find this repo useful, please cite:
```
@misc{ning2026heuristicslearnabledensitycontrol,
      title={Beyond Heuristics: Learnable Density Control for 3D Gaussian Splatting}, 
      author={Zhenhua Ning and Xin Li and Jun Yu and Guangming Lu and Yaowei Wang and Wenjie Pei},
      year={2026},
      eprint={2605.00408},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.00408}, 
}
```

---

<div align="center">

**⭐ If LeGS helps your research, please consider starring this repository!**

</div>

---
