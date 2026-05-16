# TASC-SwinMT
**Task-Adaptive Synergistic Cross-Task Swin Multi-Task Framework for Lung CT and Cardiac MRI Image Interpolation and Segmentation**

## 1. Model Introduction
TASC-SwinMT is a unified multi-task learning framework tailored for **lung CT** and **cardiac MRI**, which jointly accomplishes **image interpolation** and **multi-frame segmentation** tasks. It addresses critical limitations of conventional methods, such as computational redundancy, insufficient exploitation of spatiotemporal shared features, lack of anatomical constraints for interpolation, and absence of temporal context for segmentation.

Built upon the **Swin Transformer** backbone, the framework adopts a **shared SwinUNet encoder** to extract universal spatial features from paired input frames, and deploys two task-specific decoders to predict intermediate interpolated frames and segmentation masks respectively. Three dedicated collaborative modules are designed to boost cross-task learning:

### 🧩 Overall Framework (模型总结构图)
<center>
<img src="figures/overall_framework.png" width="85%" alt="TASC-SwinMT Overall Framework">
</center>

- **TALA (Task-Aware Lightweight Adapter)**: Captures spatial dual-bottleneck features and global frequency-domain information to generate task-adaptive representations.
  <center>
  <img src="figures/TALA_module.png" width="70%" alt="TALA Module Architecture">
  </center>

- **MSTAF (Multi-Scale Task Alignment Fusion)**: Aligns cross-level feature distributions via bidirectional cross-attention, multi-scale spatial extraction, and frequency enhancement.
  <center>
  <img src="figures/MSTAF_module.png" width="70%" alt="MSTAF Module Architecture">
  </center>

- **CTCI (Cross-Task Collaborative Interaction)**: Enables fine-grained cross-task feature interaction by integrating spatial extraction, frequency alignment, and dynamic gating fusion.
  <center>
  <img src="figures/CTCI_module.png" width="70%" alt="CTCI Module Architecture">
  </center>

A **learnable dynamic multi-task loss** is employed to adaptively balance pixel-level interpolation reconstruction and segmentation classification optimization, avoiding training bias toward either task.

Extensive experiments on the public datasets **MSD Task06_Lung** (lung tumor) and **MSD Task02_Heart** (left atrium) demonstrate:
- Cardiac MRI: Interpolation achieves **41.50±0.20 dB PSNR** and **0.990±0.0003 SSIM**; Segmentation achieves **0.967±0.005 Dice** and **0.940±0.007 IoU**.
- Lung CT: Interpolation achieves **40.76±0.38 dB PSNR** and **0.974±0.0019 SSIM**; Segmentation achieves **0.926±0.014 Dice** and **0.869±0.013 IoU**.

The framework delivers clearer lesion boundary reconstruction, more accurate small-target segmentation, and better inter-slice temporal consistency, providing a stable and generalizable solution for synchronous clinical analysis of lung CT and cardiac MRI.

## 2. Data Preparation
Organize the dataset in the following directory structure. All medical images are stored in `.nii.gz` format:
```text
Task06_Lung/
├── imagesTr/
│   ├── ._lung_001.nii.gz
│   ├── ._lung_003.nii.gz
│   ├── ._lung_004.nii.gz
│   └── ... (+123 .gz files)
├── imagesTs/
│   ├── ._lung_002.nii.gz
│   ├── ._lung_007.nii.gz
│   ├── ._lung_008.nii.gz
│   └── ... (+61 .gz files)
├── labelsTr/
│   ├── ._lung_001.nii.gz
│   ├── ._lung_003.nii.gz
│   ├── ._lung_004.nii.gz
│   └── ... (+123 .gz files)
├── ._dataset.json
├── dataset.json
├── ._imagesTr
├── ._imagesTs
└── ._labelsTr
```

## 3. Training
### Training Command
Navigate to the code directory and run the training script:
```bash
cd CODE
python train.py
```

### Training Outputs
#### Model Weights (checkpoints/)
5-fold cross-validation best model weights:
```text
checkpoints/
├── fold_1/
│   └── best_model.pth
├── fold_2/
│   └── best_model.pth
├── fold_3/
│   └── best_model.pth
├── fold_4/
│   └── best_model.pth
└── fold_5/
    └── best_model.pth
```

#### Training Convergence Analysis (analysis/)
Multi-task training feature contribution, gradient correlation visualization and quantitative results:
```text
analysis/
├── fold_1/
│   ├── feature_contribution_heatmap.png
│   └── gradient_correlation.png
├── fold_2/
│   ├── feature_contribution_heatmap.png
│   └── gradient_correlation.png
├── fold_3/
│   ├── feature_contribution_heatmap.png
│   └── gradient_correlation.png
├── fold_4/
│   ├── feature_contribution_heatmap.png
│   └── gradient_correlation.png
├── fold_5/
│   ├── feature_contribution_heatmap.png
│   └── gradient_correlation.png
├── fold_1_task_analysis.csv
├── fold_2_task_analysis.csv
├── fold_3_task_analysis.csv
├── fold_4_task_analysis.csv
└── fold_5_task_analysis.csv
```

## 4. Testing
### Testing Command
Run the test script for model inference and evaluation:
```bash
python test.py
```

### Testing Outputs
Test results, visualization and quantitative metric summaries:
```text
Heart_results/
├── fold_1/
│   ├── visualizations/
│   │   ├── sample_0000/
│   │   │   ├── grid_3x2_seg_visualization.png
│   │   │   ├── grid_4x1_raw_visualization.png
│   │   │   ├── sample0000_frame1_seg_gt.png
│   │   │   └── ... (+7 .png files)
│   │   ├── sample_0001/
│   │   │   ├── grid_3x2_seg_visualization.png
│   │   │   ├── grid_4x1_raw_visualization.png
│   │   │   ├── sample0001_frame1_seg_gt.png
│   │   │   └── ... (+7 .png files)
│   │   ├── sample_0002/
│   │   ├── sample_0003/
│   │   └── ...
│   │   ├── sample_0199/
│   │   │   ├── sample0199_frame1_seg_gt.png
│   │   │   └── ... (+7 .png files)
├── 5fold_average_summary.csv
└── test_summary.csv
```
```
