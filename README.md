
<p align="center">
  <a href="http://www.micad.org/"><img src="https://img.shields.io/badge/Conference-MICAD%202026-0052cc.svg?style=flat&logo=target" alt="MICAD 2026"></a>
  <a href="http://www.micad.org/"><img src="https://img.shields.io/badge/Presentation-Oral%20(Paper%20ID%20740)-FF6F00.svg?style=flat" alt="Oral Presentation"></a>
  <a href="https://www.springer.com/series/7818"><img src="https://img.shields.io/badge/Springer-LNEE%20Series-003366.svg?style=flat" alt="Springer LNEE"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat" alt="License: MIT"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=flat&logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=flat&logo=python" alt="Python"></a>
  <a href="https://github.com/mukut02/MICAD_unet"><img src="https://img.shields.io/badge/GitHub-MICAD__unet-181717.svg?style=flat&logo=github" alt="GitHub Code"></a>
</p>

Official repository for the paper **"BISON: Boundary-Injected Segmentation Of Nuclei via Exact Softmax Attention"**, accepted for **Oral Presentation** at the **[7th International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2026)](http://www.micad.org/)**, Edinburgh, United Kingdom, October 22–24, 2026. Published in Springer *Lecture Notes in Electrical Engineering (LNEE)* (ISSN: 1876-1100).

---

**Authors:**  
Mukut Chatterjee<sup>1,*</sup>, Diptarko Bhattacharjee<sup>1,*</sup>, Debatra Banerjee<sup>1,*</sup>, Dmitrii Kaplun<sup>2,3,✉</sup>, Asya Lyanova<sup>3</sup>, Ram Sarkar<sup>1</sup>

<sup>1</sup> Department of Computer Science & Engineering, Jadavpur University, Kolkata, India  
<sup>2</sup> School of Computer Science and Technology / School of Artificial Intelligence, China University of Mining and Technology, Xuzhou, China  
<sup>3</sup> Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", St. Petersburg, Russia  

<sup>*</sup> Equal contribution (co-first authors)  
<sup>✉</sup> Corresponding author: [dikaplun@etu.ru](mailto:dikaplun@etu.ru)

**Conference Website:** [http://www.micad.org/](http://www.micad.org/) | **Paper ID:** 740  
**Code & Reproducibility:** [https://github.com/mukut02/MICAD_unet](https://github.com/mukut02/MICAD_unet)

---

## Contents

1. [Overview](#overview)
2. [Highlights](#highlights)
3. [Model Architecture](#model-architecture)
4. [Method in Detail](#method-in-detail)
5. [Training Objective](#training-objective)
6. [Experimental Setup](#experimental-setup)
7. [Results](#results)
8. [Ablation Studies](#ablation-studies)
9. [Qualitative Results](#qualitative-results)
10. [Limitations and Future Work](#limitations-and-future-work)
11. [Citation](#citation)
12. [Acknowledgements](#acknowledgements)
13. [License](#license)
14. [Contact](#contact)

---

## Overview

Accurate instance segmentation of cell nuclei in H&E histopathology images underpins critical computational pathology tasks such as cancer grading, sub-typing, and survival prognosis. It is particularly difficult in aggressive malignancies such as triple-negative breast cancer (TNBC): nuclei vary substantially in size, shape, and chromatin texture, are densely packed or overlapping, and frequently contrast poorly with the surrounding stroma.

Standard encoder-decoder networks lose high-frequency edge detail through spatial downsampling. U-Net merges touching nuclei into single regions, and even multi-task methods such as HoVer-Net, which predict horizontal-vertical (HV) offsets and separate instances by watershed, still lose boundary detail as the network compresses the image.

**BISON** (Boundary-Injected Segmentation Of Nuclei) is a ResNet-based multi-task architecture that injects geometric boundary priors throughout the network:

- A **Scharr Pre-Bottleneck Refinement gate** that preserves high-frequency edge detail before spatial compression;
- A **Spatial Boundary-Aware (SBA) Attention** module that fuses visual self-attention with a fixed, differential-operator-derived boundary similarity at the bottleneck;
- **Actively gated skip connections** and an **HV offset head**, trained with a **Boundary Consistency Loss** and a **Mean Squared Gradient Error (MSGE)** term, to sharpen instance boundaries and improve marker-controlled watershed separation.

---

## Highlights

- **State-of-the-Art on Three Benchmark Datasets** (Dice / IoU / AJI %):

| Dataset | Dice (%) | IoU (%) | AJI (%) |
|:---|:---:|:---:|:---:|
| **TNBC** | **86.86** | **76.78** | **59.33** |
| **MoNuSeg** | **82.28** | **69.89** | **60.12** |
| **CPM-15** | **89.23** | **80.55** | **67.69** |

- **Substantial Gains Over Prior Art:** Outperforms the strongest prior baselines by **+1.86**, **+0.86**, and **+4.23** Dice points on TNBC, MoNuSeg, and CPM-15, respectively.
- **AJI Improvement:** Improves the Aggregated Jaccard Index (AJI) by up to **9.6 points** over connected-component baselines, demonstrating superior separation of touching and overlapping nuclei.
- **SBA Attention is the Primary Driver:** Yields the largest single architectural improvement in ablation (**+4.3 to +4.4 AJI** across all datasets).
- **Exact Softmax Attention:** Runs at the compact $32 \times 32$ bottleneck ($N = 1024$ tokens), making full $N \times N$ pairwise self-attention computationally tractable without quadratic explosion.

---

## Model Architecture

![BISON Architecture](assets/model_architecture.png)

*Figure 1: Overview of the BISON architecture. Left: ResNet encoder stages E1–E4. Bottom: Scharr Pre-Bottleneck gate, ASPP, and SBA-Attention bottleneck. Right: Decoder stages D1–D4 with two-stage gated skip connections (Scharr boundary gate followed by attention gate). Far right: Seg head (binary foreground), HV offset head (instance separation via marker-controlled watershed), and auxiliary deep supervision heads (stages D2 and D3).*

### Resolution and Feature Dimensions

| Stage | Resolution | Channels (Enc / Dec) | Description |
|:---|:---:|:---:|:---|
| **Input** | $512 \times 512$ | 3 | Normalized 3-channel H&E image |
| **E1 / D1** | $512 \times 512$ | 64 / 32 | Initial convolution & shallow spatial features |
| **E2 / D2** | $256 \times 256$ | 128 / 64 | Stage 2 features + auxiliary supervision on D2 |
| **E3 / D3** | $128 \times 128$ | 256 / 128 | Stage 3 features + auxiliary supervision on D3 |
| **E4 / D4** | $64 \times 64$ | 512 / 256 | High-level semantic features |
| **Bottleneck** | $32 \times 32$ | 512 | Scharr Gate $\rightarrow$ ASPP $\rightarrow$ SBA-Attention |

**Model Outputs:**
1. **Binary Foreground Mask:** Pixel-level probability map $\hat{p} \in [0, 1]^{H \times W}$.
2. **Continuous HV Offset Maps:** Normalized directional offsets $\hat{O} \in [-1, 1]^{2 \times H \times W}$ pointing toward instance centroids.
3. **Auxiliary Predictions:** Stage D2 and D3 logits supervised to regularize intermediate representations.

---

## Method in Detail

### 1. Encoder

The encoder adopts the ResNet paradigm across four stages that progressively double channel depth ($64, 128, 256, 512$) and halve spatial resolution via max-pooling. The pre-pooling feature map of each stage, $S_l$ for $l \in \{1, 2, 3, 4\}$, is retained as a skip connection. Backbone ablations demonstrate that pre-activation ResNet-18 outperforms deeper variants (ResNet-34, ResNet-50) and lightweight mobile backbones (MobileNet-v2, EfficientNet-B0, SqueezeNet) due to cleaner gradient flow for high-frequency boundary signals.

### 2. Scharr Pre-Bottleneck Refinement Gate

Spatial downsampling attenuates fine edge details. Before the bottleneck, the final pooled feature map $Z \in \mathbb{R}^{C \times \frac{H}{16} \times \frac{W}{16}}$ ($C = 512$) is filtered: a $1 \times 1$ convolution projects $Z$ to a single-channel grayscale representation, multi-scale Scharr operators at two dilation rates extract an edge-likeness map $B$, and features are residually gated:

$$Z_{\text{gated}} = Z \odot (1 + \gamma B)$$

where $\gamma$ is a learnable scalar initialized to $0.1$. Where $\gamma B \approx 0$, features pass unaltered; near edges, high-frequency boundary responses are amplified. $Z_{\text{gated}}$ then enters an Atrous Spatial Pyramid Pooling (ASPP) module (atrous rates $\{3, 6, 12\}$, global pooling branch, dropout $0.1$) to yield multi-scale context $Z_{\text{aspp}}$.

### 3. Spatial Boundary-Aware (SBA) Attention

Standard self-attention relies solely on visual feature similarity, which often fails at nuclear membranes where stroma and chromatin share similar textures. SBA fuses visual self-attention with a non-learnable geometric boundary similarity:

1. Learned $1 \times 1$ projections generate queries, keys, and values ($Q, K, V$) from $Z_{\text{aspp}}$, partitioned into $h = 4$ heads of dimension $d_k = C / h$ over $N = 32 \times 32 = 1024$ spatial tokens.
2. In parallel, a deterministic **Scharr Boundary Descriptor** computes a $d_\psi = 6$ channel geometric fingerprint $\Psi$ per token:
   - **Channel 1:** Boundary likeness (sharpened gradient magnitude).
   - **Channel 2:** Cross-scale orientation consensus.
   - **Channels 3–4:** Oriented edge components $(r \cos 2\theta, \; r \sin 2\theta)$ encoding edge directionality.
   - **Channel 5:** Hessian polarity (ridge vs. valley indicator).
   - **Channel 6:** Blobness metric, $4 \det(H) / \operatorname{tr}(H)^2$ (approaching 1 for isotropic nuclei).

Visual and boundary similarity matrices are fused prior to the softmax normalization:

$$S_{\text{vis}} = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{\text{bnd}} = \Psi\Psi^\top$$

$$A = \operatorname{Softmax}\left(S_{\text{vis}} + \lambda S_{\text{bnd}}\right)$$

where $\lambda \in \mathbb{R}^h$ is a learnable per-head scaling factor parameterized as a square to guarantee positivity. A depthwise $3 \times 3$ convolution on $V$ supplies a localized inductive bias $V_{\text{loc}}$:

$$Z_{\text{attn}} = Z_{\text{aspp}} + f_{\text{out}}\left(AV + V_{\text{loc}}\right)$$

### 4. Decoder and Gated Skip Connections

The decoder mirrors the encoder through four bilinear upsampling stages. Each skip connection undergoes two-stage filtering rather than naive concatenation:

1. **Boundary Gate:** Scharr filtering amplifies high-frequency edge content:
   $$S_l^{\text{bnd}} = S_l \odot \left(1 + \gamma \, \mathcal{F}_{\text{scharr}}(S_l)\right)$$

2. **Attention Gate:** The upsampled decoder feature $F_{\text{dec}}^{(l)}$ gates the encoder features to suppress background artifacts:
   $$\alpha_l = \sigma\left(W_\psi \, \operatorname{ReLU}\left(W_g F_{\text{dec}}^{(l)} + W_x S_l^{\text{bnd}}\right)\right), \qquad \hat{S}_l = S_l^{\text{bnd}} \odot \alpha_l$$

$\hat{S}_l$ is concatenated with $F_{\text{dec}}^{(l)}$, blended with a $1 \times 1$ convolution, and refined via residual convolution blocks.

### 5. Prediction Heads

The stage D1 decoder feature $D_1 \in \mathbb{R}^{32 \times H \times W}$ feeds:
- **Seg Head:** $1 \times 1$ convolution yielding foreground logit $\hat{Y}$.
- **Auxiliary Heads:** Stage D2 and D3 features are projected to auxiliary segmentation logits to stabilize intermediate backpropagation.
- **HV Offset Head:** $1 \times 1$ convolution with $\tanh$ activation predicting continuous horizontal and vertical offsets $\hat{O} \in [-1, 1]^{2 \times H \times W}$.

Ground-truth offsets for pixel $(x, y)$ belonging to instance $k$ with centroid $(c_x^k, c_y^k)$ and $\varepsilon = 10^{-8}$:

$$O_h(x, y) = \frac{x - c_x^k}{\max_j |x_j - c_x^k| + \varepsilon}, \qquad O_v(x, y) = \frac{y - c_y^k}{\max_j |y_j - c_y^k| + \varepsilon}$$

### 6. Inference: Thresholding and Marker-Controlled Watershed

1. **Binarization:** Sigmoidal output $\hat{p} = \sigma(\hat{Y})$ is thresholded at optimal $\tau^*$ (selected via grid search on validation Dice over $[0.05, 0.95]$ with step $0.05$), yielding binary mask $M = \mathbb{1}[\hat{p} > \tau^*]$.
2. **Topographic Energy Surface:** Sobel gradients of the predicted HV fields form an energy surface $G$ that remains low inside nuclei and peaks at instance boundaries:
   $$G = \sqrt{(\nabla_x \hat{O}^h)^2 + (\nabla_y \hat{O}^h)^2 + (\nabla_x \hat{O}^v)^2 + (\nabla_y \hat{O}^v)^2}$$
3. **Marker Generation:** Candidate seeds are extracted where normalized gradient is low ($< 0.4$) within foreground $M$, followed by disk erosion (radius 2) to prevent seed merger.
4. **Watershed Flooding:** Marker-controlled watershed floods surface $G$ within mask $M$; spurious fragments smaller than 10 pixels are discarded.

---

## Training Objective

The overall optimization objective balances segmentation accuracy, instance geometry, and boundary alignment:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{hv}\mathcal{L}_{hv} + \lambda_{bnd}\mathcal{L}_{bnd}$$

### 1. Multi-Scale Segmentation Loss
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{seg}}^{(D_1)} + w_2 \mathcal{L}_{\text{seg}}^{(D_2)} + w_3 \mathcal{L}_{\text{seg}}^{(D_3)}$$

Each term combines Focal Tversky Loss (FTL) and class-weighted Binary Cross-Entropy (BCE). The positive weight adapts to the mean foreground ratio $\bar{f}$ capped by $\zeta$:
$$w_{\text{pos}} = \min\left(\frac{1 - \bar{f}}{\bar{f}}, \; \zeta\right)$$

### 2. Instance-Geometry Loss
Supervised over all foreground pixels $M$:
$$\mathcal{L}_{hv} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{MSGE}}$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|M|}\sum_{i \in M} \left\|\hat{O}_i - O_i\right\|^2, \qquad \mathcal{L}_{\text{MSGE}} = \frac{1}{|M|}\sum_{i \in M} \left\|\nabla\hat{O}_i - \nabla O_i\right\|^2$$

### 3. Boundary Consistency Loss
Compares Scharr edge responses between predicted logits and ground truth:
$$\mathcal{L}_{\text{bnd}} = \left\| \tanh\left(\kappa \, \mathcal{F}_{\text{scharr}}(\hat{Y})\right) - \tanh\left(\kappa \, \mathcal{F}_{\text{scharr}}(Y)\right) \right\|_1$$

---

## Experimental Setup

- **Datasets:** Three widely recognized public histopathology benchmarks:
  - **TNBC:** Triple-Negative Breast Cancer dataset (Naylor et al., 2018).
  - **MoNuSeg:** Multi-Organ Nucleus Segmentation challenge (Kumar et al., 2019).
  - **CPM-15:** Computational Pathology Multi-Organ dataset (Kumar et al., 2017).
- **Evaluation Metrics:**
  - **Dice Coefficient (%)** & **Intersection over Union (IoU, %):** Semantic segmentation fidelity.
  - **Aggregated Jaccard Index (AJI, %):** Instance-level clustering and boundary separation metric.

### Hyperparameters and Training Configuration

| Parameter | Configuration / Value |
|:---|:---|
| **Input Resolution** | $512 \times 512$, ImageNet mean/std normalization |
| **Epochs & Batch Size** | 500 epochs / batch size 4 |
| **Optimizer & Schedule** | AdamW ($lr_{\text{max}} = 10^{-3}$, weight decay $10^{-4}$) / OneCycleLR |
| **Gradient Clipping** | Maximum $L_2$ norm 1.0 |
| **Precision** | Mixed precision (FP16 / AMP) |
| **Model Weight Averaging** | Exponential Moving Average (EMA, decay 0.999) |
| **Data Augmentation** | Flips (H/V), random $90^\circ$ rotations, affine jitter, color jitter, Gaussian noise & blur, random erasing |
| **SBA Attention** | 4 heads, 6-channel Scharr boundary descriptor, bottleneck tokens $32 \times 32$ |
| **ASPP Module** | Atrous dilation rates $\{3, 6, 12\}$, global average pooling, dropout 0.1 |
| **Gate Parameter $\gamma$** | Learnable scalar, initialized to 0.1 |

---

## Results

### Comparison with State-of-the-Art Baselines

Quantitative performance (Dice and IoU in %) against recent medical segmentation architectures (Table 3 of the paper):

#### 1. Triple-Negative Breast Cancer (TNBC)
| Method | Publication / Venue | Dice (%) | IoU (%) |
|:---|:---|:---:|:---:|
| WDRE-NET | IEEE 2025 | 75.63 | 60.81 |
| DyGLNet | Pattern Recognition 2025 | 80.05 | 66.93 |
| SAAMU-Net | IEEE SCM 2025 | 84.10 | 72.60 |
| DeltaSegment | IEEE ISBI 2026 | 82.95 | 70.92 |
| AFINITI | Neural Comput. & Appl. 2024 | 85.00 | 73.91 |
| **BISON (Ours)** | **MICAD 2026** | **86.86** | **76.78** |

#### 2. Multi-Organ Nucleus Segmentation (MoNuSeg)
| Method | Publication / Venue | Dice (%) | IoU (%) |
|:---|:---|:---:|:---:|
| WDRE-NET | IEEE 2025 | 77.76 | 63.61 |
| EViT-UNet | IEEE 2025 | 79.27 | 65.87 |
| VEGAN | IEEE 2025 | 79.24 | 65.68 |
| DeltaSegment | IEEE ISBI 2026 | 81.42 | 68.71 |
| Deepfuzz | PLOS ONE 2023 | 79.10 | 65.42 |
| **BISON (Ours)** | **MICAD 2026** | **82.28** | **69.89** |

#### 3. Computational Pathology Multi-Organ (CPM-15)
| Method | Publication / Venue | Dice (%) | IoU (%) |
|:---|:---|:---:|:---:|
| U-Net++ | Springer DLMIA 2018 | 75.42 | 60.54 |
| Micro-Net | MedIA 2019 | 79.20 | 61.50 |
| HoVer-Net | MedIA 2019 | 80.00 | 66.67 |
| HI-Net | IEEE 2022 | 84.80 | 73.61 |
| AFINITI | Neural Comput. & Appl. 2024 | 85.00 | 73.91 |
| **BISON (Ours)** | **MICAD 2026** | **89.23** | **80.55** |

![Dice Comparison Across Benchmarks](assets/dice_comparison.png)

*Figure 2: Benchmark comparison across TNBC, MoNuSeg, and CPM-15 showing consistent margin of improvement.*

![Table 3 Comparison](assets/table3_comparison.png)

*Figure 3: Facsimile of Table 3 from the MICAD 2026 proceedings paper.*

---

## Ablation Studies

### 1. Architectural Component Ablation
Incremental addition of proposed modules to a baseline ResNet-UNet equipped with the HV offset head:
- **A:** Scharr Pre-Bottleneck Refinement Gate
- **B:** Spatial Boundary-Aware (SBA) Attention
- **C:** Attention Skip Gates
- **D:** Boundary Consistency Loss

| A | B | C | D | TNBC Dice | IoU | AJI | MoNuSeg Dice | IoU | AJI | CPM-15 Dice | IoU | AJI |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| — | — | — | — | 80.10 | 68.80 | 48.30 | 77.20 | 63.50 | 49.30 | 83.10 | 73.80 | 56.30 |
| ✓ | — | — | — | 81.40 | 70.10 | 51.10 | 78.50 | 65.10 | 52.10 | 84.40 | 75.10 | 59.10 |
| ✓ | ✓ | — | — | 83.20 | 72.50 | 55.50 | 80.20 | 67.20 | 56.40 | 86.20 | 77.50 | 63.50 |
| ✓ | ✓ | ✓ | — | 84.90 | 74.60 | 57.60 | 81.30 | 68.50 | 58.20 | 87.90 | 79.20 | 65.80 |
| ✓ | ✓ | ✓ | ✓ | **86.86** | **76.78** | **59.33** | **82.28** | **69.89** | **60.12** | **89.23** | **80.55** | **67.69** |

![AJI Ablation Progression](assets/ablation_aji.png)

*Figure 4: AJI metric progression across architectural components. SBA attention provides the largest single performance leap (+4.3 to +4.4 AJI across all three datasets).*

### 2. Loss Function Component Ablation
Isolating each loss term starting from the baseline $\mathcal{L}_{\text{seg}} + \mathcal{L}_{\text{MSE}}$:

| $\mathcal{L}_{\text{seg}}$ | $\mathcal{L}_{\text{MSE}}$ | $\mathcal{L}_{\text{aux}}$ | $\mathcal{L}_{\text{MSGE}}$ | $\mathcal{L}_{\text{bnd}}$ | TNBC Dice | IoU | AJI | MoNuSeg Dice | IoU | AJI | CPM-15 Dice | IoU | AJI |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✓ | — | — | — | 83.10 | 71.40 | 52.40 | 79.10 | 65.80 | 53.40 | 85.10 | 76.40 | 60.10 |
| ✓ | ✓ | ✓ | — | — | 84.90 | 73.20 | 54.20 | 80.50 | 67.30 | 55.20 | 86.90 | 78.20 | 62.20 |
| ✓ | ✓ | ✓ | ✓ | — | 85.30 | 75.40 | 57.40 | 81.60 | 68.70 | 58.60 | 88.30 | 79.60 | 65.40 |
| ✓ | ✓ | ✓ | ✓ | ✓ | **86.86** | **76.78** | **59.33** | **82.28** | **69.89** | **60.12** | **89.23** | **80.55** | **67.69** |

*Takeaway:* Deep supervision ($\mathcal{L}_{\text{aux}}$) stabilizes intermediate decoder training; MSGE ($\mathcal{L}_{\text{MSGE}}$) directly boosts AJI by $+3.2$ to $+3.4$ points by smoothing the offset field for watershed; Boundary Loss ($\mathcal{L}_{\text{bnd}}$) delivers the final boundary alignment boost.

### 3. Encoder Backbone Ablation
Comparing MobileNet-v2, EfficientNet-B0, SqueezeNet, ResNet-34, and ResNet-50 against pre-activation ResNet-18:
- Lightweight backbones heavily compress feature representations, losing subtle boundary gradients.
- Deeper backbones (ResNet-50) exhibit diminishing returns and higher computational overhead.
- **Pre-activation ResNet-18** delivers optimal gradient propagation and the highest overall accuracy.

![Table 1](assets/table1_architecture_ablation.png)
![Table 2](assets/table2_loss_ablation.png)

---

## Qualitative Results

![Qualitative Visualizations](assets/qualitative_results.jpg)

*Figure 5: Qualitative results across datasets. Columns (left to right): (1) Input H&E image, (2) Ground-truth instance mask, (3) Predicted probability map $\hat{p}$, (4) Binary segmentation mask, (5) Grad-CAM attention heatmap. Rows (top to bottom): TNBC (Dice: 0.8965, $\tau^* = 0.90$), CPM-15 (Dice: 0.9039, $\tau^* = 0.85$), and MoNuSeg (Dice: 0.8603, $\tau^* = 0.80$).*

---

## Limitations and Future Work

### Limitations
- **HV Offset Supervision Fallback:** Relies on connected-component labeling when instance-level centroids are not natively annotated, occasionally introducing slight offset error in dense clusters.
- **Bottleneck-Restricted SBA Attention:** Full quadratic attention is restricted to the $32 \times 32$ bottleneck to maintain computational tractability.
- **Fixed Boundary Filters:** Fixed differential Scharr kernels may not dynamically generalize to non-H&E stains or alternative imaging modalities (e.g., IHC, fluorescence).

### Future Work
- **Linear & Windowed Attention:** Extending boundary-aware attention to higher-resolution decoder stages via linear or windowed attention mechanisms.
- **Learnable Geometric Priors:** Designing parameter-efficient, learnable differential operators adapted end-to-end to stain variations.
- **Cross-Modality Deployment:** Validating on whole-slide imaging (WSI) workflows and diverse multi-center clinical cohorts.

---

## Citation

If you find this work, codebase, or results helpful in your research, please cite our paper:

```bibtex
@inproceedings{chatterjee2026bison,
  title     = {{BISON}: Boundary-Injected Segmentation of Nuclei via Exact Softmax Attention},
  author    = {Chatterjee, Mukut and Bhattacharjee, Diptarko and Banerjee, Debatra and Kaplun, Dmitrii and Lyanova, Asya and Sarkar, Ram},
  booktitle = {Proceedings of the 7th International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD 2026)},
  series    = {Lecture Notes in Electrical Engineering (LNEE)},
  volume    = {740},
  publisher = {Springer},
  year      = {2026},
  address   = {Edinburgh, United Kingdom},
  month     = {October},
  url       = {http://www.micad.org/}
}
