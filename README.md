# BISON: Boundary-Injected Segmentation of Nuclei via Exact Softmax Attention

**MICAD 2026** (7th International Conference on Medical Imaging and Computer-Aided Diagnosis), Edinburgh, 22-24 Oct 2026 | Paper ID 740

**Authors:** Mukut Chatterjee<sup>1</sup>, Diptarko Bhattacharjee<sup>1</sup>, Debatra Banerjee<sup>1</sup>, Dmitrii Kaplun<sup>2,3,*</sup>, Asya Lyanova<sup>3</sup>, Ram Sarkar<sup>1</sup>

<sup>1</sup> Jadavpur University, Kolkata, India | <sup>2</sup> China University of Mining and Technology, China | <sup>3</sup> Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", Russia

<sup>*</sup> Corresponding author: dikaplun@etu.ru

**Code, hyperparameters and ablation results:** https://github.com/mukut02/MICAD_unet

---

## Contents

1. [Overview](#overview)
2. [Highlights](#highlights)
3. [Model architecture](#model-architecture)
4. [Method in detail](#method-in-detail)
5. [Training objective](#training-objective)
6. [Experimental setup](#experimental-setup)
7. [Results](#results)
8. [Ablation studies](#ablation-studies)
9. [Qualitative results](#qualitative-results)
10. [Limitations and future work](#limitations-and-future-work)
11. [Citation](#citation)
12. [Acknowledgements](#acknowledgements)

---

## Overview

Accurate instance segmentation of cell nuclei in H&E histopathology images underpins computational pathology tasks such as cancer grading, sub-typing and survival prognosis. It is difficult in aggressive malignancies such as triple-negative breast cancer (TNBC): nuclei vary in size, shape and chromatin texture, are densely packed or overlapping, and often contrast poorly with the surrounding stroma.

Standard encoder-decoder networks lose high-frequency edge detail through spatial downsampling. U-Net merges touching nuclei into single regions, and even multi-task methods such as HoVer-Net, which predict horizontal-vertical (HV) offsets and separate instances by watershed, still lose boundary detail as the network compresses the image.

**BISON** (Boundary-Injected Segmentation Of Nuclei) is a ResNet-based multi-task architecture that injects geometric boundary priors throughout the network:

- a **Scharr Pre-Bottleneck Refinement gate** that preserves edge detail before spatial compression;
- a **Spatial Boundary-Aware (SBA) Attention** module that fuses visual self-attention with a fixed, differential-operator-derived boundary similarity at the bottleneck;
- **actively gated skip connections** and an **HV offset head**, trained with a **Boundary Consistency Loss** and a **Mean Squared Gradient Error (MSGE)** term, to sharpen instance boundaries and improve marker-controlled watershed.

## Highlights

- State-of-the-art on three public datasets. Dice / IoU / AJI:

| Dataset | Dice | IoU | AJI |
|---|---|---|---|
| TNBC | 86.86 | 76.78 | 59.33 |
| MoNuSeg | 82.28 | 69.89 | 60.12 |
| CPM-15 | 89.23 | 80.55 | 67.69 |

- Gains of 1.86, 0.86 and 4.23 Dice points over the strongest prior baseline on TNBC, MoNuSeg and CPM-15 respectively.
- SBA attention is the largest single architectural gain in the ablation, about +4.3 to +4.4 AJI on every dataset.
- The attention module is exact softmax attention: it runs on the low-resolution bottleneck (about 1,024 tokens), so the full N x N matrix is tractable.

## Model architecture

![BISON architecture](assets/model_architecture.png)

*Left:* ResNet encoder stages E1-E4. *Bottom:* Scharr gate, ASPP and SBA-Attention bottleneck. *Right:* decoder stages D1-D4, reached via two-stage gated skips (boundary gate, then attention gate). *Far right:* the Seg head yields a binary mask by thresholding; the HV head's offsets drive watershed; both are fused into the final instance mask.

| Stage | Resolution | Channels |
|---|---|---|
| Input | 512 x 512 | 3 |
| E1 / D1 | 512 x 512 | 64 / 32 |
| E2 / D2 | 256 x 256 | 128 / 64 |
| E3 / D3 | 128 x 128 | 256 / 128 |
| E4 / D4 | 64 x 64 | 512 / 256 |
| Scharr gate, ASPP, SBA-Attention | 32 x 32 | 512 |

Outputs: a binary foreground mask, a pair of continuous HV offset maps encoding each pixel's relation to its instance centroid, and auxiliary predictions (decoder stages D2 and D3) that regularise training.

## Method in detail

### 1. Encoder

The encoder follows the ResNet paradigm with four stages that double channel depth (64, 128, 256, 512) and halve spatial resolution via max-pooling. The pre-pooling feature map of each stage, $S_l$ for $l \in \{1,2,3,4\}$, is kept as a skip connection. The paper's backbone ablation compares MobileNet-v2, EfficientNet-B0, SqueezeNet, ResNet-34 and ResNet-50 against a pre-activation ResNet-18 under an identical setup; ResNet-18 performed best.

### 2. Scharr Pre-Bottleneck Refinement gate

Pooling attenuates boundary detail, so the final pooled feature map $Z \in \mathbb{R}^{C \times \frac{H}{16} \times \frac{W}{16}}$ ($C = 512$) is gated before the bottleneck. A 1x1 convolution projects $Z$ to grayscale, fixed Scharr filters at two dilation scales compute a boundary-likeness map $B$, and features are gated residually:

$$Z_{\text{gated}} = Z \odot (1 + \gamma B)$$

$\gamma$ is a learnable scalar initialised to 0.1. Where $\gamma B \approx 0$ features pass through unchanged; near edges they are amplified. $Z_{\text{gated}}$ then passes through an ASPP module (atrous rates {3, 6, 12}, a global-pooling branch, dropout 0.1) to give multi-scale context $Z_{\text{aspp}}$, which treats every spatial location independently.

### 3. Spatial Boundary-Aware (SBA) Attention

Visual similarity alone cannot separate pixels of similar texture lying on opposite sides of a nucleus boundary. SBA fuses visual similarity with an explicit geometric boundary similarity.

- Learned 1x1 convolutions produce $Q, K, V$ from $Z_{\text{aspp}}$, reshaped into $h = 4$ heads of dimension $d_k = C/h$ over $N = \frac{H}{16} \cdot \frac{W}{16}$ tokens ($N \approx 1024$).
- In parallel, a fixed (non-learnable) **Scharr Boundary Descriptor** extracts a $d_\psi = 6$ channel structural fingerprint $\Psi$ per token:
  1. boundary likeness (a sharpened gradient magnitude),
  2. cross-scale orientation agreement,
  3. and 4. two oriented-edge components $(r\cos 2\theta,\ r\sin 2\theta)$ encoding edge strength and direction,
  5. Hessian polarity (ridge sign),
  6. blobness, $4\det(H)/\operatorname{tr}(H)^2$ (near 1 for isotropic blobs).

  These add no learnable parameters beyond a single sharpness scalar $\beta$.

The two similarities are fused before the softmax:

$$S_{\text{vis}} = \frac{QK^\top}{\sqrt{d_k}}, \qquad S_{\text{bnd}} = \Psi\Psi^\top$$

$$A = \operatorname{Softmax}\left(S_{\text{vis}} + \lambda S_{\text{bnd}}\right)$$

The per-head scalar $\lambda \in \mathbb{R}^h$ is parameterised as a square to enforce positivity. Large $\lambda$ biases attention toward pixels sharing boundary geometry; $\lambda \to 0$ recovers standard multi-head self-attention. A depthwise 3x3 convolution on $V$ supplies a local bias $V_{\text{loc}}$, giving the residual output:

$$Z_{\text{attn}} = Z_{\text{aspp}} + f_{\text{out}}\left(AV + V_{\text{loc}}\right)$$

### 4. Decoder and gated skip connections

The decoder mirrors the encoder over four upsampling stages. Each skip is filtered in two stages instead of being simply concatenated.

1. **Boundary gate:** a fixed Scharr edge detector boosts high-frequency content in the encoder feature:

$$S_l^{\text{bnd}} = S_l \odot \left(1 + \gamma\, \mathcal{F}_{\text{scharr}}(S_l)\right)$$

2. **Attention gate:** an Attention Gate suppresses irrelevant background using the upsampled decoder state $F_{\text{dec}}^{(l)}$:

$$\alpha_l = \sigma\left(W_\psi\, \operatorname{ReLU}\left(W_g F_{\text{dec}}^{(l)} + W_x S_l^{\text{bnd}}\right)\right), \qquad \hat{S}_l = S_l^{\text{bnd}} \odot \alpha_l$$

$\hat{S}_l$ is concatenated with the upsampled decoder feature, blended by a 1x1 convolution and refined by residual blocks.

### 5. Prediction heads

The final decoder output $D_1 \in \mathbb{R}^{32 \times H \times W}$ feeds:

- a **Seg head**: 1x1 convolution to a single logit $\hat{Y}$, sigmoid-thresholded at inference;
- two **auxiliary deep-supervision heads** on decoder stages 2 and 3, supervised with the same segmentation loss;
- an **HV offset head**: 1x1 convolution to 2 channels with tanh, predicting offsets $\hat{O} \in [-1,1]^{2 \times H \times W}$ toward each instance centroid.

Ground-truth offsets for each foreground pixel $(x, y)$ of instance $k$, with centroid $(c_x^k, c_y^k)$ and $\varepsilon = 10^{-8}$:

$$O_h(x,y) = \frac{x - c_x^k}{\max_j |x_j - c_x^k| + \varepsilon}, \qquad O_v(x,y) = \frac{y - c_y^k}{\max_j |y_j - c_y^k| + \varepsilon}$$

### 6. Inference: thresholding and marker-controlled watershed

1. Binarise $\hat{p} = \sigma(\hat{Y})$ at a threshold $\tau^*$ chosen by a 19-value grid search over [0.05, 0.95] that maximises validation Dice on cached predictions, giving $M = \mathbb{1}[\hat{p} > \tau^*]$.
2. Form an energy surface from Sobel gradients of the HV offset maps. It is low inside a nucleus and spikes where the offset field reverses at a boundary:

$$G = \sqrt{(\nabla_x \hat{O}^h)^2 + (\nabla_y \hat{O}^h)^2 + (\nabla_x \hat{O}^v)^2 + (\nabla_y \hat{O}^v)^2}$$

3. Pixels with low normalised gradient (< 0.4) inside $M$, eroded with a radius-2 disk, form labelled seeds.
4. Marker-controlled watershed floods $G$ from the seeds within $M$; fragments under 10 pixels are discarded.

## Training objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{hv}\mathcal{L}_{hv} + \lambda_{bnd}\mathcal{L}_{bnd}$$

**Segmentation loss.** The main prediction (decoder stage D1) is combined with the two auxiliary heads (D2, D3):

$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{seg}}^{(D_1)} + w_2 \mathcal{L}_{\text{seg}}^{(D_2)} + w_3 \mathcal{L}_{\text{seg}}^{(D_3)}$$

Each term is a Focal Tversky loss plus a class-weighted BCE. The positive weight is computed automatically from the mean foreground fraction $\bar{f}$ and capped by $\zeta$ for stability on very sparse masks:

$$w_{\text{pos}} = \min\left(\frac{1 - \bar{f}}{\bar{f}},\ \zeta\right)$$

**Instance-geometry loss** (following HoVer-Net), over foreground pixels $M$:

$$\mathcal{L}_{hv} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{MSGE}}$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{|M|}\sum_{i \in M} \left\|\hat{O}_i - O_i\right\|^2, \qquad \mathcal{L}_{\text{MSGE}} = \frac{1}{|M|}\sum_{i \in M} \left\|\nabla\hat{O}_i - \nabla O_i\right\|^2$$

**Boundary consistency loss.** Compares Scharr edge responses of predicted and ground-truth masks, supervising boundary sharpness independently of overall area overlap:

$$\mathcal{L}_{\text{bnd}} = \left\| \tanh\left(\kappa\, \mathcal{F}_{\text{scharr}}(\hat{Y})\right) - \tanh\left(\kappa\, \mathcal{F}_{\text{scharr}}(Y)\right) \right\|_1$$

## Experimental setup

**Datasets:** TNBC, MoNuSeg and CPM-15 (public histopathology datasets).

**Metrics:** Dice and IoU for semantic segmentation; Aggregated Jaccard Index (AJI) for instance-level separation.

| Setting | Value |
|---|---|
| Input size | 512 x 512, ImageNet normalisation |
| Epochs / batch size | 500 / 4 |
| Optimiser / schedule | AdamW / OneCycleLR |
| Gradient clipping | norm 1.0 |
| Precision | mixed (FP16) |
| EMA of weights | decay 0.999, used for validation and test |
| Augmentation | flips, rotations, affine jitter, colour jitter, Gaussian blur and noise, random erasing |
| SBA heads / descriptor channels | 4 / 6 |
| ASPP | rates {3, 6, 12}, global-pooling branch, dropout 0.1 |
| Gate scalar $\gamma$ | learnable, initialised to 0.1 |

Full hyperparameter values are in the code repository.

## Results

### Comparison with recent methods

Dice / IoU (%), as reported in the paper (Table 3).

**TNBC**

| Method | Dice | IoU |
|---|---|---|
| WDRE-NET | 75.63 | 60.81 |
| DyGLNet | 80.05 | 66.93 |
| SAAMU-Net | 84.10 | 72.60 |
| DeltaSegment | 82.95 | 70.92 |
| AFINITI | 85.00 | 73.91 |
| **BISON (ours)** | **86.86** | **76.78** |

**MoNuSeg**

| Method | Dice | IoU |
|---|---|---|
| WDRE-NET | 77.76 | 63.61 |
| EViT-UNet | 79.27 | 65.87 |
| VEGAN | 79.24 | 65.68 |
| DeltaSegment | 81.42 | 68.71 |
| Deepfuzz | 79.10 | 65.42 |
| **BISON (ours)** | **82.28** | **69.89** |

**CPM-15**

| Method | Dice | IoU |
|---|---|---|
| AFINITI | 85.00 | 73.91 |
| HoVer-Net | 80.00 | 66.67 |
| Micro-Net | 79.20 | 61.50 |
| U-Net++ | 75.42 | 60.54 |
| HI-Net | 84.80 | 73.61 |
| **BISON (ours)** | **89.23** | **80.55** |

![Dice comparison](assets/dice_comparison.png)

Snapshot of the paper's table:

![Table 3](assets/table3_comparison.png)

## Ablation studies

### Architectural components

A: Pre-Bottleneck Refinement gate; B: SBA Attention; C: Attention Gates; D: Boundary Loss. The baseline is a ResNet-UNet with the HV offset head.

| A | B | C | D | TNBC Dice | IoU | AJI | MoNuSeg Dice | IoU | AJI | CPM-15 Dice | IoU | AJI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | - | - | - | 80.10 | 68.80 | 48.30 | 77.20 | 63.50 | 49.30 | 83.10 | 73.80 | 56.30 |
| x | - | - | - | 81.40 | 70.10 | 51.10 | 78.50 | 65.10 | 52.10 | 84.40 | 75.10 | 59.10 |
| x | x | - | - | 83.20 | 72.50 | 55.50 | 80.20 | 67.20 | 56.40 | 86.20 | 77.50 | 63.50 |
| x | x | x | - | 84.90 | 74.60 | 57.60 | 81.30 | 68.50 | 58.20 | 87.90 | 79.20 | 65.80 |
| x | x | x | x | **86.86** | **76.78** | **59.33** | **82.28** | **69.89** | **60.12** | **89.23** | **80.55** | **67.69** |

![AJI ablation](assets/ablation_aji.png)

SBA attention gives the largest step (about +4.3 to +4.4 AJI on all three datasets), supporting the idea that boundary priors are key to separating touching nuclei.

### Loss components

| L_seg | L_MSE | L_aux | L_MSGE | L_bnd | TNBC Dice | IoU | AJI | MoNuSeg Dice | IoU | AJI | CPM-15 Dice | IoU | AJI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x | x | - | - | - | 83.10 | 71.40 | 52.40 | 79.10 | 65.80 | 53.40 | 85.10 | 76.40 | 60.10 |
| x | x | x | - | - | 84.90 | 73.20 | 54.20 | 80.50 | 67.30 | 55.20 | 86.90 | 78.20 | 62.20 |
| x | x | x | x | - | 85.30 | 75.40 | 57.40 | 81.60 | 68.70 | 58.60 | 88.30 | 79.60 | 65.40 |
| x | x | x | x | x | **86.86** | **76.78** | **59.33** | **82.28** | **69.89** | **60.12** | **89.23** | **80.55** | **67.69** |

Deep supervision improves all metrics; adding MSGE gives a substantial AJI jump (about +3.2 to +3.4), consistent with offset-gradient errors directly affecting watershed separation; the boundary consistency loss gives the final gain.

### Encoder backbone

MobileNet-v2, EfficientNet-B0, SqueezeNet, ResNet-34 and ResNet-50 were compared against the pre-activation ResNet-18 under an identical setup. Lightweight backbones underperform, ResNet-50 shows diminishing returns despite higher cost, and ResNet-18 performs best.

Snapshots of the paper's tables:

![Table 1](assets/table1_architecture_ablation.png)

![Table 2](assets/table2_loss_ablation.png)

## Qualitative results

![Qualitative results](assets/qualitative_results.jpg)

Columns, left to right: input image, ground-truth mask, predicted probabilities, binary mask, Grad-CAM overlay. Rows, top to bottom: TNBC, CPM-15, MoNuSeg. The per-image Dice printed in the panel titles is 0.8965, 0.9039 and 0.8603, with binarisation thresholds of 0.90, 0.85 and 0.80.

## Limitations and future work

**Limitations**

- HV supervision relies on a connected-component fallback when native instance-level annotations are missing, which can limit the precision of centroid predictions.
- Exact softmax SBA attention is restricted to the low-resolution bottleneck for computational tractability.
- The boundary descriptors use fixed filters that may not adapt to every imaging modality or staining protocol.

**Future work**

- Scalable approximations for boundary-aware attention at higher-resolution decoder stages.
- Learnable boundary filters adapted to diverse datasets.
- Evaluation across broader tumour types, with lightweight variants for rapid clinical deployment.

## Citation

Please check the final proceedings details before use; the entry below is based on the paper's front matter.

```bibtex
@inproceedings{chatterjee2026bison,
  title     = {{BISON}: Boundary-Injected Segmentation of Nuclei via Exact Softmax Attention},
  author    = {Chatterjee, Mukut and Bhattacharjee, Diptarko and Banerjee, Debatra and Kaplun, Dmitrii and Lyanova, Asya and Sarkar, Ram},
  booktitle = {7th International Conference on Medical Imaging and Computer-Aided Diagnosis (MICAD)},
  year      = {2026},
  address   = {Edinburgh, United Kingdom}
}
```

## Acknowledgements

This work was supported by the Russian Science Foundation (Project No. 26-19-00620).

## Contact

Corresponding author: Dmitrii Kaplun, dikaplun@etu.ru
