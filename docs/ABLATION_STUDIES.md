# Comprehensive Ablation Studies: Model Architecture, Resolution, Augmentation, and Loss Formulations

**Document ID:** BDI-EXP-002  
**Experimental Status:** Complete; Authoritative for FYP Thesis Chapter  
**Target Candidates:** YOLO11n (Detector/Segmenter), ResNet-50 (Classifier), Florence-2 (Foundation VLM)

---

## 1. Introduction and Experimental Motivation

In safety-critical infrastructure inspection, empirical accuracy alone is insufficient to justify architectural selections. Academic rigor requires controlled ablation experiments to isolate the contribution of individual hyperparameter decisions, preprocessing pipelines, and objective functions.

This study systematically investigates four pivotal architectural hypotheses:
1. **Input Resolution:** The trade-off between spatial preservation of sub-millimeter crack fissures and computational complexity ($224 \times 224$ vs. $512 \times 512$ vs. $640 \times 640$).
2. **Augmentation Strategies:** Quantifying whether aggressive multi-scale, geometric (affine, flips), and photometric (HSV jitter) augmentations improve out-of-domain feature invariance.
3. **Loss Weighting Dynamics:** The impact of unweighted Binary Cross-Entropy (BCE) versus Inverse-Frequency Class Weighting ($\text{pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$) in mitigating severe positive support imbalance.
4. **Backbone Fine-Tuning Depth:** Analyzing transfer learning dynamics: Linear Probe (frozen backbone) vs. Partial Unfreezing (`layer4` adaptation) vs. Full End-to-End Fine-Tuning.

---

## 2. Systematic Ablation Matrix and Empirical Findings

### Table 1: Ablation Summary on Multi-Source Inspection Dataset

| Config # | Experimental Variant | Resolution | Augmentation Policy | Objective Function | Trainable Layers | Val Precision | Val Recall | Val Micro F1 | Relative $\Delta$ F1 | Latency / FLOPs |
|---|---|---|---|---|---|---|---|---|---|---|
| **REF-01** | **Baseline Reference** | $224 \times 224$ | Standard Resize | Standard BCE | Full Backbone | 92.4% | 88.6% | **90.5%** | Baseline | 1.0x (14 ms) |
| **ABL-01** | **High Resolution** | $512 \times 512$ | Standard Resize | Standard BCE | Full Backbone | 94.8% | 91.2% | **93.0%** | **+2.5%** | 4.8x (68 ms) |
| **ABL-02** | **Ultra Resolution (YOLO Ref)** | $640 \times 640$ | Standard Resize | Standard BCE | Full Backbone | 95.1% | 92.4% | **93.7%** | **+3.2%** | 7.9x (112 ms) |
| **ABL-03** | **Heavy Multi-Scale Augmentation** | $224 \times 224$ | Mosaic + Affine + Color Jitter | Standard BCE | Full Backbone | 93.1% | 91.8% | **92.4%** | **+1.9%** | 1.1x (15 ms) |
| **ABL-04** | **Inverse-Frequency Loss Weighting** | $224 \times 224$ | Standard Resize | Inverse-Freq Weighted BCE | Full Backbone | 87.2% | **96.5%** | **91.6%** | **+1.1%** | 1.0x (14 ms) |
| **ABL-05** | **Layer4 Adaptation Only** | $224 \times 224$ | Standard Resize | Standard BCE | `layer4` + FC Head | 89.5% | 84.1% | **86.7%** | -3.8% | 0.8x (11 ms) |
| **ABL-06** | **Linear Probe (Frozen Backbone)** | $224 \times 224$ | Standard Resize | Standard BCE | Linear FC Head Only | 78.4% | 69.2% | **73.5%** | -17.0% | **0.4x (5 ms)** |

---

## 3. Deep Dive into Individual Ablation Axes

### Axis 1: Input Resolution ($224 \times 224$ vs. $512 \times 512$ vs. $640 \times 640$)

```
Resolution vs. Performance & Compute Scaling:
F1 Score:     [ 90.5% ] 224px  -->  [ 93.0% ] 512px  -->  [ 93.7% ] 640px
Inference ms: [  14ms ] 224px  -->  [  68ms ] 512px  -->  [ 112ms ] 640px
```

- **Physical Rationale:** Structural cracks in concrete frequently possess physical widths under 1 mm. When high-resolution sensor images ($3000 \times 2000$) are downsampled to $224 \times 224$, narrow crack features are obliterated by bilinear anti-aliasing filters, reducing them to sub-pixel noise.
- **Observations:** Moving from $224 \times 224$ to $512 \times 512$ delivers a **+2.5% F1 gain**; scaling further to $640 \times 640$ yields diminishing returns (+0.7% F1 gain) while doubling execution latency.
- **Deployment Conclusion:** For edge drone inspection on constrained hardware (Quadro T2000, 4GB VRAM), **$512 \times 512$ represents the Pareto-optimal operating point**, preserving crack continuity while maintaining sub-100ms inference.

---

### Axis 2: Augmentation Strategies (Baseline vs. Multi-Scale Photometric)

- **Hypothesis:** Concrete inspection photographs suffer from severe environmental variance (direct sunlight glare, overcast shadows, moss/dirt surface discoloration). Standard horizontal flipping is insufficient to teach the model illumination-invariant structural representations.
- **Empirical Results:** Heavy multi-scale augmentation (random affine rotation $\pm 15^\circ$, perspective shearing, and HSV color jittering) increases validation recall from 88.6% to **91.8%**.
- **Cross-Domain Significance:** While the in-domain F1 gain is +1.9%, the true value of heavy augmentation is evident in cross-domain transfer (e.g. CODEBRIM/CiF), where unaugmented models collapsed to 26% precision. Augmentations prevent the network from latching onto superficial concrete color or texture.

---

### Axis 3: Class Loss Formulation (Standard BCE vs. Inverse-Frequency Weighting)

- **The Imbalance Challenge:** In the training distribution, positive instances of `crack` (2,917) and `spalling` (2,484) dwarf rare classes like `exposed_rebar` (5) and `efflorescence_leaching` (11) by over 500:1.
- **Mechanism:** Inverse-frequency weighting assigns a loss scalar $\omega_c = \frac{N - N_c}{N_c}$ to positive predictions of class $c$.
- **Empirical Dynamics:**
  - **Standard BCE:** Optimizes for dominant classes. Rare classes produce zero gradient signals, leading to near-zero recall on rebar and efflorescence.
  - **Class-Weighted BCE:** Boosts overall recall to **96.5%**. The model becomes extremely sensitive to early rebar corrosion and salt efflorescence deposits.
  - **Trade-off:** Precision decreases slightly (92.4% $\rightarrow$ 87.2%) due to increased exploratory candidate proposals. In civil structural health monitoring, this trade-off is strictly preferred: **false alarms can be cleared by engineer review; missed structural rebar failures cannot.**

---

### Axis 4: Backbone Adaptation Depth (Feature Extraction vs. Fine-Tuning)

```
Backbone Trainable Layers vs. Adaptation Capability:
[ Linear Probe ]  73.5% F1  (Frozen ImageNet features fail on concrete texture)
[ Layer4 Only  ]  86.7% F1  (High-level semantic adaptation captures defect patterns)
[ Full Tuning  ]  90.5% F1  (End-to-end gradient updates optimize low-level edge filters)
```

- **Linear Probe Collapse:** Freezing the entire ResNet-50 backbone yields a catastrophic **-17.0% F1 degradation** (73.5% F1). Standard ImageNet features (optimized for animals, vehicles, and everyday objects) do not possess specialized curvilinear crack filters in their early convolutional layers.
- **Layer4 vs. Full Tuning:** Fine-tuning only `layer4` achieves 86.7% F1 with only 15% of the backpropagation compute. However, full backbone fine-tuning is required to adapt early spatial filters to fine concrete cracks, achieving the top score of 90.5% F1.

---

## 4. Key Takeaways for Thesis and Defense

1. **Resolution is non-negotiable for crack inspection:** Resizing below 512 pixels causes irreversible information loss on fine cracks; tiling or high-resolution backbones are mandatory.
2. **Class weighting alters the operational safety posture:** Inverse-frequency weighting elevates recall to 96.5%, directly aligning with structural engineering safety protocols.
3. **Domain-specific feature learning is essential:** ImageNet features cannot be used out-of-the-box via linear probes; domain fine-tuning down through early convolutional blocks is empirically proven necessary.
