# On-Site Real-World Field Inspection Protocol

**Protocol ID:** BDI-PRO-003  
**Application Scope:** Visual Structural Defect Assessment of Reinforced Concrete Facades, Retaining Walls, and Civil Infrastructure  
**Operational Standard:** Human-in-the-Loop AI-Assisted Structural Triage

---

## 1. Objective and Scope

This protocol defines the standardized procedure for acquiring, calibrating, and processing uncurated field imagery using the Building Defect Inspection (BDI) multi-pipeline system. 

The primary objective is to bridge the gap between academic benchmark datasets and real-world civil engineering site surveys. In real-world deployment, model predictions serve exclusively as **immutable candidate findings** requiring structural engineer sign-off, adhering to `schemas/finding_record.schema.json`.

---

## 2. On-Site Image Acquisition Guidelines

### 2.1 Environmental and Lighting Constraints
- **Illumination:** Diffuse daylight (overcast or morning/late afternoon indirect sunlight) is strictly preferred. Harsh midday direct sunlight creates deep cast shadows across crack fissures and causes specular clipping on glossy concrete coatings.
- **Surface Condition:** Inspection surfaces must be dry. Standing water or rain streaks mask crack paths and trigger false-positive efflorescence or rust leaching detections.
- **Obstructions:** Surface dust, spiderwebs, and loose vegetation should be brushed away when assessing suspected hairline cracks.

### 2.2 Camera Geometry and Sensor Configuration
- **Standoff Distance:** $0.5\text{ m} \le d \le 3.0\text{ m}$ depending on safe access and focal length.
- **Angle of Incidence:** Camera optical axis should be perpendicular ($\theta \le 15^\circ$) to the concrete plane to minimize perspective foreshortening.
- **Ground Sample Distance (GSD):** Ensure resolution reaches $\le 0.4\text{ mm/pixel}$ across the target region. At $0.4\text{ mm/pixel}$, a critical $0.2\text{ mm}$ structural crack spans at least 0.5 pixels, detectable via local intensity gradients.
- **Motion Blur Prevention:** Minimum shutter speed of $1/250\text{ s}$ for handheld inspection; $1/500\text{ s}$ for UAV aerial hovering.

---

## 3. Physical Metric Calibration Protocol

Physical dimension estimation (crack length, spall area) requires rigorous geometric scaling:

1. **Option A: Metric Reference Target (Recommended)**
   - Place a high-contrast metric scale bar or fiducial target (e.g. 50 mm circular marker) on the concrete surface within the same depth plane as the defect.
   - Record `millimeters_per_pixel` explicitly:
     $$\text{GSD} = \frac{\text{Physical Dimension (mm)}}{\text{Measured Pixel Distance (px)}}$$
2. **Option B: Fixed Planar Homography**
   - For planar facades captured at oblique angles, compute a $3 \times 3$ homography matrix $H$ using 4 known coplanar control points.
3. **Uncalibrated Fallback:**
   - In the absence of a verified physical scale target, the system strictly tags the finding with limitation `uncalibrated_measurement` and reports dimensions purely in pixels. Fabricating metric dimensions is strictly forbidden by project contract (`docs/OUTPUT_CONTRACT.md`).

---

## 4. Automated Image Quality Preflight

Before model inference, every raw field image is validated against deterministic quality thresholds:
- **Low Resolution:** $\min(\text{width}, \text{height}) < 512\text{ px} \implies$ `low_resolution` warning.
- **Exposure Quality:** Mean luma $\mu < 35 \implies$ `underexposed`; $\mu > 220 \implies$ `overexposed`.
- **Blur Detection:** Laplacian edge variance $\sigma^2 < 45 \implies$ `blur` warning.

Images triggering quality flags are accepted into the pipeline with explicit warning tags (`accepted_with_warning`) to alert the inspecting engineer.

---

## 5. Automated Triage and Human Review Workflow

```
[ Field Camera Capture ] 
          │
          ▼
[ Quality Preflight Check (Luma / Blur / GSD) ]
          │
          ▼
[ Multi-Model Inference (YOLO11n + ResNet-50 + Florence-2) ]
          │
          ▼
[ Immutable Finding Records Generated (finding_record.schema.json) ]
          │
          ▼
[ Inspector Review Dashboard (Browser Harness) ]
    ├─ Accepted   --> Structural Maintenance Work Order
    ├─ Adjusted   --> Geometry / Label Corrected by PE
    ├─ Rejected   --> Spurious / Surface Artifact Cleared
    └─ Escalated  --> Non-Destructive Testing (NDT) / Core Sampling
```

1. **Inference Pipeline:**
   - **YOLO11n:** Proposes rapid candidate bounding boxes and polygon masks for cracks, spalling, and rebar.
   - **ResNet-50:** Provides calibrated multi-label defect probabilities and Grad-CAM attention maps.
   - **Florence-2:** Performs zero-shot phrase grounding to capture unmodeled or compound degradation phenomena.
2. **Review Record:**
   - Model predictions remain immutable under `finding.model_prediction`.
   - The reviewing Professional Engineer (PE) attaches `finding.review` (`status`, `reviewer_id`, `timestamp`, `engineering_notes`).
