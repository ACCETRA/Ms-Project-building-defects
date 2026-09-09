**The Plan**

No single publicly available dataset covers the full range of conditions an infrastructure inspection system will meet in the field. This document lays out why the training strategy for this project combines several purpose-built datasets, namely CiF, DACL10K, CODEBRIM, and a UAV-based crack dataset, instead of relying on one, and how the resulting data is engineered, balanced, and validated so the combination actually helps rather than just adding noise.

**1\. Introduction**

The goal of this project is an AI-based infrastructure inspection system that can detect, localize, and measure structural defects, including concrete cracks, spalling, corrosion, surface deterioration, and related forms of damage, with minimal human review. Reaching that reliably means drawing on more than one sensing modality: RGB photographs, UAV imagery, 3D reconstructions, thermal scans, ground-penetrating radar (GPR), and historical inspection records, fused into a single engineering assessment rather than treated as separate outputs.

**2\. Why Combine Multiple Datasets**

Every dataset is a snapshot of whatever conditions its authors happened to photograph: particular structures, weather, camera equipment, and defect types. A model trained on only one of them tends to pick up on those incidental details, such as lighting, camera angle, or the local concrete mix, rather than the defect itself. Pooling several independently collected datasets forces the model to separate what actually indicates damage from what is simply an artifact of how one team gathered its data, which is the difference between a model that works in the lab and one that holds up on a bridge it has never seen.

**3\. What Each Dataset Contributes**

The four sources were chosen for what they cover that the others do not:

| **Dataset**                        | **Scale & Origin**                                                                                        | **Task Format**                                                                        | **What It Adds**                                                                                                                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CiF (Cracks in the Foundation)** | ~150,000 high-resolution images, curated over five years with civil engineers (IBM Research / ETH Zürich) | Instance segmentation                                                                  | Breadth and difficulty: the largest general civil-infrastructure segmentation set to date, and a genuinely hard benchmark, since even strong zero-shot foundation models plateau near 25% mAP on it. |
| ---                                | ---                                                                                                       | ---                                                                                    | ---                                                                                                                                                                                                  |
| **DACL10K**                        | 9,920 images from real-world inspections across 100+ bridges                                              | Multi-label semantic segmentation, with 12 damage classes plus 6 structural components | Realistic, bridge-specific damage variety, including defects that co-occur in the same image.                                                                                                        |
| ---                                | ---                                                                                                       | ---                                                                                    | ---                                                                                                                                                                                                  |
| **CODEBRIM**                       | Images from 30 bridges with varying deterioration levels                                                  | Multi-target classification: crack, spalling, exposed rebar, efflorescence, corrosion  | A clean, well-studied benchmark for defects that frequently overlap within one region of an image.                                                                                                   |
| ---                                | ---                                                                                                       | ---                                                                                    | ---                                                                                                                                                                                                  |
| **UAV-based crack dataset**        | Drone-captured imagery                                                                                    | Pixel-level crack segmentation                                                         | Aerial viewpoints and hard-to-reach structural areas that ground-level cameras rarely cover.                                                                                                         |
| ---                                | ---                                                                                                       | ---                                                                                    | ---                                                                                                                                                                                                  |

Put together, these datasets span three capture perspectives (ground-level, aerial, and a large curated multi-year collection), five defect categories beyond cracking, and two label formats: segmentation masks and multi-label classification. That combination is a meaningfully broader base than any single one of them offers alone.

**4\. Dataset Engineering: Making the Combination Work**

Combining datasets is a legitimate strategy for training, fine-tuning, and transfer learning, but only if it is treated as its own engineering problem rather than a folder merge. Each dataset needs to be profiled individually first, covering image quality, label format, class definitions, and annotation density, before any combination happens. In practice that means reconciling label formats (masks versus bounding boxes versus classification tags), removing duplicate or near-duplicate images across sources, and enforcing dataset-aware train/validation/test splits, so that near-duplicate images of the same structure cannot leak across a split and inflate reported performance.

**5\. Standardizing Labels Across Datasets**

The same defect shows up under different names depending on which dataset labeled it: CiF's crack instances, DACL10K's "ACrack" tag, and CODEBRIM's crack classification label all describe the same physical phenomenon. Before training starts, these need to be mapped onto one shared taxonomy, such as Crack, Spalling, Corrosion, Efflorescence, and Exposed Rebar, so the model reads a consistent label regardless of which dataset an image came from.

**6\. Handling Class Imbalance**

Crack images vastly outnumber examples of corrosion or severe spalling in every one of these datasets, and pooling them does not fix that; it just shifts where the imbalance sits. Left unaddressed, the model will default to a "probably a crack" answer whenever it is unsure. The plan is to measure the actual class distribution once the datasets are merged, not before, and then apply a combination of oversampling for underrepresented classes, augmentation (rotation, lighting shifts, blur, perspective warps) weighted more heavily toward minority classes, and a weighted loss function, such as weighted cross-entropy or focal loss, so training does not simply reward getting the common classes right.

**7\. Proposed AI Pipeline**

The inference pipeline is staged so that most images never touch the larger, more expensive model:

1. Input processing: quality checks, normalization, calibration, and image preparation.
2. Fast triage: Florence-2-Base (≈ 0.23B parameters) handles initial defect detection on the bulk of incoming images.
3. Escalation: Florence-2-Large (≈ 0.77B parameters) is invoked only for cases Florence-2-Base flags as uncertain or ambiguous.
4. Segmentation: a dedicated segmentation model extracts pixel-level defect boundaries and measurements once a defect is confirmed.
5. Evidence fusion: visual, 3D, thermal, GPR, and historical inspection data are combined into a single engineering report.

Both Florence-2 variants stay comfortably under the project's sub-2B-parameter budget, so the pipeline can run inference on a single consumer GPU rather than needing a dedicated inference cluster.

**8\. Evaluation Metrics**

Different stages of the pipeline are judged by different metrics:

| **Task**             | **Metrics**                                    |
| -------------------- | ---------------------------------------------- |
| **Detection**        | Precision, Recall, F1 score                    |
| ---                  | ---                                            |
| **Object detection** | mAP@50, mAP@50:95                              |
| ---                  | ---                                            |
| **Segmentation**     | IoU, Dice score                                |
| ---                  | ---                                            |
| **Measurement**      | MAE and RMSE for crack length, width, and area |
| ---                  | ---                                            |

**9\. Validation Strategy**

Testing only on held-out data from the same datasets used in training would overstate how well the model generalizes, since it would mostly measure whether it memorized that dataset's particular quirks. The validation plan instead holds out an independent test set and, separately, runs cross-dataset evaluation: for example, training on CiF and CODEBRIM and evaluating on a held-out slice of DACL10K the model has never seen. If performance holds up there, that is a much stronger signal that the model generalizes to structures and conditions it was not trained on.

**10\. Conclusion**

Combining CiF, DACL10K, CODEBRIM, and a UAV-based crack dataset into one standardized training set is the more defensible approach here, precisely because infrastructure defects vary so much by structure type, material, and inspection method that no single dataset captures all of it. CiF's own benchmark results underline this point, given that even strong general-purpose foundation models struggle on this data. Whether the combination actually pays off comes down to the less glamorous engineering work: consistent label mapping, real class-imbalance correction, and validation that is honest about testing on data the model has not seen before.