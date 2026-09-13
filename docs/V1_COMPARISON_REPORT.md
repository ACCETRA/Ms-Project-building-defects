# V1 Comparison Report

Generated from the completed v1 queue artifacts. This report uses train/validation outputs only; no locked test result is included here.

## Run status

| Model | Status | Checkpoint or output |
|---|---|---|
| YOLO11n detection | Complete, 20 epochs | `runs/detect/runs/comparison/yolo/yolo11n_detect_v1_queue/weights/best.pt` |
| YOLO11n segmentation | Complete, 20 epochs | `runs/segment/runs/comparison/yolo/yolo11n_seg_v1_queue/weights/best.pt` |
| ResNet-50 multilabel | Complete, 10 epochs | `runs/comparison/resnet50_v1_queue/resnet50_comparison.pt` |
| Florence smoke fine-tune | One step complete | `runs/florence/v1_queue_smoke/checkpoint-last/` |

## YOLO final validation row

The values below are the final rows written by Ultralytics `results.csv`.

### Detection

```text
{
  "epoch": "20",
  "time": "9442.05",
  "train/box_loss": "1.08524",
  "train/cls_loss": "0.80775",
  "train/dfl_loss": "0.98169",
  "metrics/precision(B)": "0.93166",
  "metrics/recall(B)": "0.25744",
  "metrics/mAP50(B)": "0.28038",
  "metrics/mAP50-95(B)": "0.22334",
  "val/box_loss": "0.97745",
  "val/cls_loss": "0.79626",
  "val/dfl_loss": "0.94516",
  "lr/pg0": "5.95e-05",
  "lr/pg1": "5.95e-05",
  "lr/pg2": "5.95e-05"
}
```

### Segmentation

```text
{
  "epoch": "20",
  "time": "15763.9",
  "train/box_loss": "1.08127",
  "train/seg_loss": "1.20109",
  "train/cls_loss": "0.8068",
  "train/dfl_loss": "0.96501",
  "train/sem_loss": "0",
  "metrics/precision(B)": "0.93087",
  "metrics/recall(B)": "0.25834",
  "metrics/mAP50(B)": "0.28699",
  "metrics/mAP50-95(B)": "0.22317",
  "metrics/precision(M)": "0.90131",
  "metrics/recall(M)": "0.24466",
  "metrics/mAP50(M)": "0.25581",
  "metrics/mAP50-95(M)": "0.16803",
  "val/box_loss": "1.0167",
  "val/seg_loss": "1.16102",
  "val/cls_loss": "0.88703",
  "val/dfl_loss": "0.94912",
  "val/sem_loss": "0",
  "lr/pg0": "5.95e-05",
  "lr/pg1": "5.95e-05",
  "lr/pg2": "5.95e-05"
}
```

The YOLO runs completed training and saved checkpoints. The original queue return code was affected by the Windows apostrophe path sanitizer during final checkpoint reload; it was not a training-epoch failure.

## ResNet validation history

```json
[
  {
    "epoch": 1.0,
    "train_loss": 0.08631787826205888,
    "val_loss": 0.029030248004015328
  },
  {
    "epoch": 2.0,
    "train_loss": 0.02120212589690472,
    "val_loss": 0.024602747223195984
  },
  {
    "epoch": 3.0,
    "train_loss": 0.01487257721200349,
    "val_loss": 0.018589258620680914
  },
  {
    "epoch": 4.0,
    "train_loss": 0.01309329589895387,
    "val_loss": 0.019771880918086394
  },
  {
    "epoch": 5.0,
    "train_loss": 0.010391724417970998,
    "val_loss": 0.03179740175960762
  },
  {
    "epoch": 6.0,
    "train_loss": 0.008042704229871784,
    "val_loss": 0.07840504021256177
  },
  {
    "epoch": 7.0,
    "train_loss": 0.0045164107478489895,
    "val_loss": 0.030286215119508266
  },
  {
    "epoch": 8.0,
    "train_loss": 0.0034719562609163827,
    "val_loss": 0.07962772635935347
  },
  {
    "epoch": 9.0,
    "train_loss": 0.003633890280195993,
    "val_loss": 0.03936534128496305
  },
  {
    "epoch": 10.0,
    "train_loss": 0.002255435134508804,
    "val_loss": 0.040474141743239975
  }
]
```

Prediction-level metrics and CODEBRIM test evaluation, with thresholds fitted on v1 validation only:

```json
{
  "checkpoint": "D:\\ALI's Project\\runs\\comparison\\resnet50_v1_queue\\resnet50_comparison.pt",
  "device": "cuda",
  "classes": [
    "crack",
    "spalling",
    "honeycombing_rock_pocket",
    "exposed_rebar",
    "rust_staining",
    "efflorescence_leaching"
  ],
  "threshold_selection": {
    "source": "v1 group-safe validation only",
    "seed": 20260911,
    "val_fraction": 0.2
  },
  "validation": {
    "samples": 1104,
    "thresholds": {
      "crack": 0.05000000074505806,
      "spalling": 0.699999988079071,
      "honeycombing_rock_pocket": 0.05000000074505806,
      "exposed_rebar": 0.05000000074505806,
      "rust_staining": 0.20000000298023224,
      "efflorescence_leaching": 0.15000000596046448
    },
    "per_class": {
      "crack": {
        "precision": 0.9915966386554622,
        "recall": 1.0,
        "f1": 0.9957805907172996,
        "support": 590.0,
        "pr_auc": 0.9834674665471652
      },
      "spalling": {
        "precision": 1.0,
        "recall": 0.9941747572815534,
        "f1": 0.9970788704965919,
        "support": 515.0,
        "pr_auc": 0.9987277312826917
      },
      "honeycombing_rock_pocket": {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1.0,
        "pr_auc": 1.0
      },
      "exposed_rebar": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0.0,
        "pr_auc": 0.0
      },
      "rust_staining": {
        "precision": 1.0,
        "recall": 0.6666666666666666,
        "f1": 0.8,
        "support": 12.0,
        "pr_auc": 0.7384640224750698
      },
      "efflorescence_leaching": {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 0.6666666666666666,
        "support": 2.0,
        "pr_auc": 0.8333333333333333
      }
    },
    "micro": {
      "precision": 0.9937444146559428,
      "recall": 0.9928571428571429,
      "f1": 0.9933005806163465,
      "support": 1120.0,
      "pr_auc": 0.9937393716329335
    }
  },
  "codebrim_test": {
    "samples": 632,
    "thresholds": {
      "crack": 0.05000000074505806,
      "spalling": 0.699999988079071,
      "honeycombing_rock_pocket": 0.05000000074505806,
      "exposed_rebar": 0.05000000074505806,
      "rust_staining": 0.20000000298023224,
      "efflorescence_leaching": 0.15000000596046448
    },
    "per_class": {
      "crack": {
        "precision": 0.2442622950819672,
        "recall": 0.9933333333333333,
        "f1": 0.39210526315789473,
        "support": 150.0,
        "pr_auc": 0.4347884059174019
      },
      "spalling": {
        "precision": 0.3333333333333333,
        "recall": 0.7666666666666667,
        "f1": 0.4646464646464646,
        "support": 150.0,
        "pr_auc": 0.44393136302200387
      },
      "honeycombing_rock_pocket": {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0.0,
        "pr_auc": 0.0
      },
      "exposed_rebar": {
        "precision": 0.5042016806722689,
        "recall": 0.4,
        "f1": 0.44609665427509293,
        "support": 150.0,
        "pr_auc": 0.4658671565305695
      },
      "rust_staining": {
        "precision": 0.3405994550408719,
        "recall": 0.8333333333333334,
        "f1": 0.48355899419729204,
        "support": 150.0,
        "pr_auc": 0.5331573119832044
      },
      "efflorescence_leaching": {
        "precision": 0.46153846153846156,
        "recall": 0.24161073825503357,
        "f1": 0.31718061674008813,
        "support": 149.0,
        "pr_auc": 0.41408970658139554
      }
    },
    "micro": {
      "precision": 0.2648809523809524,
      "recall": 0.5941255006675568,
      "f1": 0.36640592836558256,
      "support": 749.0,
      "pr_auc": 0.3685639128430229
    }
  }
}
```

## Florence smoke result

```json
[
  {
    "epoch": 1.0,
    "train_loss_scaled": 0.9773433208465576,
    "validation_loss": 6.3952131271362305
  }
]
```

This is a one-step smoke fine-tune, not an accuracy benchmark.

## Locked CUBIT test results

These metrics use confidence thresholds selected on the v1 validation split only:

```json
{
  "detection": {
    "confidence": 0.4,
    "precision": 0.35815716228253336,
    "recall": 0.24630367053713087,
    "f1": 0.29188135577899105,
    "metrics": {
      "metrics/precision(B)": 0.35815716228253336,
      "metrics/recall(B)": 0.24630367053713087,
      "metrics/mAP50(B)": 0.24447134201843754,
      "metrics/mAP50-95(B)": 0.20657641617489417,
      "fitness": 0.20657641617489417
    }
  },
  "segmentation": {
    "confidence": 0.4,
    "precision": 0.3584829422117674,
    "recall": 0.24587694442703945,
    "f1": 0.29168941357261996,
    "metrics": {
      "metrics/precision(B)": 0.3584829422117674,
      "metrics/recall(B)": 0.24587694442703945,
      "metrics/mAP50(B)": 0.2404814409405549,
      "metrics/mAP50-95(B)": 0.20752409308026173,
      "metrics/precision(M)": 0.328972824306733,
      "metrics/recall(M)": 0.23243059471677446,
      "metrics/mAP50(M)": 0.22332441574759465,
      "metrics/mAP50-95(M)": 0.1556852923116548,
      "fitness": 0.3632093853919165
    }
  }
}
```

The evaluator read the immutable test archives through a temporary extraction and verified that the source archives were untouched.
## YOLO-to-SAM comparison

```json
{
  "schema_version": 1,
  "run_id": "yolo-sam-v1-sample-10",
  "status": "completed",
  "scope": "training_only_feasibility_not_accuracy_evaluation",
  "samples_processed": 10,
  "findings_emitted": 79,
  "manifest": "data\\manifests\\v1_detection_manifest.csv",
  "yolo_weights": "runs\\detect\\runs\\comparison\\yolo\\yolo11n_detect_v1_queue\\weights\\best.pt",
  "sam_checkpoint": "weights\\sam2.1-hiera-tiny",
  "environment": {
    "python": "3.11.9",
    "torch": "2.6.0+cu124",
    "gpu": "Quadro T2000",
    "cuda": "12.4"
  },
  "elapsed_seconds": 86.3391,
  "limitations": [
    "training-only feasibility input",
    "YOLO confidence is not calibrated",
    "SAM mask is box-prompted and requires manual review"
  ]
}
```

This is a frozen-v1 training-image route smoke sample, not an accuracy evaluation. Full source-specific SAM scoring requires source-specific test evaluators and ground-truth adapters.

## Florence structured inference

```json
{
  "schema_version": 1,
  "run_id": "florence-v1-validation-sample-10",
  "generated_at_utc": "2026-09-13T06:40:34.413385Z",
  "scope": "training_only_feasibility_not_accuracy_evaluation",
  "status": "completed",
  "samples_processed": 10,
  "findings_emitted": 11,
  "images_escalated": 0,
  "images_with_candidates": 10,
  "images_without_candidates": 0,
  "schema_valid_findings": 11,
  "rejected_generated_geometries": 0,
  "physical_findings": 0,
  "escalation_policy": "none",
  "uncertain_below_sequence_proxy": 0.35,
  "manifest": "data/manifests/v1_florence_product_manifest.csv",
  "checkpoint_sha256": {
    "florence-2-base-ft": "1b2c35db0e11e5e48d0de7ede6e969e5a05066520872cfa79939f7efe18b3d79",
    "florence-2-large-ft": "d8609f64629a49b28afb8feda4cea22da05247437e1b09e295d8f39d7d7d8dc0"
  },
  "environment": {
    "python": "3.11.9",
    "torch": "2.6.0+cu124",
    "gpu": "Quadro T2000",
    "cuda": "12.4"
  },
  "elapsed_seconds": 31.343,
  "outputs": {
    "findings": "findings.jsonl",
    "image_results": "image_results.jsonl"
  },
  "limitations": [
    "All source samples are training-only; this run cannot establish accuracy.",
    "Florence sequence likelihood is not a calibrated per-box confidence.",
    "Physical dimensions are absent unless calibration is explicitly valid.",
    "Candidate visible conditions require manual review and are not structural-safety determinations."
  ]
}
```

This is a bounded training-only validation sample. It verifies structured parsing, schema validation, throughput, and checkpoint provenance; it is not an accuracy score.

```json
{
  "detect": {
    "dataset": "CUBIT-InSeg locked test",
    "test_images": 701,
    "task": "detect",
    "weights": "D:\\ALI's Project\\runs\\detect\\runs\\comparison\\yolo\\yolo11n_detect_v1_queue\\weights\\best.pt",
    "confidence": 0.4,
    "metrics": {
      "metrics/precision(B)": 0.8999719372409233,
      "metrics/recall(B)": 0.6419838308457712,
      "metrics/mAP50(B)": 0.6316537148965947,
      "metrics/mAP50-95(B)": 0.5410716219241134,
      "fitness": 0.5410716219241134
    },
    "source_archives_untouched": true
  },
  "segment": {
    "dataset": "CUBIT-InSeg locked test",
    "test_images": 701,
    "task": "segment",
    "weights": "D:\\ALI's Project\\runs\\segment\\runs\\comparison\\yolo\\yolo11n_seg_v1_queue\\weights\\best.pt",
    "confidence": 0.4,
    "metrics": {
      "metrics/precision(B)": 0.8991675988350415,
      "metrics/recall(B)": 0.6417309621248596,
      "metrics/mAP50(B)": 0.6316945586365987,
      "metrics/mAP50-95(B)": 0.5404190928841374,
      "metrics/precision(M)": 0.8273548588646693,
      "metrics/recall(M)": 0.607490771946718,
      "metrics/mAP50(M)": 0.5834126112620105,
      "metrics/mAP50-95(M)": 0.404962380891038,
      "fitness": 0.9453814737751753
    },
    "source_archives_untouched": true
  }
}
```

The locked evaluation covers YOLO detection and segmentation only. No ResNet or Florence locked-test score is reported because their compatible test-task evaluator is not implemented in this repository.

## Dataset and split policy

- Manifest: `data/manifests/v1_master_manifest.csv`
- Training/validation rows: 5,520
- Locked test rows in v1 manifest: 0
- Validation selection: group-safe, deterministic seed `20260911`
- Locked CUBIT test: evaluated separately by `scripts/evaluate_cubit_test.py`
- Measurements: pixel-only unless a valid calibration method is supplied

## Provenance and licenses

Checkpoint SHA-256:

```json
{
  "yolo_detection": "916beaebf3b1e1a101ebf4eee054abc391ffc64612125697f05dc2d5ef68211f",
  "yolo_segmentation": "e160b40f3071bc53ebed59507311656377859cb493aa9925b5400841995bd507",
  "resnet": "92b34d3b45ecc646904bb73f9992f656e554f5f834c625be30e062bfa7a900b8"
}
```

Manifest SHA-256:

```json
{
  "v1_master_manifest.csv": "1401ea78a4739fab51c61454e031ee05dcd614c420cb6cac3cc3522f496f01b2",
  "v1_detection_manifest.csv": "7eb1390bc941ed0326bbb69e8264480e855a6c78dc44781e1d64e128cc3ea3db",
  "v1_segmentation_manifest.csv": "18b4d3fc665551fcd6f2b597de352a187035900684620bcc003f567f28244f14",
  "v1_classification_manifest.csv": "5907eef010ff284b5e7bdf773d9fd5fa312e8b8aff696ed08bb326efffb94452"
}
```

Source licenses recorded in the v1 manifest:

```json
{
  "CUBIT-InSeg": "CC-BY-4.0",
  "CiF-tiled": "CDLA-Permissive-2.0",
  "DACL10K-v2-devphase": "CC BY-NC 4.0",
  "S2DS": "GPL-3.0 repository; verify dataset terms before distribution",
  "UAV75": "GPL-3.0 repository; verify dataset terms before distribution"
}
```

## Source-specific evaluation status

| Source | Status |
|---|---|
| CUBIT-InSeg | Locked YOLO detection and segmentation evaluation completed on 701 archived test images |
| CODEBRIM | ResNet official test evaluation completed on 632 crops |
| CiF tiled | Adapter verified on 100-record smoke subset; full 2,500-record run remains a long-run job |
| S2DS | 93-image binary foreground proxy; class identity is unavailable in supplied test masks |
| UAV75 | 15-image crack-mask evaluation completed |
| DACL10K | 975-image validation evaluation completed with six documented mappings |
| Target-site buildings | No approved target-site test manifest present |

Machine-readable source adapter results:

```json
{
  "uav75": {
    "source": "uav75",
    "task": "segment",
    "samples": 15,
    "metrics": {
      "metrics/precision(B)": 0.5505146135834218,
      "metrics/recall(B)": 0.038461538461538464,
      "metrics/mAP50(B)": 0.052907564505609,
      "metrics/mAP50-95(B)": 0.032540072619089006,
      "metrics/precision(M)": 0.0012515644555694619,
      "metrics/recall(M)": 0.038461538461538464,
      "metrics/mAP50(M)": 4.72972972972973e-05,
      "metrics/mAP50-95(M)": 2.8378378378378378e-05,
      "fitness": 0.032568450997467385
    },
    "test_archive_untouched": true,
    "s2ds_semantics": null
  },
  "s2ds": {
    "source": "s2ds",
    "task": "segment",
    "samples": 93,
    "metrics": {
      "metrics/precision(B)": 0.13463264305676614,
      "metrics/recall(B)": 0.0549738219895288,
      "metrics/mAP50(B)": 0.026248217806763587,
      "metrics/mAP50-95(B)": 0.010256257740476253,
      "metrics/precision(M)": 0.047152218377986596,
      "metrics/recall(M)": 0.020942408376963352,
      "metrics/mAP50(M)": 0.0069813553509788994,
      "metrics/mAP50-95(M)": 0.00143851414606396,
      "fitness": 0.011694771886540212
    },
    "test_archive_untouched": true,
    "s2ds_semantics": "binary_foreground_proxy; class identity is unavailable"
  },
  "dacl": {
    "source": "dacl",
    "task": "segment",
    "samples": 975,
    "metrics": {
      "metrics/precision(B)": 0.5773821176022792,
      "metrics/recall(B)": 0.024474855999019565,
      "metrics/mAP50(B)": 0.007766376325350777,
      "metrics/mAP50-95(B)": 0.0030004449606881893,
      "metrics/precision(M)": 0.5605095120160996,
      "metrics/recall(M)": 0.012487574383485162,
      "metrics/mAP50(M)": 0.003391086082279811,
      "metrics/mAP50-95(M)": 0.0007641298294322618,
      "fitness": 0.003764574790120451
    },
    "test_archive_untouched": true,
    "s2ds_semantics": null
  },
  "cif": {
    "source": "cif",
    "task": "segment",
    "samples": 100,
    "metrics": {
      "metrics/precision(B)": 0.46353140432454126,
      "metrics/recall(B)": 0.09350036310820624,
      "metrics/mAP50(B)": 0.0841296169789436,
      "metrics/mAP50-95(B)": 0.03664770034159003,
      "metrics/precision(M)": 0.44474565663735505,
      "metrics/recall(M)": 0.07855989123790512,
      "metrics/mAP50(M)": 0.06406558799522967,
      "metrics/mAP50-95(M)": 0.019950568927762747,
      "fitness": 0.05659826926935278
    },
    "test_archive_untouched": true,
    "s2ds_semantics": null
  }
}
```

## Limitations

This report is a training-readiness and validation summary. It is not a deployment accuracy claim. Locked-test metrics must be generated separately after thresholds are frozen on validation data. Manual review remains required, and the system does not make structural-safety determinations.
