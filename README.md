# Supervised Contrastive Multimodal Learning for Clinically Feasible Alzheimer's Disease Classification

## Overview

We propose a clinically feasible multimodal framework for three-class Alzheimer's disease staging (**CN / MCI / AD**) that integrates **structural MRI (sMRI)** and **routinely collected tabular clinical data**, avoiding costly or invasive modalities such as PET, CSF biomarkers, or genomic profiling.

Key components:

- **DINO-based self-supervised pretraining** adapts the MRI encoder to the target data distribution before any supervised training.
- **MedSigLIP** vision encoder provides strong domain-aware image representations (1152-d → 512-d after projection).
- **Supervised Contrastive Learning (SupCon)** is applied exclusively to the tabular branch to improve class-wise feature separability (256-d embeddings).
- **Intermediate fusion** concatenates both modalities [512 ‖ 256 = 768-d] and trains an MLP classifier end-to-end with gradients flowing back into the partially unfrozen vision encoder.

Experiments on **ADNI** (n = 1,044) demonstrate performance comparable to SOTA methods that rely on up to four modalities.

---

## Architecture


**MRI branch** — 2D axial slices (skull-stripped, 448×448, normalised to [−1, 1]). Central slices targeting medial temporal lobe, hippocampal structures, and ventricular areas are selected automatically. Slice embeddings are mean-pooled per patient.

**Tabular branch** — MoCA score, demographics, medication history, family history, and other routinely available clinical variables. SupCon loss is applied as an auxiliary objective during tabular pre-training to encourage class-discriminative clustering.

**Fusion** — Embeddings are concatenated and passed through a 3-layer fusion MLP (768 → 512 → 256 → 3). The vision encoder's last 2 transformer blocks are fine-tuned jointly with the fusion head using a differential learning rate.

---

## Results

### Ablation Study (CN / MCI / AD)

| Modality    | Model                        | ACC    | AUROC  | AUPRC  | F1     | MCC    |
|-------------|------------------------------|--------|--------|--------|--------|--------|
| Tabular     | LightGBM                     | 0.5869 | 0.8032 | 0.6953 | 0.5929 | 0.3124 |
| Tabular     | MLP                          | 0.6348 | 0.8094 | 0.7050 | 0.6344 | 0.3975 |
| Tabular     | MLP w/ SupCon                | 0.6782 | 0.8369 | 0.7241 | 0.6739 | 0.4965 |
| Tabular     | DeepGBM                      | 0.5956 | 0.6118 | 0.7300 | 0.6725 | 0.4374 |
| Tabular     | XGBoost                      | 0.5869 | 0.7910 | 0.7028 | 0.5929 | 0.3459 |
| MRI         | MedSigLIP (Linear Probing)   | 0.4708 | 0.6514 | 0.4616 | 0.4402 | 0.1929 |
| MRI         | MedSigLIP (Partial Finetuning)| 0.5047 | 0.6303 | 0.4374 | 0.4682 | 0.2079 |
| MRI         | MedSigLIP (DINO Pretraining) | 0.5376 | 0.6615 | 0.4652 | 0.4569 | 0.2542 |
| Multimodal  | Baseline Fusion              | 0.6716 | 0.8478 | 0.6879 | 0.6544 | 0.5143 |
| Multimodal  | **Fusion w/ Tabular SupCon** | **0.6961** | **0.8548** | **0.7191** | **0.7039** | **0.5453** |

### Comparison with State-of-the-Art

| Task        | Method                     | M | ACC    | AUROC  | F1     |
|-------------|----------------------------|---|--------|--------|--------|
| CN/AD       | Baseline Concat.           | 2 | 0.9130 | 0.9224 | 0.8387 |
| CN/AD       | Baseline Concat. w/ TTA    | 2 | 0.9391 | 0.9743 | 0.8852 |
| CN/AD       | Yi et al. (UniCross)       | 3 | 0.9357 | 0.9704 | 0.9230 |
| CN/MCI/AD   | **Fusion w/ Tabular SupCon** | **2** | **0.6961** | **0.8548** | **0.6849** |
| CN/MCI/AD   | Huang et al.               | 4 | 0.7321 | 0.8625 | 0.6868 |

*M = number of modalities. Our method uses only 2 modalities (sMRI + tabular).*

---

## Repository Structure

```
.
├── dino_pretraining/
│   └── dino_pretrain.py            # DINO self-supervised MRI pretraining
├── multimodal/
│   └── multimodal_fusion_pipeline.py  # SupCon tabular pretraining + end-to-end fusion
├── tabular_baselines/
│   └── tabular_baselines.py        # LightGBM / XGBoost / MLP / DeepGBM baselines
└── config_<RUN_ID>.json            # Saved run configuration and results
```


## Data

This study uses the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset.

> Data used in preparation of this article were obtained from the ADNI database (adni.loni.usc.edu). Access requires registration at [https://adni.loni.usc.edu](https://adni.loni.usc.edu). We are unable to distribute raw data directly.

**Cohort** — 1,044 patients: CN (417), MCI (469), AD (158). Split 60/20/20 at the patient level.

**MRI preprocessing:**
1. Skull-strip scans
2. Resize slices to 448 × 448
3. Normalise intensities to [−1, 1]
4. Automated central axial slice selection (medial temporal lobe, hippocampus, ventricles)

**Tabular features** — MoCA score, age, sex, education level, medication status, family history, and other routinely available clinical variables.

**Expected directory layout:**

```
data/
├── mri_patients/
│   ├── sub-001/
│   │   ├── slice_001.png
│   │   └── slice_002.png
│   └── sub-002/
│       └── ...
├── mri_labels.csv          # columns: subject_id, DIAGNOSIS (1/2/3 or 0/1/2)
├── splits.json             # {"train": [...], "val": [...], "test": [...]}
└── tabular/
    ├── train.csv
    ├── val.csv
    └── test.csv

checkpoints/
└── vision_encoder.pth      # MedSigLIP weights (key: "student_backbone")
```

---

## Usage

### Step 1 — DINO Self-Supervised Pretraining

Adapt the MedSigLIP encoder to the target MRI distribution before supervised training.

```bash
python dino_pretraining/dino_pretrain.py
```

Key hyperparameters (top of file):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_EPOCHS` | 100 | Total pretraining epochs |
| `BATCH_SIZE` | 32 | Slices per batch |
| `N_UNFREEZE` | 2 | Last N transformer blocks unfrozen |
| `N_LOCAL_CROPS` | 6 | Local crops per image |
| `TEMP_STUDENT` | 0.1 | Student softmax temperature |
| `TEMP_TEACHER_BASE` | 0.04 | Teacher temperature (start) |
| `EMA_MOM_BASE` | 0.996 | EMA momentum (start) |

Output: `outputs/pretrained_dino_<RUN_ID>.pth` — set this as `VISION_WEIGHTS` in Step 2.

### Step 2 — Multimodal Fusion Training

Runs SupCon tabular pretraining (Stage 1), then end-to-end fusion training with the DINO-pretrained vision encoder (Stage 2).

```bash
# Update VISION_WEIGHTS in the script to point to your DINO checkpoint
python multimodal/multimodal_fusion_pipeline.py
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUPCON_EP` | 100 | SupCon tabular pretraining epochs |
| `SUPCON_LR` | 3e-4 | SupCon learning rate |
| `SUPCON_WEIGHT` (λ) | 0.2 | Weight on contrastive vs CE loss |
| `MCI_WEIGHT_BOOST` | 1.8 | Class re-weighting for MCI |
| `FUS_EP` | 30 | Fusion training epochs |
| `FUS_LR` | 5e-4 | Fusion learning rate |
| `VISION_LR_SCALE` | 0.05 | LR scale for vision unfrozen blocks |
| `N_UNFREEZE_BLOCKS` | 2 | Vision transformer blocks to fine-tune |

Outputs saved to `outputs/`:
- `supcon_tab_<RUN_ID>.pth` — trained tabular SupCon model
- `fusion_<RUN_ID>.pth` — full fusion model
- `config_<RUN_ID>.json` — run config + test metrics
- `tsne_fusion_<RUN_ID>.png` — t-SNE visualisation of fusion embeddings

### Step 3 — Tabular Baselines

Evaluate LightGBM, XGBoost, MLP, and DeepGBM on the same patient splits.

```bash
python tabular_baselines/tabular_baselines.py
```

Reports Accuracy, F1 (macro), AUROC, AUPRC, and MCC for all four models.

---

## Method Details

### DINO Pretraining

The DINO framework (Caron et al., ICCV 2021) is used for domain-adaptive self-supervised pretraining of the MRI encoder. A student-teacher pair shares the same MedSigLIP backbone. For each input slice, 2 global crops and 6 local crops are generated using spatial transformations (random resized crop, horizontal flip, affine) and intensity transformations (Gaussian blur, brightness/contrast jitter, Gaussian noise, random erasing). The student processes all views; the teacher processes global views only. Training minimises:

$$\mathcal{L}_{\text{DINO}} = \frac{1}{N} \sum_{i=1}^{N} D_{\text{KL}}\!\left( p_t(x_i^{\text{MRI}}) \,\|\, p_s(x_i^{\text{MRI}}) \right)$$

Teacher outputs are sharpened (low temperature), student outputs are softened (high temperature). The teacher is updated via EMA with a momentum coefficient that increases from 0.996 → 1.0 during training.

### Supervised Contrastive Tabular Pretraining

The tabular encoder is pretrained with a combined OrdinalSupCon + cross-entropy loss:

$$\mathcal{L}_{\text{SupCon}} = -\sum_{i=1}^{N} \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_p)/\tau)}{\sum_{a \neq i} \exp(\text{sim}(\mathbf{h}_i, \mathbf{h}_a)/\tau)}$$

An ordinal penalty is applied: CN–AD pairs are pushed apart more strongly than CN–MCI or MCI–AD pairs, reflecting the clinical ordering of disease severity.

### Fusion

MRI embeddings (512-d) and tabular embeddings (256-d) are concatenated into a 768-d joint representation, which is passed through a 3-layer MLP classifier. The vision encoder's last 2 transformer blocks receive gradients during fusion training (at 5% of the fusion learning rate), enabling end-to-end adaptation while preserving pretrained structural representations.

---

## License

This project is released under the [MIT License](LICENSE).
