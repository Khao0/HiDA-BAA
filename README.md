# HiDA-BAA
HiDA-BAA : Hierarchical Domain Adaptation for Bone Age Assessment.

---

## 🧠 Overview

**HiDA-BoneAge** is a target-domain adaptation framework for **bone age assessment (BAA)** from hand and wrist X-ray images.

The framework addresses the **domain shift problem** commonly observed between:

* Public datasets (e.g., RSNA, DHA)
* Real-world clinical data (e.g., hospital-specific distributions)

HiDA-BoneAge uses a **two-stage hierarchical training paradigm**:

| Stage | Name | Purpose |
| --- | --- | --- |
| Stage 1 | **HiDA-SourceModel** | A provided base model already trained on public/source datasets such as DHA and RSNA. |
| Stage 2 | **HiDA-TargetModel** or **HiDA-AdaptedModel** | Fine-tune the source model on the researcher's target-domain dataset. |

This repository focuses on **Stage 2 target-domain adaptation**. Researchers only need their own preprocessed target dataset and the provided Stage 1 checkpoint.

The released **HiDA-SourceModel** is based on **EfficientNetB7** with ImageNet pretrained weights, following the Keras EfficientNet fine-tuning setup: [Image classification via fine-tuning with EfficientNet](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/). Because the source model was trained at **600x600** resolution, this repository expects **600x600 input images only**.

---

## 🎯 Key Contributions

* 📌 **Hierarchical Training Strategy** for regression-based bone age estimation
* 🔄 **Explicit Domain Adaptation** to mitigate dataset bias
* 🧩 Modular framework for plugging in different backbone models
* 📉 Improved robustness across heterogeneous medical datasets

---

## ⚙️ Pipeline

```
Target-domain Raw X-ray Images
        ↓
[ HandXRay-Preprocessing ]  ← Required external repo
        ↓
Preprocessed 600x600 Images
        ↓
Stage 2: Fine-tune on Target Domain Dataset
        ↑
HiDA-SourceModel checkpoint
        ↓
HiDA-TargetModel / HiDA-AdaptedModel
        ↓
Bone Age Prediction
```

---

## ⚠️ Important: Preprocessing Requirement

This repository **does NOT include preprocessing**.

Before training, you **must preprocess all images** using the dedicated preprocessing pipeline:

👉 **Required Repo:** [Khao0/HandXRay-Preprocessing](https://github.com/Khao0/HandXRay-Preprocessing)

This ensures:

* Consistent image format
* Proper hand localization
* Standardized **600x600** input for model training

Without preprocessing, model performance will significantly degrade. Images with a size other than **600x600** are not supported by the target adaptation pipeline.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/HiDA-BoneAge.git
cd HiDA-BoneAge
python3 -m pip install -r requirements.txt
```

---

### 2. Prepare Target Dataset

* Apply preprocessing using the external repo
* Organize data into:

```
data/
├── train/                  # preprocessed 600x600 target-domain training images
├── test/                   # preprocessed 600x600 target-domain test/validation images
├── train.csv
└── test.csv
```

Expected annotation format:

```csv
image_name,bone_age,sex
0001.png,132,M
0002.png,96,F
```

See `data/README.md` and the `*.csv.example` files for templates.

---

### 3. HiDA-SourceModel Resolution

The training system resolves **HiDA-SourceModel** automatically from constants defined in `hida_baa/models/load_source_model.py`:

1. It first checks the default local path:

```
checkpoints/SourceModel/HiDA-SourceModel.keras
```

2. If the file is not found, it downloads `HiDA-SourceModel.keras` from Hugging Face repo `Kwankhao/HiDA-SourceModel`.

Researchers do not need to pass a source-model path manually.

---

### 4. Train HiDA-TargetModel

```bash
python3 scripts/train.py --config configs/target.yaml
```

The training pipeline follows the original HiDA adaptation flow:

* Create stratified K-fold splits from `train.csv` using bone-age year and sex
* Load a fresh **HiDA-SourceModel** for each fold
* Warm up the regression head while `efficientnetb7` is frozen
* Reload the best warm-up model
* Fine-tune upper EfficientNetB7 layers into **HiDA-TargetModel**
* Evaluate each fold on `test.csv`
* Save fold models, histories, loss curves, prediction scatter plots, and CV metrics

---

### 5. Inference

Run prediction on one preprocessed image with one explicit `.keras` model:

```bash
python3 scripts/inference.py path/to/image.png --model checkpoints/AdaptedModel/run/best_model_fold0.keras --sex M
```

Inference accepts **one image file per call**. The script prints the predicted bone age only:

```text
120.123456
```

---

### 6. Test / Re-evaluate

Evaluate a test folder with `test.csv` metadata:

```bash
python3 scripts/test.py data/test --csv data/test.csv --model checkpoints/AdaptedModel/run/best_model_fold0.keras
```

The model path is required and must point to a `.keras` file.

`test.csv` must contain:

```csv
image_name,bone_age,sex
example.png,120,M
```

The script prints JSON metrics only:

```json
{"mae": 4.1, "n": 100, "r2": 0.91, "rmse": 5.3}
```

---

## 🧩 Framework Design

The framework is designed to be modular:

* `hida_baa/models/` → source model loading and model utilities
* `hida_baa/trainers/` → target adaptation training logic
* `configs/` → experiment configurations
* `hida_baa/datasets/` → target dataset loaders

Current scaffold:

```
configs/
└── target.yaml
scripts/
├── train.py
├── test.py
└── inference.py
hida_baa/
├── predict.py
├── datasets/
│   └── bone_age_dataset.py
├── models/
│   ├── bone_age_model.py
│   └── load_source_model.py
└── trainers/
    ├── train.py
    └── utils.py
```

---

## 📊 Task Definition

* **Input:** Preprocessed **600x600** hand X-ray image
* **Output:** Continuous bone age prediction (regression)
* **Default adaptation loss:** Mean Absolute Error (MAE)

---

## 📦 Outputs

By default, training writes adapted models and analysis artifacts to:

```
checkpoints/AdaptedModel/<run_name>/
outputs/target_model/<run_name>/
├── loss_graphs/
├── history/
├── predictions/
├── weights/
├── structure.png
├── model_summary.txt
├── config.resolved.json
└── cv_results.csv
```

---

## 🔬 Research Context

Bone age assessment models often suffer from **performance degradation when applied to new domains** due to:

* Imaging condition differences
* Demographic variation
* Annotation inconsistencies

HiDA-BoneAge mitigates these issues through **structured training stages**, enabling better generalization and adaptation.

---

## 📌 Future Work

* Multi-domain adaptation
* Semi-supervised / unsupervised domain adaptation
* Integration with uncertainty estimation
* Extension to other medical imaging tasks

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{hida_boneage,
  title={Hierarchical Domain Adaptation for Bone Age Assessment},
  author={Your Name},
  year={2026}
}
```

---

## 🤝 Acknowledgements

* Public datasets (e.g., RSNA Bone Age)
* Medical imaging and deep learning communities

---

## 📬 Contact

For questions or collaboration, please open an issue or contact the author.

---

## 💡 Tip (important for your paper)

In your paper, refer to this repo as:

> *“We release our training framework, **HiDA-BoneAge**, for reproducibility.”*

---

If you want next, I can:

* Write a **matching paper title + abstract (very strong for submission)**
* Design **figure diagram (pipeline for your paper)**
* Or refine your **method section wording (Hierarchical DA formal)**
