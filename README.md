# HiDA-BAA
HiDA-BAA : Hierarchical Domain Adaptation for Bone Age Assessment.

---

## 🧠 Overview

**HiDA-BoneAge** is a training framework for **bone age assessment (BAA)** from hand and wrist X-ray images using a **hierarchical domain adaptation strategy**.

The framework addresses the **domain shift problem** commonly observed between:

* Public datasets (e.g., RSNA)
* Real-world clinical data (e.g., hospital-specific distributions)

Instead of training a model in a single stage, HiDA-BoneAge introduces a **two-stage hierarchical training paradigm**:

1. **Stage 1 — General Representation Learning**
   Train the model on a large-scale **public dataset** to learn robust and generalizable features.

2. **Stage 2 — Domain Adaptation**
   Fine-tune the pretrained model on a **target domain dataset** to adapt to distribution shifts and improve prediction accuracy.

---

## 🎯 Key Contributions

* 📌 **Hierarchical Training Strategy** for regression-based bone age estimation
* 🔄 **Explicit Domain Adaptation** to mitigate dataset bias
* 🧩 Modular framework for plugging in different backbone models
* 📉 Improved robustness across heterogeneous medical datasets

---

## ⚙️ Pipeline

```
Raw X-ray Images
        ↓
[ Preprocessing Pipeline ]  ← (Required, external repo)
        ↓
Cleaned & Standardized Images
        ↓
Stage 1: Train on Public Dataset
        ↓
Stage 2: Fine-tune on Target Domain
        ↓
Bone Age Prediction (Regression)
```

---

## ⚠️ Important: Preprocessing Requirement

This repository **does NOT include preprocessing**.

Before training, you **must preprocess all images** using the dedicated preprocessing pipeline:

👉 **Required Repo:** *(replace with your actual repo link)*
**HandXRay-Preprocessing-Pipeline**

This ensures:

* Consistent image format
* Proper hand localization
* Standardized input for model training

Without preprocessing, model performance will significantly degrade.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/HiDA-BoneAge.git
cd HiDA-BoneAge
```

---

### 2. Prepare Dataset

* Apply preprocessing using the external repo
* Organize data into:

```
data/
├── public/        # e.g., RSNA
├── target/        # your domain-specific dataset
```

---

### 3. Train Model

#### Stage 1 — Train on Public Dataset

```bash
python train_stage1.py --config configs/stage1.yaml
```

#### Stage 2 — Domain Adaptation (Fine-tuning)

```bash
python train_stage2.py --config configs/stage2.yaml
```

---

## 🧩 Framework Design

The framework is designed to be modular:

* `models/` → backbone architectures (e.g., EfficientNet)
* `trainers/` → stage-specific training logic
* `configs/` → experiment configurations
* `datasets/` → dataset loaders (public + target)

---

## 📊 Task Definition

* **Input:** Preprocessed hand X-ray image
* **Output:** Continuous bone age prediction (regression)
* **Loss:** Typically Mean Squared Error (MSE)

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
