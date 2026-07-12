# Data layout

This repository does not include patient data or public dataset images.

This repository is for target-domain adaptation. The public/source-domain model is expected to be provided separately as the Stage 1 checkpoint.

All images must first be preprocessed with [Khao0/HandXRay-Preprocessing](https://github.com/Khao0/HandXRay-Preprocessing).

Place preprocessed **600x600** target-domain hand X-ray images and CSV annotations in this structure:

```text
data/
├── train/
├── test/
├── train.csv
└── test.csv
```

Expected CSV columns:

```csv
image_name,bone_age,sex
0001.png,132,M
0002.png,96,F
```

- `image_name` should match a file name inside `data/train/` or `data/test/`. If the extension is omitted, the default `.png` extension from `configs/target.yaml` is used.
- `bone_age` is the regression target in months.
- `sex` is expected by the released HiDA-SourceModel. Accepted values include `M`, `F`, `male`, `female`, `1`, and `0`.
- Images must be exactly `600x600` because HiDA-SourceModel uses EfficientNetB7 at that input resolution.
- `train.csv` is split internally into stratified K folds using bone-age year and sex.
- `test.csv` is used for fold-level final evaluation.
