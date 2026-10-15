# California Traffic Accidents Analysis & Severity Prediction

## Project Description
An analytical and predictive modeling (Machine Learning) project aimed at analyzing traffic accidents in California and predicting their severity. The analysis is based on the `US_Accidents_March23` dataset (~1.74M California records, 2016–2023).

**Target variable:** `Severity` (1–4), the accident's **impact on traffic (delay)**, not injury severity. Severe class imbalance: Severity 2 accounts for ~83% of records.

**Task used for modeling:** 3 ordered classes: `Sev1-2` (Severity 1 and 2 merged), `Sev3`, `Sev4`. Severity 1 appears almost only in two short labeling bursts (2020 H1, 2022), so it was merged with its ordinal neighbour. A 4-class result is reported for reference.

## Repository Structure
```
├── src/                          — all logic (data, features, split, models, evaluation, SHAP, plots)
├── notebooks/
│   ├── 01_data_and_split.ipynb   — leakage exclusions, source/time drift, train/test split
│   ├── 02_model_comparison.ipynb — baselines vs RF vs NN vs LightGBM (grouped CV)
│   ├── 03_final_model.ipynb      — final LightGBM, single evaluation on both test sets
│   ├── 04_shap.ipynb             — SHAP aggregated to original features
│   └── archive/                  — original notebooks (before the methodology revision)
├── tests/                        — leakage / split tests (python -m pytest tests)
├── reports/
│   └── results/                  — all CV and test results as CSV
├── data/                         — raw and processed data (gitignored)
├── models/                       — saved models (gitignored)
└── requirements.txt
```

## Methodology
- **Leakage removed:** `End_Time`/duration, `End_Lat`/`End_Lng`, `Distance(mi)` and derivatives, `Description` (all describe the accident's consequences or are only known afterwards).
- **Labels depend on the data provider.** `Source1` labels ~2% of accidents as Severity 3, `Source2` ~40%, and both changed their labeling over time (e.g. `Source1` has had virtually no Severity 3 since 2021). `Source` is therefore an explicit feature and all results are also reported per source.
- **Split:** `train` = 2016–2021 (85% of accident events), `test_in` = the other 15% of 2016–2021 events, `test_out` = 2022–2023 Q1 (out-of-time). Rows describing the same event (identical time and location, 12% of rows) always stay in the same part.
- **No test leakage:** imputation, scaling and encoding live in an `sklearn` `Pipeline` fitted on training folds only; model selection and tuning used grouped 5-fold CV on `train`; each test set was evaluated **once**.
- **Fair comparison:** all models share folds, preprocessing, class-weighting function and a 10-configuration random search; the strength of class weighting is tuned as a hyperparameter for every model.
- **Main metric:** macro-F1. Also reported: balanced accuracy, quadratic weighted kappa (QWK), per-class recall, confusion matrices, CV standard deviation or 95% bootstrap intervals.

## Key Results

**Model comparison (5-fold grouped CV on a 10% sample of `train`, macro-F1 mean ± SD):**

| Model | macro-F1 | QWK | Recall Sev4 |
|---|---|---|---|
| Majority class | 0.295 ± 0.000 | 0.000 | 0.00 |
| Logistic regression (weighted) | 0.554 ± 0.003 | 0.598 | 0.00 |
| Neural network (PyTorch MLP) | 0.582 ± 0.006 | 0.589 | 0.07 |
| LightGBM, ordinal | 0.636 ± 0.008 | 0.700 | 0.12 |
| LightGBM | 0.640 ± 0.006 | 0.682 | 0.15 |
| Random Forest | 0.642 ± 0.003 | 0.678 | 0.16 |

RF and LightGBM are tied overall; LightGBM was chosen because it is better **within** each data source (e.g. `Source1`: 0.43 vs 0.39). On the full `train` set the selected LightGBM configuration reached macro-F1 **0.726 ± 0.006** in 5-fold CV.

**Final LightGBM, single evaluation on held-out data (95% bootstrap CI):**

| Metric | `test_in` (2016–2021) | `test_out` (2022–2023 Q1) |
|---|---|---|
| macro-F1 | **0.735** [0.727–0.742] | **0.591** [0.589–0.592] |
| Balanced accuracy | 0.740 [0.731–0.748] | 0.634 [0.633–0.636] |
| QWK | 0.760 [0.756–0.765] | 0.516 [0.508–0.524] |
| Recall Sev1-2 / Sev3 / Sev4 | 0.95 / 0.88 / 0.39 | 0.99 / 0.91 / **0.00** |

**What this means:**
- Within the training period the model is useful, but performance **declines year by year** (macro-F1 0.76 in 2016 → 0.64 in 2021) as providers change how they label severity.
- On future data (2022–2023) the model **does not recognise Severity 4 at all** (1 of 3,362 cases) and, for `Source1` (92% of that period), predicts only `Sev1-2`. The headline 0.59 is held up mostly by `Source2` records.
- A large part of the signal is the data provider itself: removing `Source` lowers CV macro-F1 from ~0.64 to ~0.52–0.53, and `Source` is by far the top SHAP feature, ahead of `road_type`, wind speed and location.
- Severity in this dataset is partly a **labeling artifact** of each provider and period, so these numbers should not be read as how well accident impact can be predicted in general.

**Comparison with the previous version of this project:** the earlier README reported RF balanced accuracy ~0.46 and kappa ~0.32 (the notebook itself showed 0.554 and QWK 0.225). That evaluation used a random split, used the test set for model choices, and had duplicate events across train and test, so the numbers are not comparable. For reference, the same final configuration trained on the original 4 classes scores balanced accuracy 0.72 / QWK 0.76 on `test_in` and 0.48 / 0.43 on `test_out` (Severity 1 and 4 recall = 0 there).

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/MiaByte-ctrl/california-accidents-ml
cd california-accidents-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows + GPU (CUDA):** `torch` must be installed from the [PyTorch website](https://pytorch.org/get-started/locally/) with the correct CUDA version. Plain `pip install torch` installs the CPU-only build.
> Example for CUDA 11.8:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. Download the dataset
Download `US_Accidents_March23.csv` from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and place it in `data/raw/`.

### 4. Run notebooks in order
`01` creates the cleaned data cache and the saved split; `02` and `03` load saved results from `reports/results/` by default (set `RECOMPUTE = True` to recompute); `04` needs the model saved by `03`.

```bash
py -m jupyter notebook
```

### 5. Run tests
```bash
python -m pytest tests
```
