# California Traffic Accidents Analysis & Severity Prediction

## Project Description
An analytical and predictive modeling (Machine Learning) project aimed at analyzing traffic accidents in California and predicting their severity. The analysis is based on the `US_Accidents_March23` dataset (~1.7M California records, 2016–2023).

**Target variable:** `Severity` (1–4) — severe class imbalance: Severity 2 accounts for ~83% of records.

## Repository Structure
```
├── notebooks/
│   ├── 01_data_cleaning.ipynb   — EDA, outlier clipping, feature engineering
│   ├── 02_stats.ipynb           — Statistical tests, VIF analysis, geographic clustering
│   ├── 03_modeling.ipynb        — Random Forest vs Neural Network comparison
│   └── 04_shap_analysis.ipynb  — Model interpretability via SHAP values
├── data/                        — Raw and processed data (gitignored)
├── models/                      — Saved model artifacts (gitignored)
├── requirements.txt
```

## Key Results
- Random Forest (`max_depth=20`, cost-matrix class weights) outperforms a baseline MLP on tabular data out-of-the-box
- Balanced Accuracy: RF ~0.46 vs NN ~0.31 (random baseline: 0.25)
- Cohen's Kappa: RF ~0.32 (fair agreement) vs NN ~0.05 (near-random)
- SHAP analysis identifies top predictors per severity class

## Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/MiaByte-ctrl/california-accidents-ml]
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
Notebooks must be run sequentially — each depends on outputs of the previous.

```bash
py -m jupyter notebook
```
