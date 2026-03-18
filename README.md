# California Traffic Accidents Analysis & Severity Prediction

## Project Description
An analytical and predictive modeling (Machine Learning) project aimed at analyzing traffic accidents in California and predicting their severity. The analysis is based on the `US_Accidents_March23` dataset.

## Repository Structure
* `data/` - Directory for raw and processed data (ignored by Git).
* `notebooks/`
  * `01_data_cleaning.ipynb` - EDA, outlier handling (clipping), and feature engineering.
  * `02_stats_and_feature_importance.ipynb` - Statistical analysis and feature selection.
  * `03_modeling.ipynb` - ML model training (in progress).
  * `04_shap_analysis.ipynb` - Model interpretability and SHAP analysis (in progress).
* `models/` - Saved predictive models (ignored by Git).

## Installation & Setup
1. Clone the repository: `git clone [your_repo_link]`
2. Install required packages: `pip install -r requirements.txt`
3. Download the [Kaggle US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) and place the `US_Accidents_March23.csv` file in the `data/raw/` directory.
4. Run the notebooks in numerical order.