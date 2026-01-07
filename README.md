# Grocery Demand Forecasting for Inventory Optimization

A machine learning project that predicts weekly product demand for a grocery retail chain in Ecuador. The model helps optimize inventory by forecasting which products will be ordered in the upcoming week, reducing both stockouts and overstock situations.

## The Problem

Retail inventory management is tricky. Order too much and you waste money on storage and spoilage. Order too little and you lose sales. This project tackles that problem by predicting product demand at the weekly level using historical sales data, holiday information, oil prices (Ecuador's economy is oil-dependent), and weather conditions.

The goal is simple: **predict whether a specific product will be ordered next week** (binary classification).

## Dataset

The data comes from the [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) Kaggle competition, with additional weather data scraped from Weather Underground.

| Source | Description | Records |
|--------|-------------|---------|
| train.csv | Daily unit sales per item/store | ~125M rows |
| holidays_events.csv | Ecuador holidays (national, regional, local) | 350 events |
| oil.csv | Daily WTI crude oil prices | 1,218 days |
| items.csv | Product metadata (family, perishable flag) | 4,100 items |
| stores.csv | Store locations and types | 54 stores |
| transactions.csv | Daily transaction counts per store | 83K records |
| Weather data | Scraped from wunderground.com | ~1,700 days |

For this analysis, I focused on **Store #13** (Latacunga, Cotopaxi region) to develop and validate the approach before scaling to other stores.

## Approach

### Feature Engineering

The raw daily data was aggregated to weekly level and enriched with:

**Temporal features:**
- Lag features (1-12 weeks) for sales and order history
- Rolling means (4, 8, 12 weeks) for smoothing trends
- Fourier transforms for capturing seasonality patterns
- Week number encoding

**External features:**
- Oil price lags and rolling averages (economic indicator)
- Holiday indicators (local, regional, national)
- Weather conditions (temperature, precipitation, cloudy/sunny days)

**Target variable:**
- `Ordered_next_week`: 1 if the product was sold in the following week, 0 otherwise

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | **96.6%** | 0.96 | 0.97 | 0.96 |
| Random Forest | 94.7% | 0.96 | 0.97 | 0.96 |
| Logistic Regression | 94.1% | 0.96 | 0.96 | 0.96 |
| K-Nearest Neighbors | 93.1% | 0.94 | 0.97 | 0.96 |

XGBoost with optimized hyperparameters achieved the best performance.

### Validation Strategy

Used `TimeSeriesSplit` with 3 folds to respect the temporal nature of the data. This prevents data leakage by ensuring the model never sees future data during training.

## Key Findings

The most important features for prediction (based on XGBoost feature importance):

1. **Order history lags** - Whether the product was ordered in recent weeks
2. **Sales volume lags** - How much was sold in previous weeks
3. **Rolling averages** - Smoothed demand trends
4. **Transaction counts** - Overall store activity level

Interestingly, external factors like oil prices and weather had minimal impact on the predictions. The product's own sales history was by far the strongest predictor.

## Project Structure

```
favorita/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── data_load.py              # Data loading and preprocessing
│   ├── feature_engineering.py    # Feature creation pipeline
│   ├── train_test_split.py       # Time-based train/test splitting
│   └── vreme_feature_engineering.py  # Weather feature processing
├── data/                         # Data files
│   ├── holidays_events.csv
│   ├── items.csv
│   ├── oil.csv
│   ├── stores.csv
│   ├── vreme_latacunga.csv       # Scraped weather data
│   ├── train.csv                 # (not in repo - download from Kaggle)
│   └── transactions.csv          # (not in repo - download from Kaggle)
├── notebooks/                    # Jupyter notebooks
│   ├── data_scraping.ipynb       # Weather data collection
│   ├── EDA.ipynb                 # Exploratory data analysis
│   ├── EDA2.ipynb                # Extended analysis with baseline
│   ├── feature_selection_XGB.ipynb
│   ├── feature_selection_RF.ipynb
│   ├── feature_selection_LR.ipynb
│   └── feature_selection_KNN.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository
2. Download the [Kaggle dataset](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) and place CSV files in `data/` folder
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the scraping notebook to collect weather data (optional, requires Chrome):
   ```bash
   jupyter notebook notebooks/data_scraping.ipynb
   ```
5. Open any notebook in `notebooks/` folder to train models:
   ```bash
   jupyter notebook notebooks/feature_selection_XGB.ipynb
   ```

**Note:** Notebooks need to add the src folder to path before importing:
```python
import sys
sys.path.append('..')
from src import data_load as dl
from src import feature_engineering as fe
```

## Results

The final XGBoost model achieves:
- **96.6% accuracy** on the test set
- **0.96 ROC AUC** score
- Strong performance on both classes (ordered vs not ordered)

The model successfully identifies products that need restocking with high precision, which translates to better inventory management and reduced waste.

## Limitations and Future Work

**Current limitations:**
- Model trained on single store (Store #13) - would need retraining for other locations
- Binary classification only - doesn't predict quantity
- Weather data coverage is incomplete for some periods

**Potential improvements:**
- Extend to multi-store prediction with store embeddings
- Predict actual quantities using regression
- Add promotion planning features
- Implement real-time prediction pipeline

## Tech Stack

- Python 3.10+
- pandas, numpy for data manipulation
- scikit-learn for preprocessing and model evaluation
- XGBoost for gradient boosting
- Optuna for hyperparameter optimization
- Selenium for web scraping
- matplotlib, seaborn for visualization

---

*This project was developed as part of applied machine learning coursework, focusing on practical demand forecasting techniques for retail optimization.*
