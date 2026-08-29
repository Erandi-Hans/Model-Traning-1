# House Price Prediction Model

An end-to-end Machine Learning pipeline for predicting house prices using Exploratory Data Analysis (EDA), feature engineering, data preprocessing, and multiple regression algorithms.

---

## 📌 Project Overview

This project builds a predictive regression model to estimate housing prices based on structural features, location preferences, and available amenities. The dataset undergoes comprehensive preprocessing, missing-value imputation, outlier detection, and feature engineering before evaluating baseline and log-transformed regression models.

---

## 🛠️ Environment & Prerequisites

### Technical Stack

* **Language:** Python 3.13
* **Environment:** Google Colab / Jupyter Notebook

### Installed Dependencies

* **Data Processing & Analysis:** `pandas` (v2.2.3), `numpy` (v2.1.3), `scipy` (v1.16.3)
* **Visualization:** `matplotlib` (v3.10.0), `seaborn` (v0.13.2)
* **Machine Learning:** `scikit-learn` (v1.6.1), `joblib` (v1.5.3), `threadpoolctl` (v3.6.0)
* **Utilities:** `python-dateutil`, `pytz`, `tzdata`, `contourpy`, `cycler`, `fonttools`, `kiwisolver`, `packaging`, `pillow`, `pyparsing`, `six`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

---

## 📊 Dataset Summary

* **Dataset Size:** 546 rows × 13 initial columns (expanded to 22 features post-engineering)
* **Target Variable:** `price` (Right-skewed, Raw Skewness: **1.49**, Log-Transformed Skewness: **0.34**)

### Column Breakdown & Data Types

| Feature Name | Type | Missing Values | Imputation Strategy / Notes |
| --- | --- | --- | --- |
| `price` | Numerical (`int64`) | 0 (0.0%) | Target variable |
| `area` | Numerical (`float64`) | 21 (3.8%) | Imputed with **Median** (5,552 mean vs 4,971 median) |
| `bedrooms` | Numerical (`int64`) | 0 (0.0%) | Structural feature |
| `bathrooms` | Numerical (`float64`) | 30 (5.5%) | Imputed with **Median** |
| `stories` | Numerical (`int64`) | 0 (0.0%) | Structural feature |
| `parking` | Numerical (`float64`) | 33 (6.0%) | Imputed with **Mode** (Most common value: 1.0) |
| `mainroad` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `guestroom` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `basement` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `hotwaterheating` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `airconditioning` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `prefarea` | Categorical (`object`) | 0 (0.0%) | Binary (`yes`/`no`) |
| `furnishingstatus` | Categorical (`object`) | 0 (0.0%) | Multi-class (`furnished`, `semi-furnished`, `unfurnished`) |

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Missing Value Imputation:**
* `area` $\rightarrow$ Median Imputation (Right-skewed)
* `bathrooms` $\rightarrow$ Median Imputation
* `parking` $\rightarrow$ Mode Imputation (Categorical count value)


2. **Encoding Categorical Features:**
* **Binary Conversion:** Maps `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, and `prefarea` from (`yes`/`no`) to (`1`/`0`).
* **One-Hot Encoding:** Applied to `furnishingstatus` with `drop_first=True` resulting in `furnishingstatus_semi-furnished` and `furnishingstatus_unfurnished`.


3. **Feature Engineering:**
* `amenity_score`: Sum of 5 comfort features (`guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`). Shows a correlation of **0.151** with price.
* `room_total`: `bedrooms` + `bathrooms` (Correlation $r = 0.195$).
* `area_per_room`: `area` / `room_total` (Correlation $r = 0.381$).


4. **Feature Scaling & Data Splitting:**
* **Split Ratio:** 80% Train (436 samples) / 20% Test (110 samples)
* **Scaling Method:** `StandardScaler` fitted **strictly on the training set** to avoid data leakage.



---

## 📈 Exploratory Data Analysis Insights

* **Outliers:** `price` contains 28 outliers (5.1%) and `area` contains 17 outliers (3.1%). Retained in dataset as legitimate high-value properties.
* **Feature Correlations with Price:**
* `area`: **0.618** (Strongest positive correlation)
* `area_per_room`: **0.381**
* `bathrooms`: **0.150**
* `stories`: **0.132**
* `bedrooms`: **0.131**
* `parking`: **0.115**


* **Price Premium by Feature (Median Price Comparison):**
* Air Conditioning: **+15%**
* Preferred Area (`prefarea`): **+10%**
* Hot Water Heating: **+10%**
* Main Road Access: **+1%**


* **Furnishing Status Tiers:**
* **Furnished:** ₹5.17M (Median)
* **Semi-Furnished:** ₹4.57M (Median)
* **Unfurnished:** ₹3.72M (Median)



---

## 🚀 Model Performance & Evaluation

Models were evaluated using **Mean Absolute Error (MAE)**, **Root Mean Squared Error (RMSE)**, and **$R^2$ Score** on the 20% test dataset (110 samples).

| Model | Target Scale | MAE (₹) | RMSE (₹) | $R^2$ Score |
| --- | --- | --- | --- | --- |
| **Linear Regression (Baseline)** | Raw Price | ₹1,035,220 | ₹1,441,851 | **0.431** |
| **Linear Regression (Log Price)** | Log Transformed (`log1p`) | **₹1,018,110** | ₹1,449,187 | 0.425 |
| **Random Forest Regressor** | Log Transformed (`log1p`) | ₹1,128,572 | ₹1,562,681 | 0.331 |

### Key Takeaways

* **Linear Regression with Log Target** achieved the lowest Mean Absolute Error (**₹1,018,110**).
* **Baseline Linear Regression** yielded the highest variance explained ($R^2 = 0.431$).
* **Random Forest Regressor** underperformed relative to Linear Regression on this dataset size, likely due to overfitting on the small feature set (436 training samples).
