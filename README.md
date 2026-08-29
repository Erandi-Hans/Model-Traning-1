# 🏡 House Price Prediction & K-Means Clustering (Regression & Unsupervised Learning)

This repository details **Day 2 (Part 1)** of my Machine Learning journey, focusing on **Regression Analysis** and **Unsupervised Clustering**. Using the Housing dataset, various regression techniques were implemented to predict property prices, alongside K-Means clustering to uncover distinct real estate market segments.

---

## 📌 Project Overview

The primary objectives of this project were:
1. **Regression Task:** Predict house prices based on spatial and physical property attributes (`area`, `bedrooms`, `bathrooms`, `stories`, `parking`, and amenity flags).
2. **Feature Engineering & Transformation:** Implemented missing data imputation, categorical one-hot encoding, feature scaling, and target log-transformation to optimize model performance.
3. **Unsupervised Market Segmentation:** Applied K-Means clustering on feature pairs (`area` and `price`) to automatically segment the market into distinct property tiers without ground-truth labels.

---

## 🚀 Data Processing & Feature Engineering Workflow

* **Missing Value Imputation:** 
  * `area` & `bathrooms` filled using **Median** strategy due to right-skewed distributions.
  * `parking` filled using **Mode** (most frequent discrete count).
* **Feature Scaling & Encoding:**
  * Categorical/binary features (`mainroad`, `airconditioning`, `prefarea`, etc.) converted into binary flags.
  * `furnishingstatus` dummy-encoded via One-Hot Encoding.
  * Numerical features standardized using `StandardScaler` (fit strictly on training data).
* **Engineered Features:**
  * Derived composite features: `amenity_score`, `room_total`, and `area_per_room`.
  * Applied logarithmic transformation ($\log(1+x)$) to house prices to handle heavy right-skewness.

---

## 📊 Regression Model Comparison

Models were trained on an **80/20 train-test split** and evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score:

| Model Architecture | MAE (LKR/₹) | RMSE (LKR/₹) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | **1,035,220** | **1,441,851** | **0.431** |
| **Linear Regression ($\log(\text{price})$)** | 1,018,110 | 1,449,187 | 0.425 |
| **Random Forest Regressor ($\log(\text{price})$)** | 1,128,572 | 1,562,681 | 0.331 |

---

## 🔍 Unsupervised Market Segmentation (K-Means)

Using the **Elbow Method** to identify the optimal number of clusters ($K$), $K=3$ was chosen to partition the housing market into natural pricing tiers:

| Cluster Tier | Mean Area (sq. ft.) | Mean Price (LKR/₹) | Market Tier Interpretation |
| :---: | :---: | :---: | :--- |
| **Cluster 0** | ~4,087 | ~3,655,790 | **Budget / Starter Homes** |
| **Cluster 2** | ~6,058 | ~5,450,217 | **Mid-Tier Residential** |
| **Cluster 1** | ~9,165 | ~8,059,855 | **High-End Luxury Estates** |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.13
* **Libraries:**
  * Data Wrangling: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `KMeans`, `StandardScaler`)

---

## 🔑 Key Takeaways

* **Linear Baseline Dominance:** The standard Linear Regression model outperformed ensemble tree-based approaches (Random Forest) on this feature set, achieving the highest explanatory power ($R^2 = 0.431$).
* **Effective Feature Signal:** Derived ratios like `area_per_room` ($r = 0.381$) and `area` ($r = 0.618$) showed strong positive linear correlations with house values.
* **Cluster Utility:** Unsupervised K-Means successfully separated property groups by size and price without label guidance, identifying distinct market segments.

  ---

## 📅 Roadmap / Next Steps

- [x] **Day 1:** Data Preprocessing & Exploratory Data Analysis (EDA)
- [ ] **Day 2:** Model Training & Comparative Evaluation (Classification, Regression & Neural Networks)
- [ ] **Day 3:** Hyperparameter Tuning, Introduction to NLP, and Model Deployment
