# 🏠 House Price Prediction

An end-to-end Machine Learning project that predicts house prices using the Kaggle House Prices dataset.

## 🚀 Project Overview

The goal of this project is to build a complete machine learning pipeline for predicting house prices.

The project covers:

- Data understanding
- Exploratory Data Analysis (EDA)
- Missing value handling
- Numerical and categorical preprocessing
- Feature encoding
- Feature scaling
- Train-test split
- Multiple regression algorithms
- Cross-validation
- Model comparison
- Error analysis
- XGBoost model selection
- Model serialization
- Streamlit deployment

## 📊 Dataset

Dataset: Kaggle House Prices

- Rows: 1460
- Original Features: 80
- Target: `SalePrice`
- Data Types: Numerical + Categorical

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

### Numerical Features

- Median imputation for missing values
- StandardScaler

### Categorical Features

- Most-frequent imputation
- One-Hot Encoding
- `handle_unknown="ignore"`

The preprocessing is integrated directly into the ML pipeline to avoid data leakage.

## 🤖 Models Tested

The following regression algorithms were evaluated:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost

## 📈 Model Comparison

| Model | R² Score | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.8867 | 29,476 | 18,285 |
| Ridge Regression | 0.8775 | 30,655 | 19,021 |
| Lasso Regression | 0.8960 | 28,238 | 17,917 |
| Decision Tree | 0.8011 | 39,061 | 25,060 |
| Random Forest | 0.8945 | 28,450 | 17,445 |
| Gradient Boosting | 0.8986 | 27,887 | 17,082 |
| XGBoost | **0.9081** | **26,545** | **16,934** |

## 🏆 Final Model

The final selected model is:

**XGBoost Regressor**

Configuration:

- `n_estimators = 200`
- `learning_rate = 0.05`
- `max_depth = 3`
- `objective = reg:squarederror`

Test performance:

- **R²: 0.9081**
- **RMSE: $26,545**
- **MAE: $16,934**

The final model was selected considering cross-validation performance rather than simply choosing the model with the best test-set score.

## 📁 Project Structure

```text
01_House_Price_Prediction/
│
├── data/
│   └── train.csv
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│
├── models/
│   ├── house_price_xgb_pipeline.pkl
│   └── house_input_template.pkl
│
├── app/
│   └── app.py
│
└── README.md