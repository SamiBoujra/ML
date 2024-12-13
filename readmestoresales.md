# BoostedHybrid Model for Store Sales Time Series Forecasting

## Overview
This repository contains a Python implementation of a hybrid machine learning model for time series forecasting using data from the Kaggle competition "Store Sales Time Series Forecasting." The goal of the competition is to predict sales for thousands of items across multiple stores, leveraging historical data, promotions, and other features.

The model combines a **Linear Regression** model for baseline predictions and an **XGBoost Regressor** for modeling residuals, creating a robust hybrid approach to time series forecasting.

---

## Files

- `store_sales.py`: The main script containing the implementation of the BoostedHybrid model and data preprocessing steps.
- `submission.csv`: The output file containing predictions for the competition.

---

## Key Features
- **BoostedHybrid Model**: Combines two machine learning models:
  - Linear Regression for baseline trend and seasonality.
  - XGBoost Regressor to capture complex residual patterns.
- **Deterministic Process Features**: Incorporates Fourier terms, seasonal indicators, and other deterministic features.
- **Custom Data Preprocessing**: Handles categorical encoding, feature engineering, and multilevel time series data.

---

## Requirements

### Python Libraries
- `pandas`
- `matplotlib`
- `scikit-learn`
- `statsmodels`
- `xgboost`

Install the required libraries using:
```bash
pip install pandas matplotlib scikit-learn statsmodels xgboost
```

---

## Data
The competition dataset includes:
- Historical sales data for multiple stores and product families.
- Features such as promotions and date information.

The script preprocesses and structures the data into a multilevel time series format to train and test the hybrid model.

---

## Competition Details

### Goal
To forecast sales for thousands of items sold at different stores, improving accuracy compared to traditional methods.

### Evaluation Metric
The competition uses **Root Mean Squared Logarithmic Error (RMSLE)** as the evaluation metric:

\[
RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2 }
\]

Where:
- \( \hat{y}_i \): Predicted value
- \( y_i \): Actual value
- \( n \): Number of instances

### Submission Format
Predictions are saved in a CSV file with the following format:

```
id,sales
3000888,0.0
3000889,0.0
...
```

---

## How to Run the Code

### 1. Prepare the Environment
Ensure you have the required Python libraries installed and the competition dataset downloaded to the `../input/store-sales-time-series-forecasting` directory.

### 2. Train the Model
Run the script `store_sales.py` to train the hybrid model on the 2017 sales data and generate predictions for the test set.

```bash
python store_sales.py
```

### 3. Submit Predictions
Upload the generated `submission.csv` file to the Kaggle competition for evaluation.

---

## Results Visualization
The script includes a visualization step to compare actual and fitted sales for a specific store and product family. Example:

![Sales Plot](example_plot.png)

---

## Potential Impact
Accurate forecasting can:
- Reduce food waste by optimizing inventory management.
- Improve customer satisfaction by ensuring stock availability.
- Assist retailers in making data-driven decisions.

---

## References
- [Kaggle Competition: Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- Kaggle's Time Series Forecasting Course

---

## Author
This project is developed by a Kaggle participant aiming to improve time series forecasting accuracy through innovative hybrid modeling approaches.
