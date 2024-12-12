
Store Sales Time-Series Forecasting

Overview

This repository contains an implementation of a hybrid machine learning model for time-series forecasting, developed for the "Store Sales Time-Series Forecasting" competition on Kaggle. The goal of this competition is to predict unit sales for thousands of items sold at different stores of Corporación Favorita, a large Ecuadorian-based grocery retailer. Accurate predictions can help minimize overstocking, reduce food waste, and improve customer satisfaction.

Competition Description

Forecasting sales is a vital task for brick-and-mortar grocery stores. Predicting too much inventory can lead to spoilage, while under-predicting leads to stockouts and lost revenue. Traditional forecasting methods often rely on subjective estimations, which are difficult to automate. With machine learning, it is possible to make forecasts that are more accurate and adaptable to complex scenarios.

In this competition, participants forecast unit sales based on historical data and additional features such as promotions. The evaluation metric is the Root Mean Squared Logarithmic Error (RMSLE), which is calculated as follows:

Where:

 is the number of instances.

 is the predicted value.

 is the actual value.

 is the natural logarithm.


Implementation Details

The hybrid model combines two approaches:

Linear Regression (Model 1): Captures long-term trends and seasonal patterns using deterministic processes.

XGBoost Regressor (Model 2): Models the residuals from the linear regression to capture complex patterns and short-term fluctuations.

Steps:

Data Loading:

Historical sales data is loaded and indexed by store, family, and date.

Data preprocessing ensures correct data types and structure.

Feature Engineering:

A deterministic process (DP) is used to create calendar-based features.

Fourier terms are added for seasonal components.

Additional features such as promotions are included.

Model Training:

Model 1 (Linear Regression) fits the main trend and seasonal components.

Residuals from Model 1 are calculated and used to train Model 2 (XGBoost).

Prediction:

Model 1 predicts the main trend.

Model 2 predicts the residuals, which are added to the trend prediction for final outputs.

Evaluation:

Results are plotted to visualize predictions against actual sales.

The RMSLE metric is used to evaluate model performance.

Submission:

Predictions for the test set are generated and saved to a CSV file for submission.

Key Libraries

pandas: Data manipulation and preprocessing

pathlib: Path handling

scikit-learn: Linear regression and label encoding

statsmodels: Deterministic processes and Fourier features

xgboost: Gradient boosting for residual modeling

matplotlib: Visualization of results
