I. Dataset Description
![image](https://github.com/user-attachments/assets/27db9178-737b-4139-92e8-b0dee2376969)

Dataset: Historical stock prices of Netflix, Inc. (NFLX)

Period: January 2002 – December 2021

Frequency: Daily

Features Used:
Date: Timestamp of the observation

Open: Stock price at market opening

High: Highest price during the day

Low: Lowest price during the day

Close: Closing price of the stock

Adj Close: Adjusted closing price (target variable)

Volume: Number of shares traded

The target variable for forecasting was Adj Close, which reflects adjusted closing prices accounting for stock splits and dividends.

II. Model Evaluation Metrics
To assess and compare the forecasting performance of ARIMA and VAR models, we employed both information criteria and error-based metrics:

1. Akaike Information Criterion (AIC)
A measure of model fit that penalizes complexity to prevent overfitting.

Formula:
![image](https://github.com/user-attachments/assets/d11c877a-2f0e-455b-b01d-bc19388b2ae8)

Interpretation: Lower AIC values indicate a better balance between goodness-of-fit and model simplicity.

2. Bayesian Information Criterion (BIC)
Similar to AIC but imposes a stronger penalty for model complexity.

Formula:
![image](https://github.com/user-attachments/assets/6e8b233d-4145-49d9-ad16-f135c0e652f5)

Interpretation: Lower BIC favors models that generalize better, especially when sample size is large.

3. Root Mean Squared Error (RMSE)
Measures the average magnitude of the errors between predicted and actual values, in original units.

Formula:
![image](https://github.com/user-attachments/assets/2a070ae7-8da6-4361-90aa-d32aa50ed22d)

Interpretation: Lower RMSE values indicate better predictive accuracy.

4. Mean Absolute Error (MAE)
Computes the average of absolute differences between predictions and actual values.

Formula:
![image](https://github.com/user-attachments/assets/7c02e0db-18dd-4f2d-a868-a3db6ece3b8e)


Interpretation: MAE is robust to outliers and treats all errors equally.
III. Model Assumptions
ARIMA (AutoRegressive Integrated Moving Average)
Assumptions:
+ The time series is stationary (constant mean and variance over time).
+ Linearity: The series can be represented as a linear function of past values and past errors.
+ No autocorrelation in residuals (white noise).

Preprocessing:
Differencing (d=1) was applied to ensure stationarity, verified using the Augmented Dickey-Fuller (ADF) Test.

VAR (Vector AutoRegression)
Assumptions:
All variables in the system are stationary (or made stationary via differencing).

The model assumes linearity between each variable and its own lagged values as well as the lagged values of other variables in the system.

No autocorrelation and homoscedasticity in the residuals.

Note: VAR requires multivariate time series, which was constructed using multiple features (e.g., Open, High, Low, etc.).

IV. Time Series Forecasting Project – Orange Workflow Summary
In this project, we implemented a time series forecasting pipeline using Orange and Python scripting to compare the performance of ARIMA and VAR models. Below is a structured overview of our approach:

🔧 Workflow Steps in Orange
![image](https://github.com/user-attachments/assets/ea566c3c-db91-456e-b7aa-0ec6afc60e70)

1.Data Loading: Used the File widget to load the dataset containing time-indexed financial data.

2.Time Series Conversion: Transformed the dataset into a time series format using the appropriate Orange widget, ensuring proper recognition of date/time indices.

3. Handling Missing Values: Applied the Interpolate widget to fill missing values and maintain data continuity.

4.Data Smoothing: Used Moving Transform to smooth the series, which can help reduce noise and improve model performance(Using window slides = 15)

Stationarity Check and Differencing (Python Script)
![image](https://github.com/user-attachments/assets/e7d6a714-53a5-4eb7-82fd-6cef49444ccc)

Before applying ARIMA, I used a Python Script widget to test for stationarity using the Augmented Dickey-Fuller (ADF) test:
If the time series was not stationary, I applied differencing (d=1 or d=2) and re-tested.
![image](https://github.com/user-attachments/assets/af97878c-88cf-455c-8e85-8d95aa81c607)

For this project, differencing with d=1 was sufficient to achieve stationarity.

V. Target and Model Selection

Selected the column Adj Close (mean) as the target variable.
Connected the processed data to both the ARIMA and VAR model widgets.
Used the Prediction widget to compare the forecasting performance of the two models.

Exhaustive Parameter Search for ARIMA (Python Script)
To identify the best combination of ARIMA parameters (p, q) based on the Akaike Information Criterion (AIC):

This is python code: 

import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from Orange.data.pandas_compat import table_to_frame
df = table_to_frame(in_data) # Convert Orange Timeseries data to pandas DataFrame
df.dropna(subset=['ΔAdj Close (mean)'], inplace=True)
y = df['ΔAdj Close (mean)']
def find_best_arima_order(y, p_range, q_range):
    best_aic = np.inf
    best_order = None
    aic_values = []

    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(y, order=(p, 0, q))  # Series is already differenced
                results = model.fit()
                aic_values.append((p, q, results.aic))
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
            except:
                continue

    aic_values.sort(key=lambda x: x[2])
    return best_order, aic_values
p_range = range(0, 6)
q_range = range(0, 6)
best_order, aic_values = find_best_arima_order(y, p_range, q_range)
print(f"Best order (p, q): {best_order}")# Display and plot results
for order in aic_values:
    print(f"p={order[0]}, q={order[1]}, AIC={order[2]}")
p_vals = [x[0] for x in aic_values]
q_vals = [x[1] for x in aic_values]
aic_vals = [x[2] for x in aic_values]
plt.figure(figsize=(10, 6))
sc = plt.scatter(p_vals, q_vals, c=aic_vals, cmap='viridis')
plt.colorbar(sc, label='AIC')
plt.xlabel('p')
plt.ylabel('q')
plt.title('AIC values for ARIMA(p, q) combinations')
plt.show()

Selected ARIMA Model: Based on AIC results, the optimal configuration was ARIMA(4, 1, 5).

VI. Final Comparison
The performance of ARIMA(4,1,5) and VAR models was evaluated using the Prediction widget in Orange. Detailed results, plots, and analysis are available in the accompanying slide deck:
 📄 ARIMA_vs_VARMODEL.pdf

VII: Future work:  Consider testing seasonal ARIMA (SARIMA) if seasonality is present.
