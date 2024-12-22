
# **Univariate Time Series Analysis**

## **What is Time Series Analysis?**
Time series analysis is the study of data points collected or recorded at specific intervals over time. It focuses on understanding patterns, trends, and dependencies within the data to make predictions.

---

## **Key Concepts in Time Series**

### **1. Stationarity**
A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. This is critical for building reliable models.

- **Strict Stationarity**: The entire distribution is time-invariant.
- **Weak Stationarity (Covariance Stationarity)**:
  - $E(Y_t) = \mu$ (constant mean),
  - $Var(Y_t) = \sigma^2$ (constant variance),
  - $Cov(Y_t, Y_{t+h})$ depends only on lag $h$, not time $t$.

#### Example in Python:
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a stationary time series
np.random.seed(42)
stationary_series = np.random.normal(0, 1, 100)

# Plot the series
plt.plot(stationary_series)
plt.title("Stationary Time Series")
plt.show()
```

If non-stationary, use differencing or detrending to make the series stationary.

---

### **2. Autocorrelation Function (ACF)**
The ACF measures the correlation between observations at different lags. It helps identify patterns and seasonality.

#### Example in Python:
```python
from statsmodels.graphics.tsaplots import plot_acf

# Plot ACF
plot_acf(stationary_series, lags=20)
plt.show()
```

---

### **3. White Noise**
A white noise process has:
- $E(Y_t) = 0$,
- $Var(Y_t) = \sigma^2$,
- No autocorrelation.

If a series is white noise, it cannot be predicted.

#### Example in Python:
```python
# Generate white noise
white_noise = np.random.normal(0, 1, 100)

# Plot white noise
plt.plot(white_noise)
plt.title("White Noise")
plt.show()
```

---

## **Modeling Time Series**

### **1. Moving Average (MA) Models**
An MA($q$) model expresses $Y_t$ as a function of past error terms:

$Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q} $

Where $\epsilon_t$ are white noise errors.

#### Example in Python:
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit MA(2) model
ma_model = ARIMA(stationary_series, order=(0, 0, 2))
ma_fit = ma_model.fit()
print(ma_fit.summary())
```

---

### **2. Autoregressive (AR) Models**
An AR($p$) model relates $Y_t$ to its own lagged values:

$Y_t = \phi_1Y_{t-1} + \phi_2Y_{t-2} + ... + \phi_pY_{t-p} + \epsilon_t$

#### Example in Python:
```python
# Fit AR(1) model
ar_model = ARIMA(stationary_series, order=(1, 0, 0))
ar_fit = ar_model.fit()
print(ar_fit.summary())
```

---

### **3. ARMA Models**
Combines AR and MA models:

$Y_t = \phi_1Y_{t-1} + \phi_2Y_{t-2} + ... + \phi_pY_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}$

#### Example in Python:
```python
# Fit ARMA(1, 1) model
arma_model = ARIMA(stationary_series, order=(1, 0, 1))
arma_fit = arma_model.fit()
print(arma_fit.summary())
```

---

### **4. ARIMA Models**
Adds differencing ($d$) to ARMA for non-stationary series:

$Y_t = \phi_1Y_{t-1} + ... + \phi_pY_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + ...$

Where $d$ indicates the order of differencing.

#### Example in Python:
```python
# Fit ARIMA(1, 1, 1) model
arima_model = ARIMA(stationary_series, order=(1, 1, 1))
arima_fit = arima_model.fit()
print(arima_fit.summary())
```

---

### **5. Exponential Smoothing**
Weights past observations with exponentially decreasing influence.

- **Single Exponential Smoothing**:
  
$S_t = \alpha Y_t + (1 - \alpha)S_{t-1}$

- **Holt’s Method**: Adds trends.
- **Winter’s Method**: Adds seasonality.

#### Example in Python:
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit exponential smoothing
exp_model = ExponentialSmoothing(stationary_series, trend="add", seasonal=None)
exp_fit = exp_model.fit()
print(exp_fit.summary())
```

---

## **Forecasting and Accuracy**

### **Forecasting**
Forecasts can be made using AR, MA, ARMA, ARIMA, or Exponential Smoothing models.

#### Example in Python:
```python
# Forecast with ARIMA
forecast = arima_fit.forecast(steps=5)
print(f"Forecasted Values: {forecast}")
```

---

### **Measuring Accuracy**
Common metrics:
1. **Mean Absolute Error (MAE)**:

$MAE = \frac{1}{n} \sum_{i=1}^n |Y_i - \hat{Y}_i|$

2. **Mean Squared Error (MSE)**:

$MSE = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat{Y}_i)^2$

#### Example in Python:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate errors
mae = mean_absolute_error(stationary_series[1:], Y_pred[:-1])
mse = mean_squared_error(stationary_series[1:], Y_pred[:-1])

print(f"MAE: {mae}, MSE: {mse}")
```

---

## **Why Does This Matter?**
Univariate time series analysis is crucial for:
- **Understanding Patterns**: Trends, seasonality, and noise in the data.
- **Forecasting**: Predicting future values, e.g., stock prices or sales.
- **Decision-Making**: Informed planning and policy development.

Python makes it easy to analyze and model time series, providing powerful tools for both basic and advanced forecasting tasks.
