# **Classical Linear Regression Model (CRLM): Assumptions and Diagnostics**

## **Why Do Assumptions Matter?**
The validity of regression results depends on certain assumptions about the data and the model. Violating these assumptions can lead to biased, inefficient, or invalid results.

---

## **Assumptions of the CRLM**
1. **Linearity**: The relationship between $X$ and $Y$ is linear.
2. **Zero Mean of Errors**: The average value of residuals is zero ($E(u) = 0$).
3. **Homoscedasticity**: Error variance is constant ($Var(u) = \sigma^2$).
4. **No Autocorrelation**: Errors are uncorrelated ($Cov(u_i, u_j) = 0$ for $i \neq j$).
5. **Normality**: Errors follow a normal distribution ($N(0, \sigma^{2} ) $).

---

## **Diagnosing and Addressing Violations**

### **1. Linearity**
A non-linear relationship between $X$ and $Y$ can lead to incorrect coefficient estimates.

#### Example in Python:
```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated data
X = np.linspace(0, 10, 100)
Y = 2 + 3 * X + np.random.normal(0, 1, 100)

# Scatter plot to check linearity
plt.scatter(X, Y)
plt.title("Linearity Check")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

If non-linearity is detected, consider transforming variables (e.g., logarithmic or polynomial transformations).

---

### **2. Homoscedasticity (Constant Variance)**
When error variance is not constant (heteroscedasticity), OLS estimates are inefficient.

#### Detection: Residual Plot
```python
from sklearn.linear_model import LinearRegression

# Fit a linear model
model = LinearRegression()
X = X.reshape(-1, 1)
model.fit(X, Y)
Y_pred = model.predict(X)

# Residuals
residuals = Y - Y_pred

# Plot residuals
plt.scatter(Y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot for Homoscedasticity")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()
```

#### Formal Test: White’s Test
```python
import statsmodels.stats.diagnostic as smd

# Perform White's test
import statsmodels.api as sm
X_with_const = sm.add_constant(X)
white_test = smd.het_white(residuals, X_with_const)
print(f"White's Test p-value: {white_test[1]}")
```
If heteroscedasticity is present, use robust standard errors or transform variables.

---

### **3. Autocorrelation**
Autocorrelation occurs when residuals are correlated, common in time-series data.

#### Detection: Durbin-Watson Test
```python
from statsmodels.stats.stattools import durbin_watson

# Durbin-Watson test
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_stat}")
```
A value close to 2 indicates no autocorrelation.

#### Remedy:
- Include lagged variables in the model.
- Use autoregressive models like ARIMA for time-series data.

---

### **4. Multicollinearity**
Multicollinearity arises when independent variables are highly correlated, leading to unreliable coefficient estimates.

#### Detection: Correlation Matrix and Variance Inflation Factor (VIF)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Correlation matrix
import pandas as pd
data = pd.DataFrame({'X1': X.flatten(), 'X2': 0.8 * X.flatten() + np.random.normal(0, 0.1, 100)})
print(data.corr())

# VIF
X_multi = sm.add_constant(data)
vif = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]
print(f"VIF values: {vif}")
```
If multicollinearity exists, consider dropping or combining variables, or using PCA (Principal Component Analysis).

---

### **5. Normality of Errors**
Non-normal errors affect hypothesis testing and confidence intervals.

#### Detection: Histogram and Shapiro-Wilk Test
```python
from scipy.stats import shapiro

# Histogram of residuals
plt.hist(residuals, bins=20, edgecolor='black')
plt.title("Residuals Histogram")
plt.show()

# Shapiro-Wilk test
shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test p-value: {shapiro_test.pvalue}")
```
If errors are non-normal, consider transformations or non-parametric methods.

---

## **Advanced Diagnostics**

### **1. Ramsey’s RESET Test**
Tests whether the functional form of the model is correct.
```python
from statsmodels.stats.diagnostic import linear_reset

reset_test = linear_reset(ols_model, power=2, use_f=True)
print(f"RESET Test p-value: {reset_test.pvalue}")
```

### **2. Chow Test for Structural Breaks**
Tests if model coefficients are stable across sub-samples.
```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# Perform the Chow test
cusum_test = breaks_cusumolsresid(ols_model.resid)
print(f"CUSUM Test p-value: {cusum_test[1]}")
```

---

## **Practical Remedies for Assumption Violations**
1. **Transform Data**:
   - Logarithms, squares, or differencing can address non-linearity and heteroscedasticity.
2. **Use Robust Methods**:
   - Employ robust standard errors or generalized least squares (GLS).
3. **Add or Modify Variables**:
   - Include lagged variables to address autocorrelation or remove multicollinear predictors.
4. **Re-specify the Model**:
   - Adjust the functional form (e.g., Ramsey RESET test insights).

---

## **Why Do Diagnostics Matter?**
Validating regression assumptions ensures:
- **Unbiased Estimates**: Accurate representation of relationships.
- **Efficient Predictions**: Reliable forecasts.
- **Valid Inferences**: Confidence in hypothesis testing and decision-making.

Python provides a comprehensive toolkit for diagnosing and addressing assumption violations, enabling accurate and actionable econometric modeling.
