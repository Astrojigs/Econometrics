# **Analysis of the Classical Linear Regression Model**

## **Extending the Linear Regression Model**

### **What is Multiple Linear Regression?**
Multiple Linear Regression allows us to analyze how a dependent variable (`Y`) depends on multiple independent variables (`X₁, X₂, ..., Xₖ`). For example, house prices might depend on square footage, number of bedrooms, and location.

The model is represented as:

$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_kX_k + u$

Where:
- \$beta_0$: Intercept.
- \$beta_1, \beta_2, ..., \beta_k$: Coefficients for each independent variable.
- $u$: Error term.

### Example in Python:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset for multiple regression
data = {
    'SquareFootage': [1500, 2000, 2500, 1800, 2200],
    'Bedrooms': [3, 4, 3, 2, 4],
    'LocationScore': [8, 9, 7, 6, 8],
    'Price': [300000, 400000, 350000, 280000, 390000]
}
df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['SquareFootage', 'Bedrooms', 'LocationScore']]
Y = df['Price']

# Fit the regression model
model = LinearRegression()
model.fit(X, Y)

# Coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
```

---

## **Matrix Representation**
For multiple regression, we often use matrix notation:

$Y = X\beta + u$

Where:
- $Y$: Dependent variable (vector).
- $X$: Matrix of independent variables.
- \$beta$: Vector of coefficients.
- $u$: Error vector.

This is computationally efficient, especially for large datasets.

### Example in Python:
```python
import numpy as np

# Matrix representation
X_matrix = np.array([[1, 1500, 3, 8],
                     [1, 2000, 4, 9],
                     [1, 2500, 3, 7],
                     [1, 1800, 2, 6],
                     [1, 2200, 4, 8]])  # Adding a column of ones for the intercept
Y_vector = np.array([300000, 400000, 350000, 280000, 390000])

# Estimate coefficients using the OLS formula: β = (X'X)^(-1)X'Y
beta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y_vector
print(f"Estimated Coefficients: {beta}")
```

---

## **Calculating Goodness-of-Fit**

### **R² (Coefficient of Determination)**
R² measures how well the model explains the variance in `Y`:
$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$

Where:
- $\text{RSS}$: Residual Sum of Squares.
- $\text{TSS}$: Total Sum of Squares.

### Example in Python:
```python
# Predicted values
Y_pred = model.predict(X)

# RSS and TSS
RSS = np.sum((Y - Y_pred) ** 2)
TSS = np.sum((Y - np.mean(Y)) ** 2)

# R² calculation
r_squared = 1 - (RSS / TSS)
print(f"R²: {r_squared}")
```

---

## **Quantile Regression**
Quantile Regression allows modeling of relationships at different quantiles (e.g., median instead of mean). This is useful for non-linear relationships or when extreme values are of interest.

### Example in Python (using `statsmodels`):
```python
import statsmodels.formula.api as smf

# Quantile regression at the median
quantile_model = smf.quantreg('Price ~ SquareFootage + Bedrooms + LocationScore', df).fit(q=0.5)
print(quantile_model.summary())
```

---

## **Testing Hypotheses**

### **F-Test for Multiple Coefficients**
The F-test is used to evaluate whether multiple coefficients are jointly significant:
- **Null hypothesis** $(H_0$): All coefficients $(\beta_1, \beta_2, ...$) are zero.
- **Alternative hypothesis** $(H_1$): At least one coefficient is non-zero.

### Example in Python:
```python
# F-test is included in statsmodels' OLS summary
import statsmodels.api as sm

X_with_const = sm.add_constant(X)  # Add intercept
ols_model = sm.OLS(Y, X_with_const).fit()
print(ols_model.summary())
```

---

## **Handling Overfitting**

### **Adjusted R²**
Adjusted R² adjusts for the number of predictors:

$\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)$

Where:
- $n$: Number of observations.
- $k$: Number of predictors.

### Example in Python:
```python
n = len(Y)
k = X.shape[1]
adjusted_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - k - 1))
print(f"Adjusted R²: {adjusted_r_squared}")
```

---

## **Practical Applications**
1. **Hedonic Pricing Models**:
   - Used in real estate to value properties based on features like size, location, etc.
2. **Risk Assessment**:
   - Analyze how various factors (e.g., market volatility, interest rates) affect stock returns.

---

## **Why Does This Matter?**
Multiple regression expands the capability of econometrics by:
- **Explaining complex phenomena**: How multiple factors interact to influence outcomes.
- **Improving predictions**: More variables lead to better-informed models.
- **Providing actionable insights**: For policy, investment, or business strategies.

Python makes advanced regression techniques easy to implement and analyze, bridging theory and practice.
