---
jupyter:
  kernelspec:
    display_name: "Python \\[conda env:base\\] \\*"
    language: python
    name: conda-base-py
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.7
  nbformat: 4
  nbformat_minor: 4
---

::: {.cell .markdown}
# Ridge and Lasso Regression - Lab
:::

::: {.cell .markdown}
## Introduction
:::

::: {.cell .markdown}
In this lab, you\'ll practice your knowledge of ridge and lasso
regression!
:::

::: {.cell .markdown}
## Objectives
:::

::: {.cell .markdown}
In this lab you will:

-   Use lasso and ridge regression with scikit-learn
-   Compare and contrast lasso, ridge and non-regularized regression
:::

::: {.cell .markdown}
## Housing Prices Data
:::

::: {.cell .markdown}
We\'ll use this version of the Ames Housing dataset:
:::

::: {.cell .code execution_count="1"}
``` python
# Run this cell without changes
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('housing_prices.csv', index_col=0)
df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 1460 entries, 1 to 1460
    Data columns (total 80 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   MSSubClass     1460 non-null   int64  
     1   MSZoning       1460 non-null   object 
     2   LotFrontage    1201 non-null   float64
     3   LotArea        1460 non-null   int64  
     4   Street         1460 non-null   object 
     5   Alley          91 non-null     object 
     6   LotShape       1460 non-null   object 
     7   LandContour    1460 non-null   object 
     8   Utilities      1460 non-null   object 
     9   LotConfig      1460 non-null   object 
     10  LandSlope      1460 non-null   object 
     11  Neighborhood   1460 non-null   object 
     12  Condition1     1460 non-null   object 
     13  Condition2     1460 non-null   object 
     14  BldgType       1460 non-null   object 
     15  HouseStyle     1460 non-null   object 
     16  OverallQual    1460 non-null   int64  
     17  OverallCond    1460 non-null   int64  
     18  YearBuilt      1460 non-null   int64  
     19  YearRemodAdd   1460 non-null   int64  
     20  RoofStyle      1460 non-null   object 
     21  RoofMatl       1460 non-null   object 
     22  Exterior1st    1460 non-null   object 
     23  Exterior2nd    1460 non-null   object 
     24  MasVnrType     588 non-null    object 
     25  MasVnrArea     1452 non-null   float64
     26  ExterQual      1460 non-null   object 
     27  ExterCond      1460 non-null   object 
     28  Foundation     1460 non-null   object 
     29  BsmtQual       1423 non-null   object 
     30  BsmtCond       1423 non-null   object 
     31  BsmtExposure   1422 non-null   object 
     32  BsmtFinType1   1423 non-null   object 
     33  BsmtFinSF1     1460 non-null   int64  
     34  BsmtFinType2   1422 non-null   object 
     35  BsmtFinSF2     1460 non-null   int64  
     36  BsmtUnfSF      1460 non-null   int64  
     37  TotalBsmtSF    1460 non-null   int64  
     38  Heating        1460 non-null   object 
     39  HeatingQC      1460 non-null   object 
     40  CentralAir     1460 non-null   object 
     41  Electrical     1459 non-null   object 
     42  1stFlrSF       1460 non-null   int64  
     43  2ndFlrSF       1460 non-null   int64  
     44  LowQualFinSF   1460 non-null   int64  
     45  GrLivArea      1460 non-null   int64  
     46  BsmtFullBath   1460 non-null   int64  
     47  BsmtHalfBath   1460 non-null   int64  
     48  FullBath       1460 non-null   int64  
     49  HalfBath       1460 non-null   int64  
     50  BedroomAbvGr   1460 non-null   int64  
     51  KitchenAbvGr   1460 non-null   int64  
     52  KitchenQual    1460 non-null   object 
     53  TotRmsAbvGrd   1460 non-null   int64  
     54  Functional     1460 non-null   object 
     55  Fireplaces     1460 non-null   int64  
     56  FireplaceQu    770 non-null    object 
     57  GarageType     1379 non-null   object 
     58  GarageYrBlt    1379 non-null   float64
     59  GarageFinish   1379 non-null   object 
     60  GarageCars     1460 non-null   int64  
     61  GarageArea     1460 non-null   int64  
     62  GarageQual     1379 non-null   object 
     63  GarageCond     1379 non-null   object 
     64  PavedDrive     1460 non-null   object 
     65  WoodDeckSF     1460 non-null   int64  
     66  OpenPorchSF    1460 non-null   int64  
     67  EnclosedPorch  1460 non-null   int64  
     68  3SsnPorch      1460 non-null   int64  
     69  ScreenPorch    1460 non-null   int64  
     70  PoolArea       1460 non-null   int64  
     71  PoolQC         7 non-null      object 
     72  Fence          281 non-null    object 
     73  MiscFeature    54 non-null     object 
     74  MiscVal        1460 non-null   int64  
     75  MoSold         1460 non-null   int64  
     76  YrSold         1460 non-null   int64  
     77  SaleType       1460 non-null   object 
     78  SaleCondition  1460 non-null   object 
     79  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(34), object(43)
    memory usage: 923.9+ KB
:::
:::

::: {.cell .markdown}
More information about the features is available in the
`data_description.txt` file in this repository.
:::

::: {.cell .markdown}
## Data Preparation

The code below:

-   Separates the data into `X` (predictor) and `y` (target) variables
-   Splits the data into 75-25 training-test sets, with a `random_state`
    of 10
-   Separates each of the `X` values into continuous vs. categorical
    features
-   Fills in missing values (using different strategies for continuous
    vs. categorical features)
-   Scales continuous features to a range of 0 to 1
-   Dummy encodes categorical features
-   Combines the preprocessed continuous and categorical features back
    together
:::

::: {.cell .code execution_count="2"}
``` python
# Run this cell without changes
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create X and y
y = df['SalePrice']
X = df.drop(columns=['SalePrice'])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

# Separate X data into continuous vs. categorical
X_train_cont = X_train.select_dtypes(include='number')
X_test_cont = X_test.select_dtypes(include='number')
X_train_cat = X_train.select_dtypes(exclude='number')
X_test_cat = X_test.select_dtypes(exclude='number')

# Impute missing values using SimpleImputer, median for continuous and
# filling in 'missing' for categorical
impute_cont = SimpleImputer(strategy='median')
X_train_cont = impute_cont.fit_transform(X_train_cont)
X_test_cont = impute_cont.transform(X_test_cont)
impute_cat = SimpleImputer(strategy='constant', fill_value='missing')
X_train_cat = impute_cat.fit_transform(X_train_cat)
X_test_cat = impute_cat.transform(X_test_cat)

# Scale continuous values using MinMaxScaler
scaler = MinMaxScaler()
X_train_cont = scaler.fit_transform(X_train_cont)
X_test_cont = scaler.transform(X_test_cont)

# Dummy encode categorical values using OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train_cat)
X_test_cat = ohe.transform(X_test_cat)

# Combine everything back together
X_train_preprocessed = np.asarray(np.concatenate([X_train_cont, X_train_cat.todense()], axis=1))
X_test_preprocessed = np.asarray(np.concatenate([X_test_cont, X_test_cat.todense()], axis=1))
```
:::

::: {.cell .markdown}
## Linear Regression Model

Let\'s use this data to build a first naive linear regression model. Fit
the model on the training data (`X_train_preprocessed`), then compute
the R-Squared and the MSE for both the training and test sets.
:::

::: {.cell .code execution_count="7"}
``` python
# Replace None with appropriate code
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# Fit the model
linreg = LinearRegression()
linreg.fit(X_train_preprocessed, y_train)



# Print R2 and MSE for training and test sets
print("mean_squared_error train: ", mean_squared_error(y_train, linreg.predict(X_train_preprocessed)))
print("mean_squared_error test: ", mean_squared_error(y_test, linreg.predict(X_test_preprocessed)))
print("r2_score train: ", r2_score(y_train, linreg.predict(X_train_preprocessed)))
print("r2_score test: ", r2_score(y_test, linreg.predict(X_test_preprocessed)))
```

::: {.output .stream .stdout}
    mean_squared_error train:  403057708.1716895
    mean_squared_error test:  8.071035376599559e+29
    r2_score train:  0.9358292543968024
    r2_score test:  -1.2650158636860298e+20
:::
:::

::: {.cell .markdown}
Notice the severe overfitting above; our training R-Squared is very
high, but the test R-Squared is negative! Similarly, the scale of the
test MSE is orders of magnitude higher than that of the training MSE.
:::

::: {.cell .markdown}
## Ridge and Lasso Regression
:::

::: {.cell .markdown}
Use all the data (scaled features and dummy categorical variables,
`X_train_preprocessed`) to build some models with regularization - two
each for lasso and ridge regression. Each time, look at R-Squared and
MSE.

Remember that you can use the scikit-learn documentation if you don\'t
remember how to import or use these classes:

-   [`Lasso`
    documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
-   [`Ridge`
    documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
:::

::: {.cell .markdown}
### Lasso
:::

::: {.cell .markdown}
#### With default hyperparameters (`alpha` = 1) {#with-default-hyperparameters-alpha--1}
:::

::: {.cell .code execution_count="8"}
``` python
# Your code here
lasso_reg = Lasso(alpha=1)
lasso_reg.fit(X_train_preprocessed, y_train)

# Print R2 and MSE for training and test sets
print("mean_squared_error train: ", mean_squared_error(y_train, lasso_reg.predict(X_train_preprocessed)))
print("mean_squared_error test: ", mean_squared_error(y_test, lasso_reg.predict(X_test_preprocessed)))
print("r2_score train: ", r2_score(y_train, lasso_reg.predict(X_train_preprocessed)))
print("r2_score test: ", r2_score(y_test, lasso_reg.predict(X_test_preprocessed)))
```

::: {.output .stream .stdout}
    mean_squared_error train:  403443180.26656115
    mean_squared_error test:  704367122.4118747
    r2_score train:  0.9357678834535463
    r2_score test:  0.8896008328382142
:::
:::

::: {.cell .markdown}
#### With a higher regularization hyperparameter (`alpha` = 10) {#with-a-higher-regularization-hyperparameter-alpha--10}
:::

::: {.cell .code execution_count="9"}
``` python
# Your code here
lasso_reg = Lasso(alpha=10)
lasso_reg.fit(X_train_preprocessed, y_train)

# Print R2 and MSE for training and test sets
print("mean_squared_error train: ", mean_squared_error(y_train, lasso_reg.predict(X_train_preprocessed)))
print("mean_squared_error test: ", mean_squared_error(y_test, lasso_reg.predict(X_test_preprocessed)))
print("r2_score train: ", r2_score(y_train, lasso_reg.predict(X_train_preprocessed)))
print("r2_score test: ", r2_score(y_test, lasso_reg.predict(X_test_preprocessed)))
```

::: {.output .stream .stdout}
    mean_squared_error train:  414404517.19166684
    mean_squared_error test:  650157273.4932028
    r2_score train:  0.9340227309628953
    r2_score test:  0.8980974278412504
:::
:::

::: {.cell .markdown}
## Ridge
:::

::: {.cell .markdown}
#### With default hyperparameters (`alpha` = 1) {#with-default-hyperparameters-alpha--1}
:::

::: {.cell .code execution_count="10"}
``` python
# Your code here
Ridge_reg = Ridge(alpha=1)
Ridge_reg.fit(X_train_preprocessed, y_train)

# Print R2 and MSE for training and test sets
print("mean_squared_error train: ", mean_squared_error(y_train, Ridge_reg.predict(X_train_preprocessed)))
print("mean_squared_error test: ", mean_squared_error(y_test, Ridge_reg.predict(X_test_preprocessed)))
print("r2_score train: ", r2_score(y_train, Ridge_reg.predict(X_train_preprocessed)))
print("r2_score test: ", r2_score(y_test, Ridge_reg.predict(X_test_preprocessed)))
```

::: {.output .stream .stdout}
    mean_squared_error train:  497616506.0742805
    mean_squared_error test:  724504516.4583927
    r2_score train:  0.9207745651011281
    r2_score test:  0.8864445930581236
:::
:::

::: {.cell .markdown}
#### With higher regularization hyperparameter (`alpha` = 10) {#with-higher-regularization-hyperparameter-alpha--10}
:::

::: {.cell .code execution_count="11"}
``` python
# Your code here
Ridge_reg = Ridge(alpha=10)
Ridge_reg.fit(X_train_preprocessed, y_train)

# Print R2 and MSE for training and test sets
print("mean_squared_error train: ", mean_squared_error(y_train, Ridge_reg.predict(X_train_preprocessed)))
print("mean_squared_error test: ", mean_squared_error(y_test, Ridge_reg.predict(X_test_preprocessed)))
print("r2_score train: ", r2_score(y_train, Ridge_reg.predict(X_train_preprocessed)))
print("r2_score test: ", r2_score(y_test, Ridge_reg.predict(X_test_preprocessed)))
```

::: {.output .stream .stdout}
    mean_squared_error train:  697520320.9975777
    mean_squared_error test:  768710500.2896671
    r2_score train:  0.8889479144938476
    r2_score test:  0.8795159564945245
:::
:::

::: {.cell .markdown}
## Comparing the Metrics

Which model seems best, based on the metrics?
:::

::: {.cell .markdown}
# Write your conclusions here:

In both Ridge and Lasso using a higher regularization hyperparameter led
to better results but Lasso had the best metrics overall.
:::

::: {.cell .markdown}
```{=html}
<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

In terms of both R-Squared and MSE, the `Lasso` model with `alpha`=10 has the best metric results.

(Remember that better R-Squared is higher, whereas better MSE is lower.)

</details>
```
:::

::: {.cell .markdown}
## Comparing the Parameters

Compare the number of parameter estimates that are (very close to) 0 for
the `Ridge` and `Lasso` models with `alpha`=10.

Use 10\*\*(-10) as an estimate that is very close to 0.
:::

::: {.cell .code}
``` python
# Number of Ridge params almost zero
```
:::

::: {.cell .code}
``` python
# Number of Lasso params almost zero
```
:::

::: {.cell .code}
``` python
# Compare and interpret these results
```
:::

::: {.cell .markdown}
```{=html}
<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

The ridge model did not penalize any coefficients to 0, while the lasso model removed about 1/4 of the coefficients. The lasso model essentially performed variable selection for us, and got the best metrics as a result!

</details>
```
:::

::: {.cell .markdown}
## Finding an Optimal Alpha

Earlier we tested two values of `alpha` to see how it affected our MSE
and the value of our coefficients. We could continue to guess values of
`alpha` for our ridge or lasso regression one at a time to see which
values minimize our loss, or we can test a range of values and pick the
alpha which minimizes our MSE. Here is an example of how we would do
this:
:::

::: {.cell .code execution_count="12"}
``` python
# Run this cell without changes
import matplotlib.pyplot as plt
%matplotlib inline

train_mse = []
test_mse = []
alphas = np.linspace(0, 200, num=50)

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_preprocessed, y_train)
    
    train_preds = lasso.predict(X_train_preprocessed)
    train_mse.append(mean_squared_error(y_train, train_preds))
    
    test_preds = lasso.predict(X_test_preprocessed)
    test_mse.append(mean_squared_error(y_test, test_preds))

fig, ax = plt.subplots()
ax.plot(alphas, train_mse, label='Train')
ax.plot(alphas, test_mse, label='Test')
ax.set_xlabel('alpha')
ax.set_ylabel('MSE')

# np.argmin() returns the index of the minimum value in a list
optimal_alpha = alphas[np.argmin(test_mse)]

# Add a vertical line where the test MSE is minimized
ax.axvline(optimal_alpha, color='black', linestyle='--')
ax.legend();

print(f'Optimal Alpha Value: {int(optimal_alpha)}')
```

::: {.output .stream .stdout}
    Optimal Alpha Value: 28
:::
:::

::: {.cell .markdown}
Take a look at this graph of our training and test MSE against `alpha`.
Try to explain to yourself why the shapes of the training and test
curves are this way. Make sure to think about what `alpha` represents
and how it relates to overfitting vs underfitting.

------------------------------------------------------------------------

```{=html}
<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>

For `alpha` values below 28, the model is overfitting. As `alpha` increases up to 28, the MSE for the training data increases and MSE for the test data decreases, indicating that we are reducing overfitting.

For `alpha` values above 28, the model is starting to underfit. You can tell because _both_ the train and the test MSE values are increasing.

</details>
```
:::

::: {.cell .markdown}
## Summary

Well done! You now know how to build lasso and ridge regression models,
use them for feature selection and find an optimal value for `alpha`.
:::
