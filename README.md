# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset
2. Preprocess the data
3. Implement Ridge, Lasso and ElasticNet Regularization 
4. Evaluate and visualize the result

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: PRIYADHARSHINI P
RegisterNumber:  212223240128
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('car_price_prediction_.csv')

X = data[['Year', 'Engine Size', 'Mileage', 'Condition']] 
y = data['Price']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Ridge Regression': Pipeline([
        ('poly_features', PolynomialFeatures(degree=2)),
        ('ridge_regressor', Ridge(alpha=1.0))
    ]),
    'Lasso Regression': Pipeline([
        ('poly_features', PolynomialFeatures(degree=2)),
        ('lasso_regressor', Lasso(alpha=1.0))
    ]),
    'ElasticNet Regression': Pipeline([
        ('poly_features', PolynomialFeatures(degree=2)),
        ('elasticnet_regressor', ElasticNet(alpha=1.0))
    ])
}

plt.figure(figsize=(15, 10))

for i, (model_name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{model_name} - Mean Squared Error: {mse}')
    print(f'{model_name} - R-squared: {r2}')
    

    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred, color='blue', label='Predicted Prices')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs Predicted Car Prices using {model_name}')
    plt.legend()

plt.tight_layout()
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/ba77c23c-70b1-4c8a-87b5-7a2e222958f9)
![image](https://github.com/user-attachments/assets/5f0560f6-bc46-4505-9c0b-4b4fe34d6031)

![image](https://github.com/user-attachments/assets/14509bd4-5fcb-4bfe-b0b6-ee39f5b28217)



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
