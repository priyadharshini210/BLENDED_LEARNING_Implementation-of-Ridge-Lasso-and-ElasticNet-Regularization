# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Import the required libraries.

2. **Load Dataset**:  
   Load the dataset into the environment.

3. **Data Preprocessing**:  
   Handle missing values and encode categorical variables.

4. **Define Features and Target**:  
   Split the dataset into features (X) and the target variable (y).

5. **Create Polynomial Features**:  
   Generate polynomial features from the data.

6. **Set Up Pipelines**:  
   Create pipelines for Ridge, Lasso, and ElasticNet models.

7. **Train Models**:  
   Fit each model to the training data.

8. **Evaluate Model Performance**:  
   Assess performance using the R² score and Mean Squared Error (MSE).

9. **Compare Results**:  
   Compare the performance of the models.

## Program:
```python
'''
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Priyadharshini.P
RegisterNumber: 21223240128
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r'C:\Users\admin\Downloads\encoded_car_data (6).csv'
car_data = pd.read_csv(file_path)

# Splitting the dataset into features (X) and target (y)
X = car_data.drop(columns=['price'])
y = car_data['price']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to create pipelines and evaluate models
def evaluate_model(model, model_name):
    # Create a pipeline
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Model")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    print("-" * 40)
# Ridge Regression
evaluate_model(Ridge(alpha=1.0), "Ridge")

# Lasso Regression with increased max_iter
# Use PolynomialFeatures with degree=1 or 2
evaluate_model(Lasso(alpha=1, max_iter=100000), "Lasso with Reduced Polynomial Degree")

# ElasticNet Regression with increased max_iter
evaluate_model(ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000000), "ElasticNet")


```

## Output:

![image](https://github.com/user-attachments/assets/8a1ff4d8-a8f6-41ba-a936-ed37218c9bf4)



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
