import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics

gold_data = pd.read_csv('gld_price_data.csv')

# Display basic information about the data
print(gold_data.head())
print(gold_data.tail())
print(gold_data.shape)
print(gold_data.info())
print(gold_data.isnull().sum())
print(gold_data.describe())

# Convert 'Date' column to datetime
gold_data['Date'] = pd.to_datetime(gold_data['Date'])

# Calculate correlation matrix
correlation = gold_data.corr()

# Plot heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cmap="YlGnBu", annot=True)
plt.show()

# Splitting the data into input and output variables
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize Linear Regression model
regressor = LinearRegression()

# Define hyperparameter grid for Grid Search
param_grid = {'fit_intercept': [True, False]}

# Perform Grid Search for hyperparameter tuning
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, Y_train)

# Get the best model and its hyperparameters
best_regressor = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

# Fit the best model
best_regressor.fit(X_train, Y_train)

# Make predictions
test_data_prediction = best_regressor.predict(X_test)

# Evaluate model performance
error_score = metrics.r2_score(Y_test, test_data_prediction)
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
mse = metrics.mean_squared_error(Y_test, test_data_prediction)

print("R squared error:", error_score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Scatter plots to visualize relationship between features and target variable
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

for i, column in enumerate(X.columns):
    sns.scatterplot(x=X[column], y=Y, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_xlabel(column)
    axes[i//2, i%2].set_ylabel('GLD Price')

plt.tight_layout()
plt.show()

# Residual plot
residuals = Y_test - test_data_prediction
plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_data_prediction, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted GLD Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Save the trained model
import pickle

pickle.dump(best_regressor, open("gold_price_predictor.pkl", "wb"))