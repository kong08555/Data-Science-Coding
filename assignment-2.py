import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Step 1: Load California Housing Dataset
cal_housing = fetch_california_housing()

# Step 2: Pull data to display 
# Step 3: Set dataset into a DataFrame
# Create DataFrame from the dataset
df = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)

# Add target column to the DataFrame
df['target'] = cal_housing.target

# Display the dataset
print("Dataset:")
print(df.head())
# print("\nStatistical Summary of the Dataset:")
# print(df.describe())

# Step 4: Examine Correlation plot and create Boxplot
# Create Correlation plot
plt.figure(figsize=(10, 8))

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap for the correlation matrix 
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')  
plt.title('Correlation Matrix')
plt.show()

# Create Boxplot 
plt.figure(figsize=(12, 6))

# Create a boxplot for the DataFrame features
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Boxplot of Features')
plt.show()

# Step 5: Testing 10-fold Cross-Validation
# Separate features (X) and target (y)
X = df.drop('target', axis=1)  # Drop the target column from features
y = df['target']  # Target variable

# Create KFold for Cross-Validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize the Linear Regression model
regressor = LinearRegression()

# Perform Cross-Validation with 10 folds
scores = cross_val_score(regressor, X, y, cv=kf, scoring='r2')

# Print Cross-Validation scores
print(f'10-Folds Cross-Validation Scores: {scores}')
print(f'Mean R2 Score: {np.mean(scores)}')

# Step 6: Train the model on the entire dataset
# Train the Linear Regression model on all data
regressor.fit(X, y)

# Predict target values using the trained model
predictions = regressor.predict(X)

# Display a sample of predictions
print("\nSample of Predictions:")
print(predictions[:10])
