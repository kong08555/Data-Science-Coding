import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Data URL
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Define column names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# Load data into DataFrame
data = pd.read_csv(url, header=None, names=columns)

# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create StandardScaler for normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create models: Naïve Bayes and kNN
nb_model = GaussianNB()
knn_model = KNeighborsClassifier(n_neighbors=5)

# Number of experiments
n_experiments = 30

# Store results
nb_results = []
knn_results = []

for i in range(1, n_experiments + 1):
    # Change the seed number for each iteration (from 1 to 30)
    seed = i
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    
    # Evaluate Naïve Bayes model with cross-validation
    nb_scores = cross_val_score(nb_model, X_scaled, y, cv=kf, scoring='accuracy')
    nb_results.append(nb_scores.mean())
    
    # Evaluate kNN model with cross-validation
    knn_scores = cross_val_score(knn_model, X_scaled, y, cv=kf, scoring='accuracy')
    knn_results.append(knn_scores.mean())

    # Display results for each experiment
    print(f'Experiment {i:02d}: \nNaïve Bayes Accuracy: {nb_scores.mean():.4f}, \nkNN Accuracy: {knn_scores.mean():.4f}')
    print('---------------------------------------------------')

# Display average results and standard deviation of accuracy
print(f'\nNaïve Bayes Average Accuracy: {np.mean(nb_results):.4f} (+/- {np.std(nb_results) * 2:.4f})')
print(f'kNN Average Accuracy: {np.mean(knn_results):.4f} (+/- {np.std(knn_results) * 2:.4f})')


