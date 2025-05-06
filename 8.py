import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances

base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "Mall_Customers.csv")

# Load dataset
data = pd.read_csv(data_path)

# Data overview (optional)
print(data.head())
print(data.isnull().sum())
print(data.info())
print(data.describe())

# Label encoding for categorical columns
le = LabelEncoder()
for i in data.columns:
    if data[i].dtype == "object":
        data[i] = le.fit_transform(data[i])


# Drop unnecessary columns
data.drop(columns=["CustomerID"], inplace=True)

# Rename columns for clarity
data = data.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Spending'})

# Separate features and target
x = data.drop(columns=["Spending"])
y = data["Spending"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Optional: Reduce dimensions for 2D visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the PCA reduced data in 2D
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
plt.title('2D Visualization using PCA')
plt.show()

# Perform K-Means clustering with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
y_kmeans = kmeans.predict(X_scaled)

# Add cluster labels to the data
data['Cluster'] = y_kmeans


# Calculate inertia for different k values and use elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method For Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()

# Visualize the clusters with PCA 2D and color-coding by cluster labels
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans, palette='Set2')
plt.title(f'KMeans Clustering with k={k} (PCA Reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# Evaluate the clustering quality using Silhouette Score for various k
sil_scores = []
k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

# Plot Silhouette Scores vs. Number of Clusters
plt.figure(figsize=(8, 5))
plt.plot(k_values, sil_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Print the best k based on the highest Silhouette Score
best_k = k_values[sil_scores.index(max(sil_scores))]
print(f"Best k based on Silhouette Score: {best_k} (Score: {max(sil_scores):.3f})")