
 import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (update the path to your file location)
data_path = "/facebook_live_sellers_in_thailand.csv"  # Update this path
df = pd.read_csv(data_path)

# Step 1: Inspect and Preprocess the Data
print("Dataset Head:")
print(df.head())

# Select numeric columns only (e.g., remove any non-numeric identifiers)
df = df.select_dtypes(include=[np.number])

# Handle any missing values by filling them with the column mean
df.fillna(df.mean(), inplace=True)

# Standardize the data to ensure each feature contributes equally
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

print("\nData after scaling (first 5 rows):")
print(data_scaled[:5])

### K-Means Clustering ###

# Step 1: Initial Clusters using Random Initialization
# We set n_init=1 to get a single initialization (like a random initial state)
kmeans_initial = KMeans(n_clusters=3, n_init=1, init="random", random_state=42)
kmeans_initial.fit(data_scaled)
initial_labels_kmeans = kmeans_initial.labels_

# Plot Initial Clusters for K-Means
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=initial_labels_kmeans, cmap="viridis", s=50)
plt.title("Initial Clusters (K-Means - Random Initialization)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.colorbar(label="Cluster Label")
plt.show()

# Step 2: Final Clusters with K-Means (Optimal)
kmeans_final = KMeans(n_clusters=3, random_state=42)
kmeans_final.fit(data_scaled)
final_labels_kmeans = kmeans_final.labels_

# Plot Final Clusters for K-Means
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=final_labels_kmeans, cmap="viridis", s=50)
plt.title("Final Clusters (K-Means - Optimized)")
plt.xlabel("Feature 1 (scaled)")
plt.ylabel("Feature 2 (scaled)")
plt.colorbar(label="Cluster Label")
plt.show()

# K-Means Evaluation Metrics
kmeans_silhouette = silhouette_score(data_scaled, final_labels_kmeans)
print("\nK-Means Clustering Results:")
print("Number of iterations (epochs):", kmeans_final.n_iter_)
print("Final error rate (inertia):", kmeans_final.inertia_)
print("Silhouette Score:", kmeans_silhouette)

