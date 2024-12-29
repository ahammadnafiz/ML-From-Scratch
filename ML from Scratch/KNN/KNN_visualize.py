import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from KNN_from_Scratch import KNN
# Assuming you have already defined the KNN class as provided

# Generate a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
knn = KNN(k_neighbors=3)
knn.fit(X_train, y_train)

# Create a meshgrid to visualize the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict for each point in the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array(Z).reshape(xx.shape)

# Plot the decision boundary and scatter plot of the data
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='black')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, alpha=0.6, edgecolor='black', marker='s')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classification (k={})'.format(knn.k_neighbors))
plt.colorbar()

# Add a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Train',
                              markerfacecolor='gray', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='Test',
                              markerfacecolor='gray', markersize=10)]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.show()
