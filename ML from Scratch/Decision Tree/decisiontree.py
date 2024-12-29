import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, probs=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes
        self.probs = probs  # Class probabilities for leaf nodes

class DecisionTreeClassifier:
    def __init__(self, max_depth=100, min_samples_split=2, criterion='entropy', max_features=None, min_gain=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion  # 'entropy' or 'gini'
        self.max_features = max_features  # Subset of features for split
        self.min_gain = min_gain  # Minimum information gain
        
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.feature_names = [f'X{i}' for i in range(self.n_features)]
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            leaf_value = self._most_common_label(y)
            probs = self._class_probabilities(y)
            return Node(value=leaf_value, probs=probs)
        
        # Select subset of features
        features = np.random.choice(n_features, self.max_features or n_features, replace=False)
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._best_split(X, y, features)
        
        # Stop if the gain is too low
        if best_gain < self.min_gain:
            leaf_value = self._most_common_label(y)
            probs = self._class_probabilities(y)
            return Node(value=leaf_value, probs=probs)
        
        # Split the data
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, features):
        best_gain = -1
        split_idx, split_threshold = None, None
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold
                    
        return split_idx, split_threshold, best_gain
    
    def _information_gain(self, y, X_column, threshold):
        parent_impurity = self._impurity(y)
        
        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted avg. impurity of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        child_impurity = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate Information Gain
        return parent_impurity - child_impurity
    
    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs
    
    def _impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p**2 for p in ps])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _class_probabilities(self, y):
        counter = Counter(y)
        total = len(y)
        return {k: v / total for k, v in counter.items()}
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def plot_tree(self, node=None, depth=0, pos=None, ax=None, feature_names=None, vertical_spacing=1.2, horizontal_spacing=1.5):
        if node is None:
            node = self.root
        
        if pos is None:
            pos = (0, 0)  # root position
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_axis_off()

        x_offset = horizontal_spacing ** (self.max_depth - depth)
        y_offset = vertical_spacing

        # Draw the current node
        if node.value is not None:  # Leaf node
            # Use a color map for class probabilities
            self._plot_leaf_node(node, pos, ax)
        else:
            # Internal node with decision condition
            ax.text(pos[0], pos[1], f"{feature_names[node.feature]} <= {node.threshold:.2f}",
                    ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.3'))

            # Recursively plot left and right children with connections
            new_pos_left = (pos[0] - x_offset, pos[1] - y_offset)
            new_pos_right = (pos[0] + x_offset, pos[1] - y_offset)

            # Draw connection to the left child
            ax.plot([pos[0], new_pos_left[0]], [pos[1], new_pos_left[1]], 'k-', lw=1.5)
            self.plot_tree(node.left, depth + 1, new_pos_left, ax, feature_names)

            # Draw connection to the right child
            ax.plot([pos[0], new_pos_right[0]], [pos[1], new_pos_right[1]], 'k-', lw=1.5)
            self.plot_tree(node.right, depth + 1, new_pos_right, ax, feature_names)
        
        if ax is None:
            plt.show()

    def _plot_leaf_node(self, node, pos, ax):
        # Map class probabilities to colors
        prob_text = "\n".join([f"{k}: {v:.2f}" for k, v in node.probs.items()])
        cmap = plt.get_cmap('coolwarm')  # color map for class probabilities
        color = self._get_node_color(node.probs, cmap)

        ax.text(pos[0], pos[1], f"Class: {node.value}\n{prob_text}",
                ha='center', va='center', fontsize=9,
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3'))

    def _get_node_color(self, probs, cmap):
        class_ids = list(probs.keys())
        if len(class_ids) == 1:
            return cmap(1.0)  # Pure color for single class
        else:
            avg_prob = np.mean(list(probs.values()))  # Average probability
            norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize the probabilities
            return cmap(norm(avg_prob))
