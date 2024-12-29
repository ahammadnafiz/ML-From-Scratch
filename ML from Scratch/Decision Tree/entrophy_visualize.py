import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    # Avoid log(0) errors
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def gini_impurity(p):
    return 2 * p * (1 - p)

# Generate points for the plot
x = np.linspace(0, 1, 1000)
y_entropy = entropy(x)
y_gini = gini_impurity(x)

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(x, y_entropy, 'b-', linewidth=2, label='Entropy')
plt.plot(x, y_gini, 'r--', linewidth=2, label='Gini Impurity')

plt.title('Entropy and Gini Impurity vs. Probability', fontsize=16)
plt.xlabel('Probability of Class 1', fontsize=14)
plt.ylabel('Impurity Measure', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(fontsize=12)

# Add annotations
plt.annotate('Maximum at p=0.5', xy=(0.5, 1), xytext=(0.6, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.tight_layout()
plt.show()
