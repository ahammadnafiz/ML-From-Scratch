import numpy as np
import matplotlib.pyplot as plt

def gini_impurity(p):
    return 2 * p * (1 - p)

# Generate points for the plot
x = np.linspace(0, 1, 1000)
y = gini_impurity(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Gini Impurity vs. Probability', fontsize=16)
plt.xlabel('Probability of Class 1', fontsize=14)
plt.ylabel('Gini Impurity', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)

# Add annotations
plt.annotate('Maximum impurity\nat p=0.5', xy=(0.5, 0.5), xytext=(0.6, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

plt.tight_layout()
plt.show()
