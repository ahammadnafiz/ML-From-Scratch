def gini_impurity(classes):

    from collections import Counter
    
    total_count = len(classes)
    class_counts = Counter(classes)
    
    # Calculate proportions of each class
    proportions = [count / total_count for count in class_counts.values()]
    
    # Calculate Gini impurity
    gini = 1 - sum(p ** 2 for p in proportions)
    
    return gini

# Example usage
classes = ['A'] * 6 + ['B'] * 2 + ['C'] * 2
gini_value = gini_impurity(classes)
print(f"Gini Impurity: {gini_value}")
