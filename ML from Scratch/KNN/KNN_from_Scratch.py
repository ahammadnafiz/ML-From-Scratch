import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k_neighbors :int = 3):
        self.k_neighbors = k_neighbors

    def fit(self, X_train, y_pred):
        self.X_train = X_train
        self.y_pred = y_pred

    def get_distance(self, a, b):
        return np.sum((a - b) ** 2) ** 0.5

    def get_k_neighbors(self, X_test_single):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((i, self.get_distance(self.X_train[i], X_test_single)))
        
        distances.sort(key=lambda x: x[1])
        return distances[:self.k_neighbors]

    def predict(self, X_test):
        predictions = []
        
        for idx in range(len(X_test)):
            X_test_single = X_test[idx]
            k_neighbors_list = self.get_k_neighbors(X_test_single)
            k_y_value = [self.y_pred[item[0]] for item in k_neighbors_list]
            # prediction = np.mean(k_y_value) # for regression model
            # predictions.append(prediction)

            majority_vote = Counter(k_y_value).most_common(1)[0][0]

            # Use majority voting to get the predicted class
            predictions.append(majority_vote)

           
        return predictions
