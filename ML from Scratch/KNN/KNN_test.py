from KNN_from_Scratch import KNN
import numpy as np 

def test_knn():
    # Create a KNN instance with k=3
    knn = KNN(k_neighbors=3)
    
    # Training data (X_train) and labels (y_train)
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    y_train = np.array([0, 0, 1, 1, 1])
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    # Test prediction
    X_test = np.array([
        [2, 2],  # Expect 0
        [6, 6]   # Expect 1
    ])
    
    predictions = knn.predict(X_test)
    
    # Validate predictions
    assert np.allclose(predictions, [0, 1]), "Prediction failed"
    
    print("All test cases passed!")

test_knn()
