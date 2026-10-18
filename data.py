import numpy as np

def generate_data():
    np.set_printoptions(precision=2)
    np.random.seed(42)
    noise = np.random.randn(50, 1)
    ground_truth = np.array([[2], [1]])
    
    x = np.random.uniform(low=0, high=10, size=(50,1))
    A = np.hstack([x, np.ones((50,1))])
    y = A @ ground_truth + noise

    return x, A, y
