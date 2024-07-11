import numpy as np
import pandas as pd

# Number of data points
num_points = 200

# Generate random data
np.random.seed(0)
X1 = np.random.rand(num_points) * 100  # Random x1 values
X2 = np.random.rand(num_points) * 100  # Random x2 values
noise = np.random.randn(num_points) * 10  # Random noise
y = 3 * X1 + 2 * X2 + 5 + noise  # Linear relationship y = 3*X1 + 2*X2 + 5 + noise

# Create a DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})

# Save DataFrame to a CSV file
data.to_csv("Artificial-Intelligence/Machine-Learning/Linear-Regression/Data/random.csv", index=False)

