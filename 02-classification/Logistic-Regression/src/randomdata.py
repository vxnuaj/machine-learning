import pandas as pd
import numpy as np

m = 200

X1 = np.random.normal(10,10,m)
X2 = np.random.normal(10,10,m)
y = (X1 + X2 > 0).astype(int)

df = pd.DataFrame({"Feature 1": X1, "Feature 2": X2, "Label": y})

df.to_csv("randomtest.csv", index = False)