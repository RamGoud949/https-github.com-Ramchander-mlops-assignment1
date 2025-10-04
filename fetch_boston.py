from sklearn.datasets import fetch_openml
import pandas as pd
import os

# Create 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Fetch Boston dataset from OpenML
boston = fetch_openml(name="house_prices", as_frame=True)
df = pd.concat([boston.data, boston.target.astype(float)], axis=1)

# Save locally for offline use
df.to_csv("data/BostonHousing.csv", index=False)
print("Boston dataset saved locally as data/BostonHousing.csv")
