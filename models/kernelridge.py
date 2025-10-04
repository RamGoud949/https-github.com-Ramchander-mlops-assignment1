import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Load dataset
df = pd.read_csv("data/BostonHousing.csv")

# Split features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Impute missing values with median
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Kernel Ridge model
model = KernelRidge(alpha=1.0, kernel='rbf')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Kernel Ridge RMSE: {rmse:.2f}")
