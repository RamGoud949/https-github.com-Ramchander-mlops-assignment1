import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import joblib

# Load Boston Housing dataset manually
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
df = pd.DataFrame(data, columns=feature_names)
df['MEDV'] = target

X = df.drop("MEDV", axis=1)
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kr_model = KernelRidge()
kr_model.fit(X_train, y_train)
y_pred_kr = kr_model.predict(X_test)
mse_kr = mean_squared_error(y_test, y_pred_kr)
print(f"Kernel Ridge MSE: {mse_kr}")
joblib.dump(kr_model, "kernel_ridge_model.pkl")
print("Kernel Ridge model saved as kernel_ridge_model.pkl")
