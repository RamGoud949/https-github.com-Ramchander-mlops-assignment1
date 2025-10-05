import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load Boston Housing dataset manually
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
        'DIS','RAD','TAX','PTRATIO','B','LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # Target variable
    return df

# Load dataset
data = load_data()

# Features and target
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred)
print(f"Decision Tree MSE: {mse_dt}")

# Save model
joblib.dump(dt_model, "decision_tree_model.pkl")
print("Decision Tree model saved as decision_tree_model.pkl")
