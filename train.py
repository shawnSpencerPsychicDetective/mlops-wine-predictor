# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import sys
import numpy as np

# 1. Load Data
# (Using a standard online dataset for simplicity)
csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(csv_url, sep=";")

# 2. Prepare Data
train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# 3. Set Hyperparameters (Simulating an experiment)
# We take these from command line args, or default to specific values
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

# ---------------- MLOPS SECTION ----------------
# Start an MLflow run to track this specific experiment
with mlflow.start_run():
    # A. Log the Parameters (So you know what configuration you used)
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    # B. Train the Model
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    # C. Evaluate
    predicted_qualities = lr.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
    mae = mean_absolute_error(test_y, predicted_qualities)
    r2 = r2_score(test_y, predicted_qualities)

    # D. Log the Metrics (So you can compare performance later)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    # E. Log the Model itself (Save the artifact)
    mlflow.sklearn.log_model(lr, "model")
    
    print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
    print(f"  RMSE: {rmse}")
    print(f"  R2: {r2}")