import pandas as pd
import mlflow
import mlflow.sklearn
import argparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def main(data_path):
    
    mlflow.set_experiment("churn-ci")
    
    with mlflow.start_run():
    
        # membaca dataset hasil preprocessing
        df = pd.read_csv(data_path)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
           X, y,
           test_size=0.2,
           random_state=42,
           stratify=y
      )

    # grid search untuk tuning hyperparameter
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear"]
    }

    base_model = LogisticRegression(max_iter=1000)

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # prediksi
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # logging ke MLflow (file-based)
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("true_negative", tn)
    mlflow.log_metric("false_positive", fp)
    mlflow.log_metric("false_negative", fn)
    mlflow.log_metric("true_positive", tp)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        input_example=X_test.iloc[:5]
    )

    print("Training selesai")
    print("Accuracy:", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="churn_preprocessing.csv"
    )
    args = parser.parse_args()

    main(args.data_path)
