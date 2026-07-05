
import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split, GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

## Experiment

mlflow.set_experiment("BreastCancer_Tuning")

## Load Data

df = pd.read_csv("breast_cancer_preprocessing.csv")
X = df.drop("target", axis=1)
y = df["target"]

## Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

## Grid Search

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

## Train

with mlflow.start_run(run_name="GridSearchCV"):

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    prediction = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    ## Parameter

    mlflow.log_param(
        "best_n_estimators",
        grid.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "best_max_depth",
        grid.best_params_["max_depth"]
    )

    mlflow.log_param(
        "best_min_samples_split",
        grid.best_params_["min_samples_split"]
    )

    ## Metric

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    ## Save Artifact

    os.makedirs("artifacts", exist_ok=True)

    # Model
    joblib.dump(best_model, "artifacts/best_model.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction)

    pd.DataFrame(cm).to_csv(
        "artifacts/confusion_matrix.csv",
        index=False
    )

    # Classification Report
    report = classification_report(
        y_test,
        prediction,
        output_dict=True
    )

    pd.DataFrame(report).transpose().to_csv(
        "artifacts/classification_report.csv"
    )

    # Best Parameter
    pd.DataFrame(
        [grid.best_params_]
    ).to_csv(
        "artifacts/best_parameters.csv",
        index=False
    )

    ## Log Artifact

    mlflow.log_artifact("artifacts/best_model.pkl")
    mlflow.log_artifact("artifacts/confusion_matrix.csv")
    mlflow.log_artifact("artifacts/classification_report.csv")
    mlflow.log_artifact("artifacts/best_parameters.csv")

    ## Log Model

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model"
    )

    print("="*60)
    print("GRID SEARCH RESULT")
    print("="*60)

    print("Best Parameter")
    print(grid.best_params_)

    print()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
