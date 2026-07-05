
import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

## MLFlow

mlflow.set_experiment("BreastCancer_Basic")
mlflow.sklearn.autolog()

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

## Train

with mlflow.start_run():
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=5
    )
    model.fit(
        X_train, y_train
    )
    prediction = model.predict(
        X_test
    )
    accuracy = accuracy_score(
        y_test,prediction
    )
    precision = precision_score(
        y_test, prediction
    )
    recall = recall_score(
        y_test, prediction
    )
    f1 = f1_score(

        y_test, prediction
    )


    print("="*60)
    print("Model Performance")
    print("="*60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print()

    print(classification_report(
        y_test, prediction
    ))

    # Save confusion matrix

    cm = confusion_matrix(
        y_test, prediction

    )

    os.makedirs(
        "artifacts",exist_ok=True
    )

    pd.DataFrame(cm).to_csv(
        "artifacts/confusion_matrix.csv",
        index=False
    )

    joblib.dump(
        model,
        "artifacts/model.pkl"
    )

    mlflow.log_artifact(
        "artifacts/confusion_matrix.csv"
    )

    mlflow.log_artifact(
        "artifacts/model.pkl"
    )

print("\nTraining completed.")
