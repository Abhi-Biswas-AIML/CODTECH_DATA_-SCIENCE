"""task1_etl_pipeline.py

Simple ETL (Extract-Transform-Load) pipeline using pandas and scikit-learn.

- Extract: Read raw CSV data
- Transform: Handle missing values, encode categoricals, scale numerics
- Load: Train a simple model and save the processed dataset + model

Run:
    python task1_etl_pipeline.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

THIS_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(THIS_DIR, "sample_customer_data.csv")
OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Extract step: load raw data from CSV."""
    df = pd.read_csv(path)
    return df


def build_preprocessing_pipeline(numeric_features, categorical_features):
    """Create a ColumnTransformer pipeline for numeric + categorical features."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def build_full_pipeline(preprocessor):
    """Attach a classifier on top of preprocessing to form a full pipeline."""
    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )
    return pipe


def main():
    print("[Task 1] Starting ETL pipeline...")
    df = load_data(DATA_PATH)

    # Split features and target
    X = df.drop("bought", axis=1)
    y = df["bought"]

    numeric_features = ["age", "income"]
    categorical_features = ["gender", "city"]

    preprocessor = build_preprocessing_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    model = build_full_pipeline(preprocessor)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit model (Load step in a loose sense, persisting downstream artifact)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"[Task 1] Train accuracy: {train_score:.3f}")
    print(f"[Task 1] Test accuracy:  {test_score:.3f}")

    # Save processed datasets and model
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    model_path = os.path.join(OUTPUT_DIR, "customer_churn_model.joblib")
    joblib.dump(model, model_path)
    print(f"[Task 1] Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()
