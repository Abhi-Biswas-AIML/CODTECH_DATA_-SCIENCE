"""model_training.py

End-to-end example using the Iris dataset.

- Data collection: load from scikit-learn
- Preprocessing + model training
- Save trained model to disk

Run:
    python model_training.py
"""

import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_DIR, "model.joblib")


def main():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("[Task 3] Accuracy on test set:", acc)
    print("[Task 3] Classification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(pipe, MODEL_PATH)
    print("[Task 3] Saved trained model to:", MODEL_PATH)


if __name__ == "__main__":
    main()
