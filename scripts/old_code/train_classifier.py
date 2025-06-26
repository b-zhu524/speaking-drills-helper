from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


def train_classifier():
    # Load data
    data = np.load("embeddings/embeddings.npz")
    X, y = data["X"], data["y"]


    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(X_train, y_train)

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

