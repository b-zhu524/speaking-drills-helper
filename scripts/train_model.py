import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


def train_model():
    df = pd.read_csv('data/processed/data.csv')
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("Model training completed.")
    print(classification_report(y_test, clf.predict(X_test)))

    with open("models/model.pkl", "wb") as f:
        pickle.dump(clf, f)
