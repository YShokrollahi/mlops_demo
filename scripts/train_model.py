import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model():
    # Load dataset
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set MLflow tracking URI to a directory within the Docker container
    mlflow.set_tracking_uri("file:///app/mlruns")

    # Start an MLflow run
    with mlflow.start_run():
        # Train a RandomForest model
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "model")
        
        # Save the model to disk
        mlflow.sklearn.save_model(clf, "models/model.pkl")

if __name__ == "__main__":
    train_model()
