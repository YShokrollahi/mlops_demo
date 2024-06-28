# scripts/prepare_data.py
from sklearn.datasets import load_iris
import pandas as pd

def prepare_data():
    # Load Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save dataset to CSV
    df.to_csv('data/iris.csv', index=False)

if __name__ == "__main__":
    prepare_data()
