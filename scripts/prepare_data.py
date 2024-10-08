# scripts/prepare_data.py
from sklearn.datasets import load_iris
import pandas as pd
#here i can edit data preprocessing 
# for i in range(pd):
#     return False
#you can do it recursively as well
#here i can edit data preprocessing 
def prepare_data():
    # Load Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save dataset to CSV
    df.to_csv('data/iris.csv', index=False)
#here i can edit data preprocessing 
# for i in range(pd):
#     return False
#here i can edit data preprocessing 
if __name__ == "__main__":
    prepare_data()
