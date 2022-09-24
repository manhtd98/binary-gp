
from src.main import train_pipeline, evaluation_pipeline
from skmultilearn.dataset import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from utils import get_random_state
get_random_state(42)

if __name__ == "__main__":
    df = pd.read_csv("../data/mammographic_masses.data", header=None)#.astype(float)
    df = df.replace("?", 0)
    target_name = df.columns[-1]
    # print(df.iloc[0], df[target_name].unique())
    X_train = df.values[:, :5].astype(int)
    y_train = df.values[:, 5].astype(int).reshape(-1, 1)
    num_attr = X_train.shape[1]
    sample = int(X_train.shape[0]*0.2)
    # print(y_train[0])
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    toolboxes = train_pipeline(X_train, y_train, num_attr, 300, sample)
    print(X_train.shape, y_train.shape)
    evaluation_pipeline(toolboxes, X_train, y_train, "train", num_attr)
    # evaluation_pipeline(toolboxes, X_test, y_test, "test", num_attr)
