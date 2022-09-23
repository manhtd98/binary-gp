from src.main import train_pipeline, evaluation_pipeline
from skmultilearn.dataset import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
if __name__ == "__main__":
    # X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
    # X_test, y_test, _, _ = load_dataset("emotions", "test")
    # X_train = X_train.toarray()
    # y_train = y_train.toarray()[:, 0].reshape(-1, 1)

    # X_test = X_test.toarray()
    # y_test = y_test.toarray()[:, 0].reshape(-1, 1)
    # num_attr = X_train.shape[1]
    df = pd.read_csv("spambase.csv",header = None)
    X_train = df.values[:, :57] 
    y_train = df.values[:, 57].reshape(-1, 1)
    num_attr = X_train.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

    toolboxes = train_pipeline(X_train, y_train, num_attr, 300)
    evaluation_pipeline(toolboxes, X_train, y_train, "train", num_attr)
    evaluation_pipeline(toolboxes, X_test, y_test, "test", num_attr)
