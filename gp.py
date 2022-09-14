from src.main import train_pipeline, evaluation_pipeline
from skmultilearn.dataset import load_dataset


if __name__ == "__main__":
    # samples = np.linspace(-1, 1, 10)
    # values = Sigmoid(samples ** 2 + samples ** 3 + samples ** 2 + samples)
    # values = np.float32(values > 0.5)

    X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
    X_test, y_test, _, _ = load_dataset("emotions", "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()[:, 2].reshape(-1, 1)

    X_test = X_test.toarray()
    y_test = y_test.toarray()[:, 2].reshape(-1, 1)
    num_attr = X_train.shape[1]

    toolboxes = train_pipeline(X_train, y_train, num_attr)
    evaluation_pipeline(toolboxes, X_test, y_test)
