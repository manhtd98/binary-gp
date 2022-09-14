import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from .pset import create_pset
from .toolbox import init_toolbox
from sklearn.linear_model import LogisticRegression
from .helpers import test_score, Sigmoid
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.multiclass import OneVsRestClassifier

random.seed(43)
np.random.seed(43)


def train_pipeline(x_train, y_train, num_attr):
    pset = create_pset(num_attr)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolboxes = []
    for i in range(y_train.shape[1]):
        print(f"Training tree of class: {i}")
        x_train = np.hstack([x_train, y_train[:, i].reshape(-1, 1)]).astype(np.float32)
        toolbox = init_toolbox(pset, x_train, num_attr)
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(
            pop, toolbox, 0.5, 0.1, 40, stats, halloffame=hof
        )
        toolboxes.append((pop, toolbox, hof))
    return toolboxes


def evaluation_pipeline(toolboxes, x_test, y_test):
    x_test = [x_test[:, i] for i in range(72)]
    preds = []
    for id, (pop, toolbox, hof) in enumerate(toolboxes):
        func = toolbox.compile(expr=hof[0])
        predict = Sigmoid(func(*x_test)).reshape(-1, 1)
        predict = np.int32(predict > 0.5)
        preds.append(predict)
    preds = np.hstack(preds)
    hamming_loss, f1, acc = test_score(y_test, preds)


if __name__ == "__main__":
    X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
    X_test, y_test, _, _ = load_dataset("emotions", "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()
    num_attr = X_train.shape[1]

    toolboxes = train_pipeline(X_train, y_train, num_attr)
    evaluation_pipeline(toolboxes, X_test, y_test)
