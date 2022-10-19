import random
from time import time
import numpy as np
from deap import algorithms, base, creator, tools, gp
from skmultilearn.dataset import load_dataset
import multiprocessing
from skmultilearn.problem_transform import BinaryRelevance
from src.helpers import test_score
from src.main import GPClasification
import os 
import json 
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from src.pset import create_pset
from src.helpers import test_score
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import f1_score

class GPClasification:
    def __init__(
        self, population=512, sample=0.2, epoch=50, alpha=0.8, beta=0.2, hallofframe=1
    ):
        self.population = population
        self.sample = sample
        self.epoch = epoch
        self.toolbox = None
        self.hoft = None
        self.alpha = alpha
        self.beta = beta
        self.hallofframe = hallofframe

    def fit(self, x_train, y_train):
        num_attr = x_train.shape[1]
        self.num_attr = num_attr
        x_train = np.hstack([x_train, y_train.reshape(-1, 1)])
        pset = create_pset(num_attr)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        def evalMultiplexer(individual, sample=x_train):
            # Transform the tree expression in a callable function
            func = toolbox.compile(expr=individual)
            # spam_samp = random.sample(range(x_train.shape[0]), sample)
            # Evaluate the sum of correctly identified
            preds = np.array([func(*ip) for ip in x_train[:,:num_attr]])
            preds = np.where(preds > 0, 1, 0)
            result = -f1_score(preds, x_train[:, num_attr])  # weighted
            return (result,)

        toolbox.register("evaluate", evalMultiplexer)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        # Process Pool of 4 workers
        pool = multiprocessing.Pool(processes=8)
        toolbox.register("map", pool.map)

        pop = toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.hallofframe)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(
            pop, toolbox, self.alpha, self.beta, self.epoch, stats, halloffame=hof
        )
        self.toolbox = toolbox
        self.hoft = hof[0]

    def predict(self, x_test):
        func = self.toolbox.compile(expr=self.hoft)
        pred = np.array([func(*val) for val in x_test])
        predict = np.where(pred > 0, 1, 0).reshape(-1, 1)
        return predict


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1]
    out_dir = sys.argv[2]
    run_idx = int(sys.argv[3])

    random.seed(1617*run_idx)
    np.random.seed(1617*run_idx)

    file_path = out_dir+'%d.txt' % run_idx
    if not os.path.exist(out_dir):
        os.makedirs(out_dir)
    X_train, y_train, feature_names, label_names = load_dataset(dataset, "train")
    X_test, y_test, _, _ = load_dataset(dataset, "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()
    num_attr = X_train.shape[1]

    classifier = GPClasification()
    
    classifier = BinaryRelevance(classifier)
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time
    prediction = classifier.predict(X_train)
    res_train = test_score(y_train, prediction)
    res_test = test_score(y_test, prediction)
    res = {
            "result_train": res_train,
            "result_test": res_test,
            "training sample": X_train.shape[0],
            "test sample": X_test.shape[0],
            "training time": train_time,
      }
    with open(file_path, "w") as f:
        json.dump(res, f)


