import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from .helpers import Sigmoid


def init_toolbox(pset, samples):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalMultiplexer(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        spam_samp = random.sample(range(samples.shape[0]), 50)
        # Evaluate the sum of correctly identified
        inputs = [samples[spam_samp, i] for i in range(72)]
        outputs = samples[spam_samp, 72]
        result = np.sum((func(*inputs) - outputs) ** 2)
        return (result,)

    toolbox.register("evaluate", evalMultiplexer)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox
