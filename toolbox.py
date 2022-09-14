import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
from helpers import Sigmoid


def init_toolbox(pset, samples, values):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalSpambase(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Randomly sample 400 mails in the spam database
        spam_samp = random.sample(range(samples.shape[0]), 50)
        # Evaluate the sum of correctly identified mail as spam
        result = sum(bool(func(*samples[mail, :71])) is bool(samples[mail, 71]) for mail in spam_samp)
        return result,

    toolbox.register("evaluate", evalSpambase)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox
