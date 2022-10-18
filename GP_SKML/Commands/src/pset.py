import itertools
import random
import numpy as np
from .helpers import protectedDiv, Sigmoid
from deap import gp
import operator


def create_pset(num_attr):
    # pset = gp.PrimitiveSet("MAIN", num_attr, "IN")
    # pset.addPrimitive(np.add, 2, name="vadd")
    # pset.addPrimitive(np.subtract, 2, name="vsub")
    # pset.addPrimitive(np.multiply, 2, name="vmul")
    # pset.addPrimitive(protectedDiv, 2)
    # pset.addPrimitive(np.negative, 1, name="vneg")
    # pset.addPrimitive(np.cos, 1, name="vcos")
    # pset.addPrimitive(np.sin, 1, name="vsin")
    # # defined a new primitive set for strongly typed GP
    # # terminals
    # pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    # # pset.addTerminal(False, bool)
    # # pset.addTerminal(True, bool)
    # return pset

    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, num_attr), bool, "IN")
    # boolean operators
    # pset.addPrimitive(operator.and_, [bool, bool], bool)
    # pset.addPrimitive(operator.or_, [bool, bool], bool)
    # pset.addPrimitive(operator.not_, [bool], bool)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    # logic operators
    # Define a new if-then-else function
    def if_then_else(input, output1, output2):
        if input:
            return output1
        else:
            return output2

    pset.addPrimitive(operator.lt, [float, float], bool)
    pset.addPrimitive(operator.eq, [float, float], bool)
    pset.addPrimitive(if_then_else, [bool, float, float], float)
    pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1), float)
    pset.addTerminal(False, bool)
    pset.addTerminal(True, bool)

    return pset
