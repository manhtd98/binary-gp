import itertools
import random
import numpy as np
from helpers import protectedDiv, Sigmoid
from deap import gp
import operator

def create_pset():
    pset = gp.PrimitiveSet("MAIN", 71, "IN")
    pset.addPrimitive(np.add, 2, name="vadd")
    pset.addPrimitive(np.subtract, 2, name="vsub")
    pset.addPrimitive(np.multiply, 2, name="vmul")
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(np.negative, 1, name="vneg")
    pset.addPrimitive(np.cos, 1, name="vcos")
    pset.addPrimitive(np.sin, 1, name="vsin")


  
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    pset.addTerminal(False, "false")
    pset.addTerminal(True, "true")
    return pset
