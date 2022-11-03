import sys
sys.path.insert(0, 'C:/Users/cpfiffer/research/pyblp/')

import pyblp
import importlib
importlib.reload(pyblp)

import numpy as np
import pandas as pd

# SEt up some id data
id_data = pyblp.build_id_data(T=10, J=15, F=5)

# An Integration builds agent data
integration = pyblp.Integration('product', 9)
print(integration)

# Create a simulator
sim = pyblp.Simulation(
    product_formulations=(
        pyblp.Formulation('1 + prices + x'),
        pyblp.Formulation('0 + x'),
        pyblp.Formulation('0 + x + z')
    ),
    beta = [1, -2, 2], # Demand side parameters
    sigma=np.diag(np.ones(5)),
    gamma=[1,4], # Supply side parameters
    pi = np.zeros((5, 1)),
    product_data=id_data,
    agent_formulation=pyblp.Formulation("1"),
    integration=integration,
    seed=1,
    rc_types=['linear', 'linear', 'log', 'logit', 'log']
)

sim_results = sim.replace_endogenous()

problem = sim_results.to_problem()
results = problem.solve(
    sigma = 0.5 * sim_results._sigma,
    pi = 0.5 * sim_results._pi,
    beta = [None, 0.5 * sim_results._beta[1], None]
)


print(np.c_[sim.beta, results.beta])

