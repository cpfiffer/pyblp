import sys
sys.path.insert(0, 'C:/Users/cpfiffer/research/pyblp/')

import pyblp
import importlib
importlib.reload(pyblp)

import numpy as np
import pandas as pd

# SEt up some id data
# T is markets
# J is products per market
# F is firms across all markets
id_data = pyblp.data_to_dict(pyblp.build_id_data(T=15, J=20, F=10))
fake_k = np.random.lognormal(mean = -1, size = id_data['market_ids'].shape)
id_data['k'] = fake_k

# An Integration builds agent data
integration = pyblp.Integration('product', 9)

# Create a simulator
sim = pyblp.Simulation(
    product_formulations=(
        pyblp.Formulation('1 + x + prices + k'),
        pyblp.Formulation('0 + x'),
        pyblp.Formulation('0 + x + z')
    ),
    beta = [1, 2, -2, 0], # Demand side parameters
    sigma=np.diag(np.array([1,0,0,0,0])),
    gamma=[1,4], # Supply side parameters
    pi = np.array([[0, 1.0, 1.5, 0.2, 0.3]]).T,
    product_data=id_data,
    agent_formulation=pyblp.Formulation("1"),
    integration=integration,
    seed=1,
    rc_types=['linear', 'linear', 'log', 'logit', 'log']
)

sim_results = sim.replace_endogenous()

problem = sim_results.to_problem()

print("Going to try to solve it now, homie")
results = problem.solve(
    sigma = 0.5 * sim_results._sigma,
    pi = 0.5 * sim_results._pi,
    beta = [None, None, 0.5 * sim_results._beta[2], None],
    finite_differences=True
)


# print(np.c_[sim.beta, results.beta])

print(np.hstack([sim_results._beta, results.beta]))

np.hstack([results.pi.round(2), sim_results._pi.round(2)])