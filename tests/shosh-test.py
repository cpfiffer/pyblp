import sys
sys.path.insert(0, 'C:/Users/cpfiffer/research/pyblp/')
sys.path.insert(0, '/home/cameron/research/pyblp/')

import pyblp
import importlib
importlib.reload(pyblp)

import numpy as np
import pandas as pd

# SEt up some id data
# T is markets
# J is products per market
# F is firms across all markets
id_data = pyblp.data_to_dict(pyblp.build_id_data(T=50, J=3, F=3))
fake_k = np.random.lognormal(mean = 0, size = id_data['market_ids'].shape)
id_data['costs'] = fake_k
price_fixed_effects = np.random.lognormal(0,1,3)
price_draws = np.random.lognormal(0,.1,150)
prices = price_draws + fake_k
id_data['prices'] = prices

print(id_data)
prices[0:50] = np.random.lognormal(price_fixed_effects[0],.1,50)
prices[50:100] = np.random.lognormal(price_fixed_effects[1],.1,50)
prices[100:150] = np.random.lognormal(price_fixed_effects[2],.1,50)

# An Integration builds agent data
integration = pyblp.Integration('product', 6)

# Create a simulator
# Extra columns of X2 are [mu_alpha, sigma_alpha, mu_eta, sigma_eta]
# These parameters are stored in pi
sim = pyblp.Simulation(
    product_formulations=(
        pyblp.Formulation('1 + x'),
        pyblp.Formulation('0 + prices + costs')
        # pyblp.Formulation('0 + x + z')
    ),
    beta = [1, 2], # Demand side parameters
    sigma=np.diag(np.array([-1, 0,0,0,0,0])), # Nonlinear parameters
    xi=np.random.normal(0,1,150),
    # gamma=[1,4], # Supply side parameters
    # pi = np.array([[0, 1.0, 1.5, 0.2, 0.3, 0.0]]).T,
    # pi = np.zeros((6,1)),
    product_data=id_data,
    costs_type='linear',
    # agent_formulation=pyblp.Formulation("1"),
    integration=integration,
    seed=1,
    rc_types=['linear'] * 6#['linear', 'linear', 'linear', 'linear', 'linear', 'linear']
)

print(prices[0:3])
sim_results = sim.replace_endogenous(prices=prices, costs=id_data['costs'], iteration=pyblp.Iteration('return'))

# problem = sim.to_problem()

# print("Going to try to solve it now, homie")
# results = problem.solve(
#     sigma = 0.5 * sim_results._sigma,
#     pi = 0.5 * sim_results._pi,
#     beta = [None, None, 0.5 * sim_results._beta[2], None],
#     finite_differences=False
# )


# # print(np.c_[sim.beta, results.beta])

# print(np.hstack([sim_results._beta, results.beta]))

# np.hstack([results.pi.round(2), sim_results._pi.round(2)])
