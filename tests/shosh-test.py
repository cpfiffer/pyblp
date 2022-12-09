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
T = 100
J = 5
F = 5
id_data = pyblp.data_to_dict(pyblp.build_id_data(T=T, J=J, F=F))
fake_k = np.random.lognormal(mean = 0, sigma=0.1, size = id_data['market_ids'].shape)
id_data['costs'] = fake_k
price_fixed_effects = np.random.lognormal(0, 1, J)
price_draws = np.random.lognormal(0,.1,T * J)
prices = price_draws + fake_k

prices[0:T] += np.random.lognormal(price_fixed_effects[0],.1,T)
prices[T:(2*T)] += np.random.lognormal(price_fixed_effects[1],.1,T)
prices[(2*T):(3*T)] += np.random.lognormal(price_fixed_effects[2],.1,T)

id_data['prices'] = prices

print(id_data)

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
    sigma=np.diag(np.array([-1, 0,-0.1,1,0.5,0.1])), # Nonlinear parameters
    xi=np.random.normal(0,1,T * J),
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

problem = sim_results.to_problem()
lower_bounds = np.zeros(sim_results._sigma.shape)
upper_bounds = np.zeros(sim_results._sigma.shape)
lower_bounds[:,:] = -np.inf
upper_bounds[:,:] = np.inf

for i in [3, 5]:
    lower_bounds[i,i] = 0

print("Going to try to solve it now, homie")
# with pyblp.parallel(8):
#     results = problem.solve(
#         sigma = 0.5 * sim_results._sigma,
#         pi = 0.5 * sim_results._pi,
#         beta = [None, 0.5 * sim_results._beta[1]],
#         finite_differences=True,
#         iteration=pyblp.Iteration('squarem', {'atol': 1e-11, 'max_evaluations':100_000}),
#         sigma_bounds = (lower_bounds, upper_bounds)
#     )
results = problem.solve(
    sigma = 0.1 * sim_results._sigma,
    beta = [None, 0.1 * sim_results._beta[1]],
    finite_differences=False,
    iteration=pyblp.Iteration('squarem', {'atol': 1e-11, 'max_evaluations':100_000}),
    sigma_bounds = (lower_bounds, upper_bounds)
)

# # print(np.c_[sim.beta, results.beta])

print(np.hstack([sim_results._beta, results.beta]))
print(np.vstack([np.diag(results.sigma.round(2)), np.diag(sim_results._sigma.round(2))]))
