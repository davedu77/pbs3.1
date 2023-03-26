# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import lognorm

# Define parameters
beta = 0.95
r = 0.04
mu = 0.5
rho = 0.8
sigma_epsilon = 0.1
sigma = 2

# Define the utility function
def u(c, sigma):
    if c < 1e-8:
        c = 1e-8
    return c**(1 - sigma) / (1 - sigma)

# define the number of grids for assets and income
a_grid_size = 100
a_min = 0
a_max = 10
a_grid = np.linspace(a_min, a_max, a_grid_size)

y_grid_size = 100
y_min = 0.1
y_max = 2
y_grid = np.linspace(y_min, y_max, y_grid_size)

# define the value function and the consumption policy function
C = np.zeros((a_grid_size, y_grid_size))

# Define the expectation function
def lognorm_expectation(y, mu, rho, sigma_epsilon):
    return lognorm.pdf(y, s=sigma_epsilon, scale=np.exp(mu + rho * np.log(y)))

# define the Policy function iteration
max_iter = 1000
tol = 1e-6
diff = 1


# %%
# define the Policy function iteration
max_iter = 1000
tol = 1e-6
diff = 1

for iter in range(max_iter):
    C_new = np.zeros_like(C)
    
    for i, a in enumerate(a_grid):
        for j, y in enumerate(y_grid):
            
            def bellman_operator(consumption_share):
                c = consumption_share * ((1 + r) * a + y)
                a_prime = (1 + r) * a + y - c
                V_prime_interp = interp1d(a_grid, [u(C_ij, sigma) for C_ij in C[:, j]], kind='linear', fill_value='extrapolate')
                expectation = np.mean([lognorm_expectation(y_next, mu + rho * np.log(y), rho, sigma_epsilon) * V_prime_interp(a_prime) for y_next in y_grid])
                return - (u(c, sigma) + beta * expectation)
            
            bounds = (1e-8, 1)
            res = minimize_scalar(bellman_operator, bounds=bounds)
            C_new[i, j] = res.x * ((1 + r) * a + y)
    
    diff = np.max(np.abs(C_new - C))
    C = C_new.copy()

    if diff < tol:
        break

# %%

# Plot the consumption policy function
plt.figure()
for j, y in enumerate(y_grid):
    plt.plot(a_grid, C[:, j], label=f"y = {y:.2f}")
plt.xlabel("Assets")
plt.ylabel("Consumption")
plt.legend()
plt.title("Consumption Policy Function")
plt.show()

# Plot consumption as a share of available resources
plt.figure()
for j, y in enumerate(y_grid):
    plt.plot(a_grid[1:], C[1:, j] / ((1 + r) * a_grid[1:] + y), label=f"y = {y:.2f}")
plt.xlabel("Assets")
plt.ylabel("Consumption Share")
plt.legend()
plt.title("Consumption as a Share of Available Resources")
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# define the Parameters
beta = 0.95
r = 0.04
mu = 0.5
rho = 0.8
sigma_epsilon = 0.1
sigma = 2

# define the Utility function
def u(c, sigma):
    return (c**(1 - sigma)) / (1 - sigma)

# define the Asset grid
a_min = 0
a_max = 30
n_a = 100
a_grid = np.linspace(a_min, a_max, n_a)

# define the Income grid
n_y = 100
y_min = 0.1
y_max = 3
y_grid = np.linspace(y_min, y_max, n_y)


# Simulation part
np.random.seed(42)  # Set random seed for reproducibility

n_simulations = 5
n_periods = 100

# Initial conditions
a0 = a_grid[len(a_grid) // 2]
y0 = y_grid[len(y_grid) // 2]

# Initialize arrays for assets, income, and consumption
assets = np.zeros((n_simulations, n_periods))
income = np.zeros_like(assets)
consumption = np.zeros_like(assets)

# %%
def value_iteration(a_grid, y_grid, u, beta, r, n_a, n_y):
    # define the value function
    V = np.zeros((n_a, n_y))
    V_new = np.zeros_like(V)

    # define the policy functions
    policy_c = np.zeros_like(V)
    policy_a_prime = np.zeros_like(V)

    # Tolerance and maximum iterations
    tol = 1e-6
    max_iter = 1000

    for it in range(max_iter):
        for ia, a in enumerate(a_grid):
            for iy, y in enumerate(y_grid):
                objective = lambda c: -(u(c, sigma) + beta * np.interp(a + (1 + r) * a - c, a_grid, V[:, iy]))
                result = minimize_scalar(objective, bounds=(1e-6, (1 + r) * a + y), method='bounded')
                c_star = result.x
                a_prime_star = (1 + r) * a + y - c_star

                # Update the value and policy functions
                V_new[ia, iy] = -result.fun
                policy_c[ia, iy] = c_star
                policy_a_prime[ia, iy] = a_prime_star

        # Check for convergence
        if np.max(np.abs(V - V_new)) < tol:
            break

        V = np.copy(V_new)

    return V, policy_c, policy_a_prime

V, policy_c, policy_a_prime = value_iteration(a_grid, y_grid, u, beta, r, n_a, n_y)


# %%
# Simulation part
np.random.seed(42)  # Set random seed for reproducibility

n_simulations = 5
n_periods = 100

# Initial conditions
a0 = a_grid[len(a_grid) // 2]
y0 = y_grid[len(y_grid) // 2]

# Initialize arrays for assets, income, and consumption
assets = np.zeros((n_simulations, n_periods))
income = np.zeros_like(assets)
consumption = np.zeros_like(assets)

# Simulate sample paths
for sim in range(n_simulations):
    a = a0
    y = y0
    for t in range(n_periods):
        ia = np.searchsorted(a_grid, a, side='right') - 1
        iy = np.searchsorted(y_grid, y, side='right') - 1

        c = policy_c[ia, iy]
        a_next = policy_a_prime[ia, iy]

        assets[sim, t]
        assets[sim, t] = a
        income[sim, t] = y
        consumption[sim, t] = c

        a = a_next
        y = np.exp(mu + rho * np.log(y) + sigma_epsilon * np.random.randn())

# %%

# Plot assets, income, and consumption
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ['Assets', 'Income', 'Consumption']
data = [assets, income, consumption]

for i, ax in enumerate(axes):
    for sim in range(n_simulations):
        ax.plot(data[i][sim, :], lw=1, alpha=0.7)
    ax.set_title(titles[i])
    ax.set_xlabel('Periods')

plt.tight_layout()
plt.show()

# %%
import numpy as np

n_workers = 10000
n_periods = 100

# Initialize the worker states
initial_assets = np.random.uniform(a_min, a_max, n_workers)
initial_incomes = np.random.uniform(y_min, y_max, n_workers)

# Function to simulate the workers
def simulate_workers(initial_assets, initial_incomes, policy_c, policy_a_prime, n_workers, n_periods):
    assets = np.zeros((n_workers, n_periods))
    incomes = np.zeros_like(assets)
    assets[:, 0] = initial_assets
    incomes[:, 0] = initial_incomes

    for t in range(1, n_periods):
        for i in range(n_workers):
            ia = np.searchsorted(a_grid, assets[i, t - 1], side='right') - 1
            iy = np.searchsorted(y_grid, incomes[i, t - 1], side='right') - 1
            assets[i, t] = policy_a_prime[ia, iy]
            incomes[i, t] = np.exp(mu + rho * np.log(incomes[i, t - 1]) + np.random.normal(0, sigma_epsilon))
    
    return assets, incomes

# Simulate the population for 100 periods
assets, incomes = simulate_workers(initial_assets, initial_incomes, policy_c, policy_a_prime, n_workers, n_periods)

# Plot the distribution of assets after 100 periods
plt.figure()
plt.hist(assets[:, -1], bins=50, density=True)
plt.xlabel('Assets')
plt.ylabel('Density')
plt.title('Asset Distribution After 100 Periods')
plt.show()

# %%

# Simulate the population for another 100 periods
assets2, incomes2 = simulate_workers(assets[:, -1], incomes[:, -1], policy_c, policy_a_prime, n_workers, n_periods)

# Plot the distribution of assets after another 100 periods
plt.figure()
plt.hist(assets2[:, -1], bins=50, density=True)
plt.xlabel('Assets')
plt.ylabel('Density')
plt.title('Asset Distribution After Another 100 Periods')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# define the Parameters
beta = 0.95
r = 0.04
mu = 0.5
rho = 0.8
sigma_epsilon = 0.1
sigma = 2

# define the Utility function
def u(c, sigma):
    return (c**(1 - sigma)) / (1 - sigma)

# define the Asset grid
a_min = 0
a_max = 30
n_a = 100
a_grid = np.linspace(a_min, a_max, n_a)

# define the Income grid
n_y = 100
y_min = 0.1
y_max = 3
y_grid = np.linspace(y_min, y_max, n_y)

# Value iteration function
# ... (same as in the previous code snippet)

V, policy_c, policy_a_prime = value_iteration(a_grid, y_grid, u, beta, r, n_a, n_y)

# Simulation code
n_workers = 10000
n_periods = 100

# define the worker states
initial_assets = np.random.uniform(a_min, a_max, n_workers)
initial_incomes = np.random.uniform(y_min, y_max, n_workers)


# %%

# define the Function to simulate the workers
def simulate_workers(initial_assets, initial_incomes, policy_c, policy_a_prime, n_workers, n_periods):
    assets = np.zeros((n_workers, n_periods))
    incomes = np.zeros_like(assets)
    assets[:, 0] = initial_assets
    incomes[:, 0] = initial_incomes

    for t in range(1, n_periods):
        for i in range(n_workers):
            ia = np.searchsorted(a_grid, assets[i, t - 1], side='right') - 1
            iy = np.searchsorted(y_grid, incomes[i, t - 1], side='right') - 1
            assets[i, t] = policy_a_prime[ia, iy]
            incomes[i, t] = np.random.normal(mu, sigma_epsilon)
    
    return assets, incomes

assets, incomes = simulate_workers(initial_assets, initial_incomes, policy_c, policy_a_prime, n_workers, n_periods)

# %%

# Plot histogram of the distribution of assets after 100 periods
plt.hist(assets[:, -1], bins=30)
plt.xlabel('Assets')
plt.ylabel('Frequency')
plt.title('Distribution of Assets After 100 Periods')
plt.show()

# %%
# Simulate another 100 periods
assets2, incomes2 = simulate_workers(assets[:, -1], incomes[:, -1], policy_c, policy_a_prime, n_workers, n_periods)

# Plot histogram of the distribution of assets after 200 periods
plt.hist(assets2[:, -1], bins=30)
plt.xlabel('Assets')
plt.ylabel('Frequency')
plt.title('Distribution of Assets After 200 Periods')
plt.show()


