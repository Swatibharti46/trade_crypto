def almgren_chriss_optimal_execution(S0, X, T, N, sigma, eta, gamma, lam):
    import numpy as np

    dt = T / N
    t = np.linspace(0, T, N + 1)

    # Optimal trading trajectory
    k = np.sqrt(lam * sigma**2 / eta)
    sinh_kT = np.sinh(k * T)
    x_star = X * np.sinh(k * (T - t)) / sinh_kT

    # Cost and variance
    expected_cost = lam * sigma**2 * np.sum(np.diff(x_star)**2 * dt)
    variance = sigma**2 * np.sum((x_star[:-1] - x_star[1:])**2 * dt)

    return t, x_star, expected_cost, variance
