import numpy as np

def lagrangian(x1, x2, lambda1, lambda2):
    return x1**2 + 3*x1*x2 + 9*x2**2 + 2*x1 - 5*x2 + lambda1 * (2*x1 + x2 - 3) + lambda2 * (-x1 - 2*x2 + 3)

def kkt_conditions(x1, x2, lambda1, lambda2):
    # Gradient of the Lagrangian with respect to x1 and x2
    dL_dx1 = 2*x1 + 3*x2 + 2 + 2*lambda1 - lambda2
    dL_dx2 = 3*x1 + 18*x2 - 5 + lambda1 - 2*lambda2
    print(dL_dx1, dL_dx2)
    # Constraints
    g1 = 2*x1 + x2 - 3
    g2 = -x1 - 2*x2 + 3
    
    # KKT Conditions
    # 1. Stationarity
    stationarity_x1 = dL_dx1 <1e-6
    stationarity_x2 = dL_dx2 < 1e-5
    
    # 2. Primal feasibility
    primal_feasibility_g1 = g1 >= 0
    primal_feasibility_g2 = g2 >= 0
    
    # 3. Dual feasibility
    dual_feasibility_lambda1 = lambda1 >= 0
    dual_feasibility_lambda2 = lambda2 >= 0

    # 4. Complementary slackness
    complementary_slackness_1 = np.isclose(lambda1 * g1, 0)
    complementary_slackness_2 = np.isclose(lambda2 * g2, 0)
    
    return (stationarity_x1 and stationarity_x2 and
            primal_feasibility_g1 and primal_feasibility_g2 and
            dual_feasibility_lambda1 and dual_feasibility_lambda2 and
            (complementary_slackness_1 or complementary_slackness_2))

# Example usage (replace with your actual solutions)
x1_star = 1.285714   # Example value
x2_star = 0.857142  # Example value
lambda1_star = 0.0 # Example value
lambda2_star = 7.14285714 # Example value

is_optimal = kkt_conditions(x1_star, x2_star, lambda1_star, lambda2_star)

if is_optimal:
    print("The solution satisfies the KKT conditions and is likely optimal.")
else:
    print("The solution does not satisfy the KKT conditions.")