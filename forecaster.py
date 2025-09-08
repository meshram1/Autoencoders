import numpy as no
from scipy.integrate import solve_ivp

def sindy_forecaster(library, coeff, x0, t_span, t_eval):

    def rhs(t, x):
        Theta = library(x.reshape(1, -1))  # shape (1, n_features)
        return (Theta @ coeff).ravel()        # shape (n_states,)

    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8, max_step=0.01 )
    return sol.t, sol.y.T, t_eval[-1], sol.t[-1]




