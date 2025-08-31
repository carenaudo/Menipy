import numpy as np

def toy_young_laplace(params, physics, geometry):
    R0 = float(params[0])
    phi = np.linspace(0.0, np.pi, 200)
    r = R0*np.sin(phi)
    z = R0*(np.cos(phi) - 1.0)
    return np.column_stack([r, z])

SOLVERS = {"toy_yl": toy_young_laplace}
