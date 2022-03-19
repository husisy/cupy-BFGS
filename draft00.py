import numpy as np
import cupy as cp
import scipy.optimize

from utils import cupy_optimize_minimize_BFGS


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_der.html#scipy.optimize.rosen_der
def hf_rosen_np(x):
    x = np.asarray(x)
    fval = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    grad = np.zeros_like(x)
    grad[1:-1] = (200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return fval, grad


def hf_rosen_cp(x):
    x = cp.asarray(x)
    fval = cp.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    grad = cp.zeros_like(x)
    grad[1:-1] = (200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return fval, grad


np_rng = np.random.default_rng()

N0 = 5

x0 = np_rng.uniform(0, 1, N0)
ret_np = scipy.optimize.minimize(hf_rosen_np, x0, method='BFGS', jac=True, options={'gtol':1e-6})
print('# scipy.optimize.minimize')
print('x:', ret_np.x)
print('fun:', ret_np.fun)
print('nfev:', ret_np.nfev)

ret_cp = cupy_optimize_minimize_BFGS(hf_rosen_cp, x0, gtol=1e-6, tag_kernel=False)
print('# cupy_optimize_minimize_BFGS')
print('x:', ret_cp.x)
print('fun:', ret_cp.fun)
print('num_step:', ret_cp.num_step)
