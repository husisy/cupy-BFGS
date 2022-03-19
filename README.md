# BFGS in cupy

reference

1. [github/trsav/bfgs](https://github.com/trsav/bfgs) BFGS in numpy
2. [wiki/Wolfe-conditions](https://en.wikipedia.org/wiki/Wolfe_conditions)
3. [wiki/BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
4. `scipy.optimize.minimize`
5. [scipy.minpack2.dcsrch](https://github.com/scipy/scipy/blob/main/scipy/optimize/minpack2/dcsrch.f) fortran

quickstart

```bash
$ python draft00.py
# scipy.optimize.minimize
x: [1.         1.         1.         1.         1.00000001]
fun: 1.555543986341927e-17
nfev: 32
# cupy_optimize_minimize_BFGS
x: [1. 1. 1. 1. 1.]
fun: 9.178304967072598e-17
num_step: 68
```
