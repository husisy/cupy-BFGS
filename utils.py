# https://github.com/trsav/bfgs
import numpy as np
import cupy as cp
import argparse


def line_search_cp(hf0, x0, p, fx0, grad0, c1=1e-4, c2=0.9):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    Wolfe conditions https://en.wikipedia.org/wiki/Wolfe_conditions
    '''
    a = 1
    grad0_x_p = cp.dot(grad0, p)
    c1_nabla_x_p = c1 * grad0_x_p
    c2_nabla_x_p = c2 * grad0_x_p
    num_step = 0
    while True:
        x_new = x0 + a * p
        fval_k,grad_k = hf0(x_new)
        num_step += 1
        if (fval_k < (fx0+a*c1_nabla_x_p)) and (cp.dot(grad_k, p) > c2_nabla_x_p):
            break
        a *= 0.5
    return a,x_new,fval_k,grad_k,num_step

hf_bfgs_ekernel = cp.ElementwiseKernel(
    in_params='T H, T s0, T s1, T hy0, T hy1, raw T r, raw T t',
    out_params='T ret',
    operation='ret = H - s0*hy1*r[0] - hy0*s1*r[0] + s0*s1*t[0];',
    name='bfgs_ekernel',
)


_epsilon = np.sqrt(np.finfo(float).eps)

def cupy_optimize_minimize_BFGS(hf0, x0, gtol=_epsilon, maxiter=None, tag_kernel=False):
    # https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
    x0 = cp.asarray(x0, dtype=float).reshape(-1)
    N0 = len(x0)
    if maxiter is None:
        maxiter = N0 * 200
    I = cp.eye(N0, dtype=x0.dtype)
    H = I
    H_buffer = cp.empty_like(H)

    xk = x0
    fval_k,grad_k = hf0(xk)
    fval_history = [fval_k]
    num_step_total = 1
    for _ in range(maxiter):
        if cp.amax(cp.abs(grad_k))<gtol:
            break
        p = -(H @ grad_k)
        a,xk1,fval_k1,grad_k1,num_step_i = line_search_cp(hf0, xk, p, fval_k, grad_k)
        num_step_total += num_step_i
        y = grad_k1 - grad_k
        s = a*p
        r = 1/cp.dot(y, s)
        if tag_kernel:
            # tag_kernel=True and tag_kernel=False should give the same results, tag_kernel=True might be a little faster
            hy = H @ y
            t = cp.dot(y,y)*r**2 + r
            hf_bfgs_ekernel(H, s[:,np.newaxis], s[np.newaxis], hy[:,np.newaxis], hy[np.newaxis], r, t, H_buffer)
            H,H_buffer = H_buffer,H
        else:
            tmp0 = I - (r*s)[:,np.newaxis]*y
            H = tmp0 @ H @ tmp0.T + (r*s)[:,np.newaxis]*s
        xk,fval_k,grad_k = xk1,fval_k1,grad_k1
        fval_history.append(fval_k)
    fval_history = cp.array(fval_history)
    ret = argparse.Namespace(x=xk, fun=fval_k, grad=grad_k, fval_history=fval_history, num_step=num_step_total)
    return ret

