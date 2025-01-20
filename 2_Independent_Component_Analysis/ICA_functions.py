import numpy as np
import sympy as sym


# symbolic differentioation and convertion into python functions
def get_g_dg_d2g_over_dg(sym_g, different_g=False):
    sym_x = sym.symbols('x', real=True)
    sym_g = sym_g(sym_x)
    sym_dg = sym.diff(sym_g).simplify()
    sym_d2g = sym.diff(sym_dg).simplify()
    sym_d2g_over_dg = (sym_d2g / sym_dg).simplify()
    
    g = sym.lambdify(sym_x, sym_g)
    dg = sym.lambdify(sym_x, sym_dg)
    d2g_over_dg = sym.lambdify(sym_x, sym_d2g_over_dg)
    
    return g, dg, d2g_over_dg


def calc_h(T, dg_y, W):
    return 1/T * np.sum(np.log(dg_y)) + np.log(np.abs(np.linalg.det(W)))


def calc_grad_W(T, x, d2g_over_dg_y, W):
    return np.linalg.inv(W).T + 1/T * np.einsum('in,jn->ij', d2g_over_dg_y, x)
