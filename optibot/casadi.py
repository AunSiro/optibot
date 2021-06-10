# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:52:24 2021

@author: Siro Moreno

sympy2casadi function original author: Joris Gillis
https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8

"""

import casadi as cas
from casadi import sin, cos


def get_str(x):
    return x.__str__()


def list2casadi(vallist):
    """convert a list into a casadi array of the apropiate shape"""
    return cas.horzcat(*vallist).T


def sympy2casadi(sympy_expr, sympy_var, casadi_var):
    """
    Transforms a sympy expression into a casadi function.

    Parameters
    ----------
    sympy_expr : sympy expression
        
    sympy_var : list of sympy symbols
        
    casadi_var : list of casady symbols
        

    Returns
    -------
    Casadi Function

    """
    # assert casadi_var.is_vector()
    # if casadi_var.shape[1] > 1:
    #    casadi_var = casadi_var.T
    # casadi_var = cas.vertsplit(casadi_var)
    from sympy.utilities.lambdify import lambdify

    mapping = {
        "ImmutableDenseMatrix": cas.blockcat,
        "MutableDenseMatrix": cas.blockcat,
        "Abs": cas.fabs,
    }
    f = lambdify(sympy_var, sympy_expr, modules=[mapping, cas])
    return f(*casadi_var)


def symlist2cas(symlist):
    caslist = []
    for symbol in symlist:
        caslist.append(cas.MX.sym(symbol.__str__()))
    return caslist


def RHS2casF(RHS, n_var):
    from sympy import symbols, Symbol

    RHS = list(RHS)
    q_args = []
    v_args = []
    u_args = []
    params = []
    args = []
    funcs = []
    for jj in range(n_var):
        q = symbols(f"q_{jj}")
        q_args.append(q)
        v = symbols(f"v_{jj}")
        v_args.append(v)
        u = symbols(f"u_{jj}")
        u_args.append(u)
        args += [q, v, u]
    x_args = q_args + v_args
    for ii in range(len(RHS)):
        expr = RHS[ii]
        var_set = expr.atoms(Symbol)
        for symb in var_set:
            if not symb in args:
                if not symb in params:
                    params.append(symb)
        funcs.append(expr)
    funcs = v_args + funcs
    params = sorted(params, key=get_str)
    all_vars = x_args + u_args + params
    msg = "Function Arguments:\n"
    msg += f"\tx: {x_args}\n"
    msg += f"\tu: {u_args}\n"
    msg += f"\tparams: {params}\n"
    print(msg)
    cas_x_args = cas.MX.sym("x", len(x_args))
    cas_u_args = cas.MX.sym("u", len(u_args))
    cas_params = cas.MX.sym("p", len(params))
    cas_all_vars = [cas_x_args[ii] for ii in range(n_var * 2)]
    cas_all_vars += [cas_u_args[ii] for ii in range(n_var)]
    cas_all_vars += [cas_params[ii] for ii in range(len(params))]
    cas_funcs = []
    for function in funcs:
        cas_funcs.append(sympy2casadi(function, all_vars, cas_all_vars))
    cas_funcs = cas.horzcat(*cas_funcs)
    return cas.Function(
        "F",
        [cas_x_args, cas_u_args, cas_params],
        [cas_funcs,],
        ["x", "u", "params"],
        ["x_dot"],
    )


def unpack(arr):
    arr = cas.horzcat(arr)
    if arr.shape[-1] == 1:
        arr = arr.T
    dim = arr.shape[-1]
    res = [arr[:, ii] for ii in range(dim)]
    return res


def restriction2casadi(F_scheme, F, n_vars, n_params):
    """
    Converts a restriction funtion F to a casadi function that can be
    more efficiently used in casadi

    Parameters
    ----------
    F_scheme : Function of the form F(x, x_n, u, u_n, F, dt, p)
        Restriction function that each step has to be equal to zero
    F : Function of the form F(x, u, p)
        Physics function that describes the system
    n_vars : int
        Number of q variables or coordinates in the problem
    n_params : int
        Number of parameters in the problem

    Returns
    -------
    Casadi Function
        A casadi function of the form F(x, x_n, u, u_n, dt, p)
        Restriction function that each step has to be equal to zero

    """
    x = cas.MX.sym("x", 2 * n_vars).T
    x_n = cas.MX.sym("x_n", 2 * n_vars).T
    u = cas.MX.sym("u", n_vars).T
    u_n = cas.MX.sym("u_n", n_vars).T
    p = cas.MX.sym("p", n_params)
    dt = cas.MX.sym("dt")
    result = F_scheme(x, x_n, u, u_n, F, dt, p)
    return cas.Function(
        "Restriction",
        [x, x_n, u, u_n, dt, p],
        [result,],
        ["x", "x_n", "u", "u_n", "dt", "params"],
        ["residue"],
    )


# --- Double Pendulum ---


def doub_pend_F(x, u, params=[1, 1, 1, 1, 1]):
    q_0, q_1, v_0, v_1 = unpack(x)
    u_0, u_1 = unpack(u)
    m_1, l_1, l_0, m_0, g, m_1, l_1, l_0, m_0, g = params
    result = [
        v_0,
        v_1,
    ]
    result.append(
        (
            l_0
            * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0 ** 2 * sin(q_0 - q_1)) - u_1)
            * cos(q_0 - q_1)
            + l_1
            * (
                -l_0
                * (
                    g * m_0 * sin(q_0)
                    + g * m_1 * sin(q_0)
                    + l_1 * m_1 * v_1 ** 2 * sin(q_0 - q_1)
                )
                + u_0
            )
        )
        / (l_0 ** 2 * l_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    )
    result.append(
        (
            -l_0
            * (m_0 + m_1)
            * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0 ** 2 * sin(q_0 - q_1)) - u_1)
            + l_1
            * m_1
            * (
                l_0
                * (
                    g * m_0 * sin(q_0)
                    + g * m_1 * sin(q_0)
                    + l_1 * m_1 * v_1 ** 2 * sin(q_0 - q_1)
                )
                - u_0
            )
            * cos(q_0 - q_1)
        )
        / (l_0 * l_1 ** 2 * m_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    )

    return cas.horzcat(*result)
