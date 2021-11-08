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


def RHS2casF(
    RHS, q_vars, u_vars=None,
):
    from .symbolic import find_arguments, standard_notation, diff_to_symb_expr

    RHS = list(RHS)
    RHS = [standard_notation(diff_to_symb_expr(expr)) for expr in RHS]
    arguments = find_arguments(RHS, q_vars, u_vars)
    q_args, v_args, x_args_found, u_args, u_args_found, params = arguments
    x_args = q_args + v_args

    funcs = v_args + RHS
    all_vars = x_args + u_args_found + params
    msg = "Function Arguments:\n"
    msg += f"\tx: {x_args}\n"
    msg += f"\tu: {u_args_found}\n"
    msg += f"\tparams: {params}\n"
    print(msg)
    cas_x_args = cas.MX.sym("x", len(x_args))
    cas_u_args = cas.MX.sym("u", len(u_args_found))
    cas_params = cas.MX.sym("p", len(params))
    cas_all_vars = [cas_x_args[ii] for ii in range(len(x_args))]
    cas_all_vars += [cas_u_args[ii] for ii in range(len(u_args_found))]
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


def restriction2casadi(F_scheme, F, n_vars, n_u, n_params, n_scheme_params=0):
    """
    Converts a restriction funtion F to a casadi function that can be
    more efficiently used in casadi

    Parameters
    ----------
    F_scheme : Function of the form F(x, x_n, u, u_n, F, dt, p, [sch_p])
        Restriction function that each step has to be equal to zero,
        argument sch_p is only mandatory if n_scheme_params != 0
    F : Function of the form F(x, u, p)
        Physics function that describes the system
    n_vars : int
        Number of q variables or coordinates in the problem, x variables
        will be then twice this amount as they include velocities.
    n_u : int
        Number of u variables or actions in the problem
    n_params : int
        Number of parameters in the problem
    n_scheme_params : int, default 0
        Number of scheme parameters, not passed to F(x, u, p)

    Returns
    -------
    Casadi Function
        A casadi function of the form F(x, x_n, u, u_n, dt, p, sch_p)
        Restriction function that each step has to be equal to zero

    """
    from inspect import signature

    if n_scheme_params != 0 and len(signature(F_scheme).parameters) == 7:
        raise ValueError(
            "Detected a value of n_scheme_params larger than zero in a function F_scheme that does not contain sch_p argument"
        )
    x = cas.SX.sym("x", 2 * n_vars).T
    x_n = cas.SX.sym("x_n", 2 * n_vars).T
    u = cas.SX.sym("u", n_u).T
    u_n = cas.SX.sym("u_n", n_u).T
    p = cas.SX.sym("p", n_params)
    dt = cas.SX.sym("dt")
    if n_scheme_params == 0:
        result = F_scheme(x, x_n, u, u_n, F, dt, p)
        return cas.Function(
            "Restriction",
            [x, x_n, u, u_n, dt, p],
            [result,],
            ["x", "x_n", "u", "u_n", "dt", "params"],
            ["residue"],
        )
    else:
        sch_p = cas.SX.sym("sch_p", n_scheme_params)
        result = F_scheme(x, x_n, u, u_n, F, dt, p, sch_p)
        return cas.Function(
            "Restriction",
            [x, x_n, u, u_n, dt, p, sch_p],
            [result,],
            ["x", "x_n", "u", "u_n", "dt", "params", "scheme_params"],
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
