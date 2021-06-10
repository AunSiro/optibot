# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:52:22 2021

@author: Siro Moreno
"""

import numpy as np
from numpy import sin, cos


def get_str(x):
    return x.__str__()


def unpack(arr):
    arr = np.array(arr)
    dim = arr.shape[-1]
    axnum = len(arr.shape)
    if axnum == 1:
        res = [arr[ii] for ii in range(dim)]
    elif axnum == 2:
        res = [arr[:, ii] for ii in range(dim)]
    else:
        raise ValueError("The array has too many values to unpack")
    return res


def congruent_concatenate(varlist):
    x = np.array(varlist[0])
    if x.size == 1:
        assert np.all([len(np.array(ii).shape) == 0 for ii in varlist])
        res = np.array(varlist)
    else:
        assert np.all([len(np.array(ii).shape) == 1 for ii in varlist])
        vert_varlist = [np.expand_dims(ii, axis=1) for ii in varlist]
        res = np.concatenate(vert_varlist, 1)
    return res


def RHS2numpy(RHS, n_var):
    from sympy import symbols, Symbol, lambdify

    RHS = list(RHS)
    q_args = []
    v_args = []
    u_args = []
    param_list = []
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
                if not symb in param_list:
                    param_list.append(symb)
        funcs.append(expr)
    param_list = sorted(param_list, key=get_str)
    funcs = v_args + funcs
    all_vars = x_args + u_args + param_list
    msg = "Function Arguments:\n"
    msg += f"\tx: {x_args}\n"
    msg += f"\tu: {u_args}\n"
    msg += f"\tparams: {param_list}\n"
    print(msg)
    np_funcs = []
    for function in funcs:
        np_funcs.append(lambdify(all_vars, function))

    def New_F(x, u, params):
        all_np_vars = unpack(x) + unpack(u) + unpack(params)
        results = []
        for func in np_funcs:
            results.append(func(*all_np_vars))
        return congruent_concatenate(results)

    return New_F


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

    return congruent_concatenate(result)
