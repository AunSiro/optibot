# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:52:22 2021

@author: Siro Moreno

Here we present some tools and functions designed to operate with and
conert to numpy arrays.
"""

import numpy as np
from numpy import sin, cos, expand_dims


def get_str(x):
    return x.__str__()


def unpack(arr):
    arr = np.array(arr)
    # If arr was a number, it will produce errors later
    # We have to to convert it
    # into a 1D array of lenght 1
    if arr.size == 1 and arr.shape == ():
        arr = expand_dims(arr, axis=0)
    dim = arr.shape[-1]
    axnum = len(arr.shape)
    if axnum == 1:
        res = [arr[ii] for ii in range(dim)]
    elif axnum == 2:
        res = [arr[:, ii] for ii in range(dim)]
    else:
        raise ValueError("The array has too many dimensions to unpack")
    return res


def combinefunctions(*functions):
    def combined(*args, **kwargs):
        res = [f(*args, **kwargs) for f in functions]
        return np.array(res).T

    return combined


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


def trapz_integrate(X, t_array):
    new_X = np.zeros_like(X)
    for i in range(1, X.shape[0]):
        new_X[i] = np.trapz(X[: i + 1], t_array[: i + 1])
    return new_X


def num_derivative(X, h):
    X_dot = np.zeros_like(X)
    X_dot[1:-1] = (X[2:] - X[:-2]) / (2 * h)
    X_dot[0] = (X[1] - X[0]) / h
    X_dot[-1] = (X[-2] - X[-1]) / h
    return X_dot


def RHS2numpy(RHS, q_vars, u_vars=None, verbose=False, mode="x"):
    """
    Converts an array of symbolic expressions RHS(x, u, params) to a Numpy function.
    Designed to work with systems so that
        x' = RHS(x, u, params)

    Parameters
    ----------
    RHS : Sympy matrix
        Vertical symbolic matrix RHS(x, u, lambdas, params)
    q_vars : int or list of dynamic symbols
        Determine the symbols that will be searched
        if int, the program will assume q as q_i for q in [0,q_vars]
    u_vars : None, int or list of symbols. Default is None.
        Symbols that will be sarched and separated. 
        If None, symbols of the form u_ii where ii is a number will be 
        assumed
    verbose : Bool, optional
        wether to print aditional information of expected and found variables
        in the given expression
    mode : str
        if mode == 'x', a function F(x, u, params) = [x'] will be returned
        if mode == 'q', a function F(q, v, u, params) = [a] will be returned

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from sympy import lambdify
    from .symbolic import find_arguments, standard_notation, diff_to_symb_expr

    RHS = list(RHS)
    RHS = [standard_notation(diff_to_symb_expr(expr)) for expr in RHS]
    arguments = find_arguments(RHS, q_vars, u_vars, verbose=verbose)
    q_args, v_args, a_args, u_args, params, lambda_args = arguments
    x_args = q_args + v_args

    if mode == "x":
        if len(q_args) == len(RHS):
            funcs = v_args + RHS
        elif len(RHS) == len(x_args):
            funcs = RHS
        else:
            raise ValueError(
                f"Unrecognized RHS shape, detected elements = {len(RHS)}, expected {len(q_args)} or {len(x_args)}"
            )
    elif mode == "q":
        if len(q_args) == len(RHS):
            funcs = RHS
        elif len(RHS) == len(x_args):
            funcs = RHS[len(RHS) // 2 :]
        else:
            raise ValueError(
                f"Unrecognized RHS shape, detected elements = {len(RHS)}, expected {len(q_args)} or {len(x_args)}"
            )
    else:
        raise ValueError(f'Unexpected mode {mode}, valid values are "x" and "q"')

    all_vars = x_args + u_args + params
    msg = "Function Arguments:\n"
    if mode == "x":
        msg += f"\tx: {x_args}\n"
    elif mode == "q":
        msg += f"\tq: {q_args}\n"
        msg += f"\tv: {v_args}\n"
    msg += f"\tu: {u_args}\n"
    msg += f"\tparams: {params}\n"
    print(msg)
    np_funcs = []
    for function in funcs:
        np_funcs.append(lambdify(all_vars, function))

    if mode == "x":

        def New_F(x, u, params):
            all_np_vars = unpack(x) + unpack(u) + unpack(params)
            results = []
            for func in np_funcs:
                results.append(func(*all_np_vars))
            return congruent_concatenate(results)

    elif mode == "q":

        def New_F(q, v, u, params):
            all_np_vars = unpack(q) + unpack(v) + unpack(u) + unpack(params)
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
