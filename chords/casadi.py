# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:52:24 2021

@author: Siro Moreno

Here we present some tools and functions dedicated to operate with and
convert to casadi objects.

sympy2casadi function adapted from Joris Gillis:
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

    casadi_var : list of casadi symbols


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


def unpack(arr):
    arr = cas.horzcat(arr)
    if arr.shape[-1] == 1:
        arr = arr.T
    dim = arr.shape[-1]
    res = [arr[:, ii] for ii in range(dim)]
    return res


def rhs_to_casadi_function(
    RHS, q_vars, u_vars=None, verbose=False, mode="x", silent=False
):
    """
    Converts an array of symbolic expressions RHS(x, u, params) to a casadi
    function.
    Designed to work with systems so that either
        x' = RHS(x, u, params)
    or
        a = RHS(x, u, params)

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
    verbose : Bool, optional, default = False
        wether to print aditional information of expected and found variables
        in the given expression
    mode : str
        if mode == 'x', a function F(x, u, params) = [x'] will be returned
        if mode == 'q', a function G(q, v, u, params) = [a] will be returned
    silent: Bool, optional, default False
        if True, nothing will be printed. If verbose = True, silent will be ignored.

    Returns
    -------
    Casadi Function
        Either F(x, u, params) or G(q, v, u, params)

    """
    from .symbolic import find_arguments, standard_notation, diff_to_symb_expr

    if verbose:
        silent = False
    RHS = list(RHS)
    RHS = [standard_notation(diff_to_symb_expr(expr)) for expr in RHS]
    arguments = find_arguments(RHS, q_vars, u_vars, verbose=verbose)
    q_args, v_args, _, u_args_found, params, lambda_args = arguments
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

    all_vars = x_args + u_args_found + params
    msg = "Function Arguments:\n"
    if mode == "x":
        msg += f"\tx: {x_args}\n"
    elif mode == "q":
        msg += f"\tq: {q_args}\n"
        msg += f"\tv: {v_args}\n"
    msg += f"\tu: {u_args_found}\n"
    msg += f"\tparams: {params}\n"
    if not silent:
        print(msg)
    cas_x_args = cas.MX.sym("x", len(x_args))
    cas_q_args = cas.MX.sym("q", len(q_args))
    cas_v_args = cas.MX.sym("v", len(v_args))
    cas_u_args = cas.MX.sym("u", len(u_args_found))
    cas_params = cas.MX.sym("p", len(params))
    if mode == "x":
        cas_all_vars = [cas_x_args[ii] for ii in range(len(x_args))]
        f_arg_list = [cas_x_args, cas_u_args, cas_params]
        f_arg_names = ["x", "u", "params"]
        f_out_names = ["x_dot"]
    elif mode == "q":
        cas_all_vars = [cas_q_args[ii] for ii in range(len(q_args))]
        cas_all_vars += [cas_v_args[ii] for ii in range(len(v_args))]
        f_arg_list = [cas_q_args, cas_v_args, cas_u_args, cas_params]
        f_arg_names = ["q", "v", "u", "params"]
        f_out_names = ["a"]
    cas_all_vars += [cas_u_args[ii] for ii in range(len(u_args_found))]
    cas_all_vars += [cas_params[ii] for ii in range(len(params))]
    cas_funcs = []
    for function in funcs:
        cas_funcs.append(sympy2casadi(function, all_vars, cas_all_vars))
    cas_funcs = cas.horzcat(*cas_funcs)
    return cas.Function(
        "F",
        f_arg_list,
        [
            cas_funcs,
        ],
        f_arg_names,
        f_out_names,
    )


def implicit_dynamic_x_to_casadi_function(
    D, x_vars, u_vars=None, verbose=False, silent=False
):
    """
    Converts an array D(x, x', u, lambdas, params) of symbolic expressions to a
    Casadi function.

    Symbols in the expressions not found in x_vars, x_dot_vars or u_vars
    will be considered parameters.

    Parameters
    ----------
    D : Sympy matrix
        Vertical symbolic matrix D(x, x', u, lambdas, params)
    x_vars : int or list of Sympy dynamic symbols
        list of x symbols to look for in the expressions.
        If int, they will be generated as 'x_i' for i in [0, x_vars]
    u_vars : list of Sympy dynamic symbols
        List of u symbols to look for. The default is None.
    verbose : Bool, optional, default = False
        wether to print aditional information of expected and found variables
        in the given expression
    silent: Bool, optional, default False
        if True, nothing will be printed. If verbose = True, silent will be ignored.

    Returns
    -------
    Casadi Function
        Casadi Function of x, x', u, lambdas, params.

    """
    from .symbolic import find_arguments, standard_notation, diff_to_symb_expr
    from sympy.physics.mechanics import dynamicsymbols

    if verbose:
        silent = False
    D = list(D)
    D = [standard_notation(diff_to_symb_expr(expr)) for expr in D]
    if type(x_vars) == int:
        x_vars = list(dynamicsymbols("x_0:" + str(x_vars)))
    elif type(x_vars) != list:
        raise TypeError("x_vars must be int or list of symbols")
    arguments = find_arguments(
        D, x_vars, u_vars, separate_lambdas=True, verbose=verbose
    )
    x_args, x_dot_args, _, u_args, params, lambda_args = arguments

    all_vars = x_args + x_dot_args + u_args + lambda_args + params
    msg = "Function Arguments:\n"
    msg += f"\tx: {x_args}\n"
    msg += f"\tx_dot: {x_dot_args}\n"
    msg += f"\tu: {u_args}\n"
    msg += f"\tlambdas: {lambda_args}\n"
    msg += f"\tparams: {params}\n"
    if not silent:
        print(msg)
    cas_x_args = cas.MX.sym("x", len(x_args))
    cas_x_dot_args = cas.MX.sym("x", len(x_dot_args))
    cas_u_args = cas.MX.sym("u", len(u_args))
    cas_lambda_args = cas.MX.sym("u", len(lambda_args))
    cas_params = cas.MX.sym("p", len(params))
    cas_all_vars = [cas_x_args[ii] for ii in range(len(x_args))]
    cas_all_vars += [cas_x_dot_args[ii] for ii in range(len(x_dot_args))]
    cas_all_vars += [cas_u_args[ii] for ii in range(len(u_args))]
    cas_all_vars += [cas_lambda_args[ii] for ii in range(len(lambda_args))]
    cas_all_vars += [cas_params[ii] for ii in range(len(params))]

    cas_funcs = []
    for function in D:
        cas_funcs.append(sympy2casadi(function, all_vars, cas_all_vars))
    cas_funcs = cas.horzcat(*cas_funcs)
    return cas.Function(
        "M",
        [cas_x_args, cas_x_dot_args, cas_u_args, cas_lambda_args, cas_params],
        [
            cas_funcs,
        ],
        ["x", "x_dot", "u", "lambdas", "params"],
        ["residue"],
    )


def implicit_dynamic_q_to_casadi_function(
    D, q_vars, u_vars=None, verbose=False, silent=False
):
    """
    Converts an array D(q, q', q'', u, lambdas, params) of symbolic expressions to a
    Casadi function.

    Symbols in the expressions not found in x_vars, x_dot_vars or u_vars
    will be considered parameters.

    Parameters
    ----------
    D : Sympy matrix
        Vertical symbolic matrix D(q, q', q'', u, lambdas, params)
    q_vars : int or list of Sympy dynamic symbols
        list of q symbols to look for in the expressions.
        If int, they will be generated as 'q_i' for i in [0, q_vars]
    u_vars : list of Sympy dynamic symbols
        List of u symbols to look for. The default is None.
    verbose : Bool, optional, default = False
        wether to print aditional information of expected and found variables
        in the given expression
    silent: Bool, optional, default False
        if True, nothing will be printed. If verbose = True, silent will be ignored.

    Returns
    -------
    Casadi Function
        Casadi Function of q, q', q'', u, lambdas, params.

    """
    from .symbolic import find_arguments, standard_notation, diff_to_symb_expr
    from sympy.physics.mechanics import dynamicsymbols

    if verbose:
        silent = False
    D = list(D)
    D = [standard_notation(diff_to_symb_expr(expr)) for expr in D]
    if type(q_vars) == int:
        q_vars = list(dynamicsymbols("q_0:" + str(q_vars)))
    elif type(q_vars) != list:
        raise TypeError("q_vars must be int or list of symbols")

    arguments = find_arguments(
        D, q_vars, u_vars, separate_as=True, separate_lambdas=True, verbose=verbose
    )
    q_args, v_args, a_args, u_args, params, lambda_args = arguments

    all_vars = q_args + v_args + a_args + u_args + lambda_args + params
    msg = "Function Arguments:\n"
    msg += f"\tq: {q_args}\n"
    msg += f"\tv: {v_args}\n"
    msg += f"\ta: {a_args}\n"
    msg += f"\tu: {u_args}\n"
    msg += f"\tlambda: {lambda_args}\n"
    msg += f"\tparams: {params}\n"
    if not silent:
        print(msg)
    cas_q_args = cas.MX.sym("q", len(q_args))
    cas_v_args = cas.MX.sym("v", len(v_args))
    cas_a_args = cas.MX.sym("a", len(a_args))
    cas_u_args = cas.MX.sym("u", len(u_args))
    cas_lambda_args = cas.MX.sym("lambda", len(lambda_args))
    cas_params = cas.MX.sym("p", len(params))
    cas_all_vars = [cas_q_args[ii] for ii in range(len(q_args))]
    cas_all_vars += [cas_v_args[ii] for ii in range(len(v_args))]
    cas_all_vars += [cas_a_args[ii] for ii in range(len(a_args))]
    cas_all_vars += [cas_u_args[ii] for ii in range(len(u_args))]
    cas_all_vars += [cas_lambda_args[ii] for ii in range(len(lambda_args))]
    cas_all_vars += [cas_params[ii] for ii in range(len(params))]
    cas_funcs = []
    for function in D:
        cas_funcs.append(sympy2casadi(function, all_vars, cas_all_vars))
    cas_funcs = cas.horzcat(*cas_funcs)
    return cas.Function(
        "F",
        [cas_q_args, cas_v_args, cas_a_args, cas_u_args, cas_lambda_args, cas_params],
        [
            cas_funcs,
        ],
        ["q", "v", "a", "u", "lambda", "params"],
        ["Residue"],
    )


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
            [
                result,
            ],
            ["x", "x_n", "u", "u_n", "dt", "params"],
            ["residue"],
        )
    else:
        sch_p = cas.SX.sym("sch_p", n_scheme_params)
        result = F_scheme(x, x_n, u, u_n, F, dt, p, sch_p)
        return cas.Function(
            "Restriction",
            [x, x_n, u, u_n, dt, p, sch_p],
            [
                result,
            ],
            ["x", "x_n", "u", "u_n", "dt", "params", "scheme_params"],
            ["residue"],
        )


def accelrestriction2casadi(F_scheme, n_vars, n_scheme_params=0, order=2):
    """
    Converts a restriction funtion F to a casadi function that can be
    more efficiently used in casadi

    Parameters
    ----------
    F_scheme : Function of the form F(x, x_n, a, a_n, dt, scheme_params)
        Restriction function that each step has to be equal to zero
    n_vars : int
        Number of q variables or coordinates in the problem, x variables
        will be then twice this amount as they include velocities.
    n_scheme_params : int, default 0
        Number of scheme parameters, not involved in the dynamics

    Returns
    -------
    Casadi Function
        A casadi function of the form F(x, x_n, a, a_n, dt, scheme_params)
        Restriction function that each step has to be equal to zero

    """
    x = cas.SX.sym("x", order * n_vars).T
    x_n = cas.SX.sym("x_n", order * n_vars).T
    a = cas.SX.sym("a", n_vars).T
    a_n = cas.SX.sym("a_n", n_vars).T
    dt = cas.SX.sym("dt")

    sch_p = cas.SX.sym("sch_p", n_scheme_params).T
    result = F_scheme(x, x_n, a, a_n, dt, sch_p)
    return cas.Function(
        "Restriction",
        [x, x_n, a, a_n, dt, sch_p],
        [
            result,
        ],
        ["x", "x_n", "a", "a_n", "dt", "scheme_params"],
        ["residue"],
    )


# --- Double Pendulum Example---


def doub_pend_F(x, u, params=(1, 1, 1, 1, 1)):
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
            * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0**2 * sin(q_0 - q_1)) - u_1)
            * cos(q_0 - q_1)
            + l_1
            * (
                -l_0
                * (
                    g * m_0 * sin(q_0)
                    + g * m_1 * sin(q_0)
                    + l_1 * m_1 * v_1**2 * sin(q_0 - q_1)
                )
                + u_0
            )
        )
        / (l_0**2 * l_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    )
    result.append(
        (
            -l_0
            * (m_0 + m_1)
            * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0**2 * sin(q_0 - q_1)) - u_1)
            + l_1
            * m_1
            * (
                l_0
                * (
                    g * m_0 * sin(q_0)
                    + g * m_1 * sin(q_0)
                    + l_1 * m_1 * v_1**2 * sin(q_0 - q_1)
                )
                - u_0
            )
            * cos(q_0 - q_1)
        )
        / (l_0 * l_1**2 * m_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    )

    return cas.horzcat(*result)
