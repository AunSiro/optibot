# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:05:38 2020

@author: Siro Moreno
"""
from sympy import (
    symbols,
    pi,
    cos,
    sin,
    simplify,
    integrate,
    Eq,
    solve,
    dsolve,
    Matrix,
    preorder_traversal,
    Float,
    solve_linear_system,
    eye,
    zeros,
    lambdify,
    sqrt,
    Symbol,
)
from sympy.physics.mechanics import dynamicsymbols
from sympy.functions import sign


def integerize(expr):
    expr2 = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr2 = expr2.subs(a, round(a))
    return expr2


def roundize(expr, n=4):
    expr2 = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr2 = expr2.subs(a, round(a, n))
    return expr2


def lagrange(L_expr, var):
    """
    Parameters
    ----------
    L_expr : Symbolic Expression of Lagrange Function
        
    var : Symbol of a variable
        

    Returns
    -------
    Symbolic Expressions
        Lagrange equation respect to the variable.

    """
    t = symbols("t")
    vardot = var.diff(t)
    lag1 = L_expr.diff(vardot).simplify().diff(t).simplify()
    lag2 = L_expr.diff(var).simplify()
    lag = lag1 - lag2
    return lag.simplify().expand()


def get_lagr_eqs(T, U, n_vars):
    """
    Get a list of lagrange equations. T and U are Kinetic energy
    and Potential energy, as functions of coordinates q, its
    derivatives and other parameters

    Parameters
    ----------
    T : Symbolic Expression
        Kinetic Energy as function of q_i with i in (0,n_vars).
    U : Symbolic Expression
        Potential Energy as function of q_i with i in (0,n_vars).
    n_vars : int
        Amount of variables.

    Returns
    -------
    res : List of symbolic expressions
        List of simbolic lagrange equations.

    """
    L = T - U
    res = []
    for ii in range(n_vars):
        q = dynamicsymbols(f"q_{ii}")
        res.append(lagrange(L, q))
    return res


def diff_to_symb(expr, n_var):
    """Transform an expression with derivatives to symbols"""
    t = symbols("t")
    for jj in range(n_var):
        q = dynamicsymbols(f"q_{jj}")
        expr = expr.subs(
            ([q.diff(t, 2), symbols(f"a_{jj}")], [q.diff(t), symbols(f"v_{jj}")])
        )
    return expr


def lagr_to_RHS(lagr_eqs):
    """
    Takes lagrangian equations,
    Calculates the Right Hand Side functions of
    the second order derivatives as used in optimal control

    Parameters
    ----------
    lagr_eqs : List of symbolic expressions
        Lagrange Equations.

    Returns
    -------
    Symbolic Matrix of Nx1 size
        Contains the symbolic expressions of the right hand side
        of system equations for the second derivative of coordinates
        as used in optimal control problems

    """
    n_var = len(lagr_eqs)
    coeff_mat = []
    acc_mat = []
    c_mat = []
    u_mat = []
    for ii in range(n_var):
        expr = diff_to_symb(lagr_eqs[ii], n_var)
        coeff_line = []
        rest = expr
        for jj in range(n_var):
            a = symbols(f"a_{jj}")
            coeff_line.append(expr.collect(a).coeff(a))
            rest = rest - a * expr.collect(a).coeff(a)
        coeff_mat.append(coeff_line)
        acc_mat.append(
            [symbols(f"a_{ii}"),]
        )
        u_mat.append(
            [symbols(f"u_{ii}"),]
        )
        c_mat.append(
            [simplify(rest),]
        )
    coeff_mat = Matrix(coeff_mat)
    acc_mat = Matrix(acc_mat)
    c_mat = Matrix(c_mat)
    u_mat = Matrix(u_mat)
    RHS = simplify(coeff_mat.inv() @ (u_mat - c_mat))
    new_RHS = []
    for expr in RHS:
        for jj in range(n_var):
            expr = expr.subs(dynamicsymbols(f"q_{jj}"), symbols(f"q_{jj}"))
        new_RHS.append(expr)
    return Matrix(new_RHS)


def print_funcs(RHS, n_var):
    """
    Prints the Right Hand Side of the control ecuations, formatted
    to be used as python functions.

    Parameters
    ----------
    RHS : Matrix or list of symbolic expressions
        
    n_var : int
        Number of variables

    Returns
    -------
    string
        when outputted by print(), can be copypasted to define functions
        associated with RHS.

    """
    RHS = list(RHS)
    msg_total = ""
    for ii in range(len(RHS)):
        expr = RHS[ii]
        var_set = expr.atoms(Symbol)
        f_args = []
        for jj in range(n_var):
            q = symbols(f"q_{jj}")
            f_args.append(q)
            if q in var_set:
                var_set.remove(q)
        for jj in range(n_var):
            v = symbols(f"v_{jj}")
            f_args.append(v)
            if v in var_set:
                var_set.remove(v)
        for jj in range(n_var):
            u = symbols(f"u_{jj}")
            f_args.append(u)
            if u in var_set:
                var_set.remove(u)
        f_args = f_args + list(var_set)
        msg = f"def f_{ii}({f_args.__str__()[1:-1]}):\n"
        msg += "    result = " + expr.__str__()
        msg += "\n    return result\n"

        print(msg)
        msg_total += msg + "\n"
    return msg_total
