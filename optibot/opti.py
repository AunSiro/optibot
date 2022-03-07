#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:03:53 2022

@author: smorenom
"""

from .casadi import rhs_to_casadi_function
import casadi as cas

_implemented_equispaced_schemes = [
    "euler",
    "trapz",
    "trapz_mod",
    "hs",
    "hs_mod",
    "hs_parab",
    "hs_mod_parab",
]
_implemented_pseudospectral_schemes = [
    "LG",
    "LGL",
    "D2",
    "LG2",
    "LGLm",
]

# --- Explicit Dynamics ---


def _get_f_g_funcs(RHS, q_vars, u_vars=None, verbose=False, silent=True):
    """
    Converts an array of symbolic expressions RHS(x, u, params) to the 3 casadi 
    dynamics residual functions.
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
    verbose : Bool, optional, default = False
        wether to print aditional information of expected and found variables
        in the given expression
    silent: Bool, optional, default False
        if True, nothing will be printed. If verbose = True, silent will be ignored.

    Returns
    -------
    Casadi Function D(x, x', u, params) so that
        D = x'- F(x, u, params)
        
    Casadi Function D(x, x', u, params) so that
        D = x'[bottom] - G(x, u, params)
        
    Casadi Function D(q, q', q'', u, params) so that
        D = q'' - G(q, q', u, params)

    """
    if verbose:
        silent = False
    q_type = type(q_vars)
    if q_type == int:
        q_len = q_vars
    elif q_type == list or q_type == tuple:
        q_len = len(q_vars)
    else:
        raise ValueError("q_vars type not supported. Valid types are int and list")
    if not silent:
        print("Generating F function")
    F_cas_x = rhs_to_casadi_function(
        RHS, q_vars, u_vars, verbose, mode="x", silent=silent
    )
    if not silent:
        print("Generating G function")
    G_cas_q = rhs_to_casadi_function(
        RHS, q_vars, u_vars, verbose, mode="q", silent=silent
    )
    u_len = F_cas_x.mx_in()[1].shape[0]
    p_len = F_cas_x.mx_in()[2].shape[0]

    x_sym = cas.SX.sym("x", q_len * 2)
    x_dot_sym = cas.SX.sym("x_dot", q_len * 2)
    q_sym = cas.SX.sym("q", q_len)
    q_dot_sym = cas.SX.sym("q_dot", q_len)
    q_dot_dot_sym = cas.SX.sym("q_dot_dot", q_len)
    u_sym = cas.SX.sym("u", u_len)
    p_sym = cas.SX.sym("p", p_len)

    dynam_f_x = cas.Function(
        "dynamics_f_x",
        [x_sym, x_dot_sym, u_sym, p_sym],
        [x_dot_sym.T - F_cas_x(x_sym, u_sym, p_sym)],
        ["x", "x_dot", "u", "params"],
        ["residue"],
    )

    dynam_g_x = cas.Function(
        "dynamics_g_x",
        [x_sym, x_dot_sym, u_sym, p_sym],
        [x_dot_sym[q_len:].T - F_cas_x(x_sym, u_sym, p_sym)[q_len:]],
        ["x", "x_dot", "u", "params"],
        ["residue"],
    )

    dynam_g_q = cas.Function(
        "dynamics_g_q",
        [q_sym, q_dot_sym, q_dot_dot_sym, u_sym, p_sym],
        [q_dot_dot_sym.T - G_cas_q(q_sym, q_dot_sym, u_sym, p_sym)],
        ["q", "q_dot", "q_dot_dot", "u", "params"],
        ["residue"],
    )

    return dynam_f_x, dynam_g_x, dynam_g_q


class _Opti_Problem:
    def __init__(
        self,
        LM,
        scheme="trapz",
        ini_guess="zero",
        solve_repetitions=1,
        t_start=0,
        t_end=1,
        verbose=False,
        silent=True,
    ):
        if scheme in _implemented_equispaced_schemes:
            self.scheme_mode = "equispaced"
        elif scheme in _implemented_pseudospectral_schemes:
            self.scheme_mode = "pseudospectral"
        else:
            _v = _implemented_equispaced_schemes + _implemented_pseudospectral_schemes
            raise NotImplementedError(
                f"scheme {scheme} not implemented. Valid methods are {_v}."
            )
        self.LM = LM
        self.scheme = scheme
        self.ini_guess = ini_guess
        self.solve_repetitions = solve_repetitions
        self.t_start = t_start
        self.t_end = t_end
        self.verbose = verbose
        self.silent = silent

        self.LM.form_lagranges_equations()


class _Pseudospectral:
    pass


class _Equispaced:
    pass


class _Explicit_Dynamics:
    def dynamic_setup(self, u_vars=None):
        q_vars = list(self.LM.q)
        RHS = self.LM.rhs
        dynam_f_x, dynam_g_x, dynam_g_q = _get_f_g_funcs(
            RHS, q_vars, u_vars, verbose=self.verbose, silent=self.silent
        )
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q


class _Implicit_Dynamics:
    def dynamic_setup(self, u_vars=None):
        from .casadi import (
            implicit_dynamic_q_to_casadi_function,
            implicit_dynamic_x_to_casadi_function,
        )
        from .symbolic import q_2_x

        impl_x = self.LM.implicit_dynamics_x
        impl_q = self.LM.implicit_dynamics_q

        q_vars = list(self.LM.q)
        q_dot_vars = list(self.LM._qdots)
        x_vars = [q_2_x(ii, q_vars, q_dot_vars) for ii in q_vars + q_dot_vars]

        dynam_f_x = implicit_dynamic_x_to_casadi_function(
            impl_x, x_vars, verbose=self.verbose, silent=self.silent
        )
        dynam_g_q = implicit_dynamic_q_to_casadi_function(
            impl_q, q_vars, verbose=self.verbose, silent=self.silent
        )
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q


class Pseudospectral_Explicit_Opti_Problem(
    _Opti_Problem, _Pseudospectral, _Explicit_Dynamics
):
    pass


class Equispaced_Explicit_Opti_Problem(_Opti_Problem, _Equispaced, _Explicit_Dynamics):
    pass


class Pseudospectral_Implicit_Opti_Problem(
    _Opti_Problem, _Pseudospectral, _Implicit_Dynamics
):
    pass


class Equispaced_Implicit_Opti_Problem(_Opti_Problem, _Equispaced, _Implicit_Dynamics):
    pass


def Opti_Problem(
    LM,
    scheme="trapz",
    ini_guess="zero",
    solve_repetitions=1,
    t_start=0,
    t_end=1,
    verbose=False,
    silent=True,
):
    from .symbolic import ImplicitLagrangesMethod, SimpLagrangesMethod
    from sympy.physics.mechanics import LagrangesMethod

    if isinstance(LM, ImplicitLagrangesMethod):
        dynamics_mode = "implicit"
    elif isinstance(LM, (SimpLagrangesMethod, LagrangesMethod)):
        dynamics_mode = "explicit"
    else:
        ValueError(f"LM must be a Lagranges Method object, not: {type(LM)}")

    if scheme in _implemented_equispaced_schemes:
        if dynamics_mode == "explicit":
            return Equispaced_Explicit_Opti_Problem()
        elif dynamics_mode == "implicit":
            return Equispaced_Implicit_Opti_Problem()
    elif scheme in _implemented_pseudospectral_schemes:
        if dynamics_mode == "explicit":
            return Pseudospectral_Explicit_Opti_Problem()
        elif dynamics_mode == "implicit":
            return Pseudospectral_Implicit_Opti_Problem()
    else:
        _v = _implemented_equispaced_schemes + _implemented_pseudospectral_schemes
        raise NotImplementedError(
            f"scheme {scheme} not implemented. Valid methods are {_v}."
        )
