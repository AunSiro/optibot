#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:03:53 2022

@author: smorenom
"""

from .casadi import rhs_to_casadi_function, find_arguments
import casadi as cas
from numpy import array, linspace

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
    def opti_setup(
        self, col_points, precission=20,
    ):
        from .pseudospectral import (
            base_points,
            coll_points,
            matrix_D_bary,
            LG_end_p_fun_cas,
            LG_inv_diff_start_p_fun_cas,
        )
        from .casadi import sympy2casadi

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end

        opti = cas.Opti()
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        self.opti = opti

        opt_dict = {
            "LGL": [col_points,],
            "D2": [col_points,],
            "LG2": [col_points + 2,],
            "LGLm": [col_points + 2,],
            "LG": [col_points + 1,],
        }
        N = opt_dict[scheme][0]

        D_mat = sympy2casadi(matrix_D_bary(N, scheme, precission), [], [])
        self.D_mat = D_mat

        if scheme in ["LGL", "LG"]:
            x_opti = opti.variable(N, 2 * self.n_q)
            x_dot_opti = 2 / (t_end - t_start) * D_mat @ x_opti
            q_opti = x_opti[:, : self.n_q]
            v_opti = x_opti[:, self.n_q :]
            a_opti = x_dot_opti[:, self.n_q :]

        elif scheme in ["LG2", "D2", "LGLm"]:
            q_opti = opti.variable(N, self.n_q)
            v_opti = 2 / (t_end - t_start) * D_mat @ q_opti
            a_opti = 2 / (t_end - t_start) * D_mat @ v_opti
            x_opti = cas.horzcat(q_opti, v_opti)
            x_dot_opti = cas.horzcat(v_opti, a_opti)
        else:
            raise NotImplementedError(f"scheme {scheme} not implemented in opti_setup.")

        u_opti = opti.variable(col_points, self.n_u)
        tau_arr = array(base_points(N, scheme, precission), dtype=float)
        col_arr = array(coll_points(col_points, scheme, precission), dtype=float)
        t_arr = ((t_end - t_start) * tau_arr + (t_start + t_end)) / 2
        t_col_arr = ((t_end - t_start) * col_arr + (t_start + t_end)) / 2

        self.opti_arrs = {
            "x": x_opti,
            "x_d": x_dot_opti,
            "q": q_opti,
            "v": v_opti,
            "a": a_opti,
            "u": u_opti,
            "tau": tau_arr,
            "tau_col": col_arr,
            "t": t_arr,
            "t_col": t_col_arr,
        }

        if scheme == "LGinv":
            start_p_func = LG_inv_diff_start_p_fun_cas(N)
            x_start = start_p_func(x_opti)
            x_dot_start = start_p_func(x_dot_opti)
            q_start = start_p_func(q_opti)
            v_start = start_p_func(v_opti)
            a_start = start_p_func(a_opti)
        else:
            x_start = x_opti[0, :]
            x_dot_start = x_dot_opti[0, :]
            q_start = q_opti[0, :]
            v_start = v_opti[0, :]
            a_start = a_opti[0, :]

        if scheme == "LG":
            end_p_func = LG_end_p_fun_cas(N)
            x_end = end_p_func(x_opti)
            x_dot_end = end_p_func(x_dot_opti)
            q_end = end_p_func(q_opti)
            v_end = end_p_func(v_opti)
            a_end = end_p_func(a_opti)
        else:
            x_end = x_opti[-1, :]
            x_dot_end = x_dot_opti[-1, :]
            q_end = q_opti[-1, :]
            v_end = v_opti[-1, :]
            a_end = a_opti[-1, :]

        self.opti_points = {
            "x_s": x_start,
            "x_e": x_end,
            "x_d_s": x_dot_start,
            "x_d_e": x_dot_end,
            "q_s": q_start,
            "q_e": q_end,
            "v_s": v_start,
            "v_e": v_end,
            "a_s": a_start,
            "a_e": a_end,
        }


class _Equispaced:
    def opti_setup(self, segment_number):

        N = segment_number
        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        opti = cas.Opti()
        p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        s_opts = {
            "max_iter": 10000,
            "tol": 1e-26,
        }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        x_opti = opti.variable(N + 1, 2 * self.n_q)
        x_dot_opti = opti.variable(N + 1, 2 * self.n_q)
        u_opti = opti.variable(N + 1, self.n_u)
        q_opti = x_opti[:, : self.n_q]
        v_opti = x_opti[:, self.n_q :]
        a_opti = x_dot_opti[:, self.n_q :]
        t_arr = linspace(t_start, t_end, N + 1)

        self.opti_arrs = {
            "x": x_opti,
            "x_d": x_dot_opti,
            "q": q_opti,
            "v": v_opti,
            "a": a_opti,
            "u": u_opti,
            "t": t_arr,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
            "q_s": q_opti[0, :],
            "q_e": q_opti[-1, :],
            "v_s": v_opti[0, :],
            "v_e": v_opti[-1, :],
            "a_s": a_opti[0, :],
            "a_e": a_opti[-1, :],
        }

        if "hs" in scheme:
            u_c_opti = opti.variable(N, self.n_u)
            x_c_opti = opti.variable(N, 2 * self.n_q)
            x_dot_c_opti = opti.variable(N, 2 * self.n_q)
            q_c_opti = x_c_opti[:, : self.n_q]
            v_c_opti = x_c_opti[:, self.n_q :]
            a_c_opti = x_dot_c_opti[:, self.n_q :]
            t_c_arr = (t_arr[:-1] + t_arr[1:]) / 2
            self.opti_arrs = {
                **self.opti_arrs,
                "x_c": x_c_opti,
                "x_d_c": x_dot_c_opti,
                "q_c": q_c_opti,
                "v_c": v_c_opti,
                "a_c": a_c_opti,
                "u_c": u_c_opti,
                "t_c": t_c_arr,
            }


class _Explicit_Dynamics:
    def dynamic_setup(self, u_vars=None):
        q_vars = list(self.LM.q)
        self.n_q = len(q_vars)
        RHS = self.LM.rhs
        _arguments = find_arguments(RHS, q_vars, u_vars)
        self.params_sym = _arguments[4]
        dynam_f_x, dynam_g_x, dynam_g_q = _get_f_g_funcs(
            RHS, q_vars, u_vars, verbose=self.verbose, silent=self.silent
        )
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = dynam_f_x.mx_in()[2].shape[0]


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
        self.n_q = len(q_vars)

        _arguments = find_arguments(impl_q, q_vars, separate_lambdas=True)
        self.params_sym = _arguments[4]

        dynam_f_x = implicit_dynamic_x_to_casadi_function(
            impl_x, x_vars, verbose=self.verbose, silent=self.silent
        )
        dynam_g_q = implicit_dynamic_q_to_casadi_function(
            impl_q, q_vars, verbose=self.verbose, silent=self.silent
        )
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = dynam_f_x.mx_in()[2].shape[0]


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
