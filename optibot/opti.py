#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:03:53 2022

@author: smorenom
"""

from .casadi import rhs_to_casadi_function, find_arguments
import casadi as cas
from numpy import array, linspace
from .pseudospectral import (
    base_points,
    coll_points,
    matrix_D_bary,
    LG_end_p_fun_cas,
    LG_inv_diff_start_p_fun_cas,
    get_bary_extreme_f,
)
from .casadi import sympy2casadi
from Sympy import symbols

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
    dynamics residual functions. They have a blind argument "lam" that allows them
    to share structure with implicit dinamic functions that require lagrange multipliers.
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
    Casadi Function D(x, x', u, lag, params) so that
        D = x'- F(x, u, params)
        
    Casadi Function D(x, x', u, lag, params) so that
        D = x'[bottom] - G(x, u, params)
        
    Casadi Function D(q, q', q'', u, lam, params) so that
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
    lam_sym = cas.SX.sym("lam", 0)

    dynam_f_x = cas.Function(
        "dynamics_f_x",
        [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
        [x_dot_sym.T - F_cas_x(x_sym, u_sym, p_sym)],
        ["x", "x_dot", "u", "lam", "params"],
        ["residue"],
    )

    dynam_g_x = cas.Function(
        "dynamics_g_x",
        [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
        [x_dot_sym[q_len:].T - F_cas_x(x_sym, u_sym, p_sym)[q_len:]],
        ["x", "x_dot", "u", "lam", "params"],
        ["residue"],
    )

    dynam_g_q = cas.Function(
        "dynamics_g_q",
        [q_sym, q_dot_sym, q_dot_dot_sym, u_sym, lam_sym, p_sym],
        [q_dot_dot_sym.T - G_cas_q(q_sym, q_dot_sym, u_sym, p_sym)],
        ["q", "q_dot", "q_dot_dot", "u", "lam", "params"],
        ["residue"],
    )

    return dynam_f_x, dynam_g_x, dynam_g_q


# --- Pseudospectral functions


def _get_cost_obj_trap_int(scheme, N):
    t_arr = (
        [-1,] + coll_points(N, scheme) + [1,]
    )
    t_arr = [float(ii) for ii in t_arr]
    start_p_f = get_bary_extreme_f(scheme, N, mode="u", point="start")
    end_p_f = get_bary_extreme_f(scheme, N, mode="u", point="end")

    def obj_f(coefs):
        start_p = start_p_f(coefs)
        end_p = end_p_f(coefs)
        coef_list = [start_p] + [coefs[jj] for jj in range(N)] + [end_p]
        sum_res = 0
        for jj in range(N + 1):
            sum_res += (
                (coef_list[jj] ** 2 + coef_list[jj + 1] ** 2)
                * (t_arr[jj + 1] - t_arr[jj])
                / 2
            )
        return sum_res

    return obj_f


def _get_cost_obj_trap_int_cas(scheme, N):
    u_sym = cas.SX.sym("u", N)
    u_sympy = symbols(f"c0:{N}")
    fun = _get_cost_obj_trap_int(scheme, N)
    sympy_expr = fun(u_sympy)
    cas_expr = sympy2casadi(sympy_expr, u_sympy, cas.vertsplit(u_sym))
    cas_f = cas.Function("cost_func", [u_sym,], [cas_expr,])
    return cas_f


# --- Opti problem


class _Opti_Problem:
    def __init__(
        self,
        LM,
        params,
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
        self.params = params
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

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        self.col_points = col_points

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
        self.N = N

        D_mat = sympy2casadi(matrix_D_bary(N, scheme, precission), [], [])
        self.D_mat = D_mat

        try:
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
                raise NotImplementedError(
                    f"scheme {scheme} not implemented in opti_setup."
                )
        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        u_opti = opti.variable(col_points, self.n_u)
        tau_arr = array(base_points(N, scheme, precission), dtype=float)
        col_arr = array(coll_points(col_points, scheme, precission), dtype=float)
        t_arr = ((t_end - t_start) * tau_arr + (t_start + t_end)) / 2
        t_col_arr = ((t_end - t_start) * col_arr + (t_start + t_end)) / 2
        lam_opti = opti.variable(col_points, self.n_lambdas)

        self.opti_arrs = {
            "x": x_opti,
            "x_d": x_dot_opti,
            "q": q_opti,
            "v": v_opti,
            "a": a_opti,
            "u": u_opti,
            "lam": lam_opti,
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

    def u_sq_cost(self):
        try:
            U = self.opti_arrs["u"]
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before defining cost, use opti_setup()"
            )

        dt = self.t_end - self.t_start

        f_u_cost = _get_cost_obj_trap_int_cas(self.scheme, self.col_points)
        cost = dt * cas.sum2(f_u_cost(U))

        self.cost = cost
        self.opti.minimize(cost)

    def apply_scheme(self):
        scheme = self.scheme
        try:
            N = self.N
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )
        dynam_f_x = self.dyn_f_restr
        dynam_g_q = self.dyn_g_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        q_opti = self.opti_arrs["q"]
        v_opti = self.opti_arrs["v"]
        a_opti = self.opti_arrs["a"]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params

        if scheme == "LGL":
            for ii in range(N):
                self.opti.subject_to(
                    dynam_f_x(
                        x_opti[ii, :],
                        x_dot_opti[ii, :],
                        u_opti[ii, :],
                        lam_opti[ii, :],
                        params,
                    )
                    == 0
                )
        elif scheme == "LG":
            for ii in range(1, N):
                self.opti.subject_to(
                    dynam_f_x(
                        x_opti[ii, :],
                        x_dot_opti[ii, :],
                        u_opti[ii - 1, :],
                        lam_opti[ii - 1, :],
                        params,
                    )
                    == 0
                )
        elif scheme == "D2":
            for ii in range(N):
                self.opti.subject_to(
                    dynam_g_q(
                        q_opti[ii, :],
                        v_opti[ii, :],
                        a_opti[ii, :],
                        u_opti[ii, :],
                        lam_opti[ii, :],
                        params,
                    )
                    == 0
                )
        elif scheme == "LG2":
            for ii in range(1, N - 1):
                self.opti.subject_to(
                    dynam_g_q(
                        q_opti[ii, :],
                        v_opti[ii, :],
                        a_opti[ii, :],
                        u_opti[ii - 1, :],
                        lam_opti[ii - 1, :],
                        params,
                    )
                    == 0
                )
        elif scheme == "LGLm":
            for ii in range(1, N - 1):
                self.opti.subject_to(
                    dynam_g_q(
                        q_opti[ii, :],
                        v_opti[ii, :],
                        a_opti[ii, :],
                        u_opti[ii - 1, :],
                        lam_opti[ii - 1, :],
                        params,
                    )
                    == 0
                )
        else:
            raise NotImplementedError(
                f"scheme {scheme} not implemented in apply_scheme."
            )


class _Equispaced:
    def opti_setup(self, segment_number):

        N = segment_number
        self.N = N
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

        try:
            x_opti = opti.variable(N + 1, 2 * self.n_q)
            x_dot_opti = opti.variable(N + 1, 2 * self.n_q)
            u_opti = opti.variable(N + 1, self.n_u)
            q_opti = x_opti[:, : self.n_q]
            v_opti = x_opti[:, self.n_q :]
            a_opti = x_dot_opti[:, self.n_q :]
            t_arr = linspace(t_start, t_end, N + 1)
        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        lam_opti = opti.variable(N + 1, self.n_lambdas)

        self.opti_arrs = {
            "x": x_opti,
            "x_d": x_dot_opti,
            "q": q_opti,
            "v": v_opti,
            "a": a_opti,
            "u": u_opti,
            "t": t_arr,
            "lam": lam_opti,
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
            lam_c_opti = opti.variable(N, self.n_lambdas)
            self.opti_arrs = {
                **self.opti_arrs,
                "x_c": x_c_opti,
                "x_d_c": x_dot_c_opti,
                "q_c": q_c_opti,
                "v_c": v_c_opti,
                "a_c": a_c_opti,
                "u_c": u_c_opti,
                "t_c": t_c_arr,
                "lam_c": lam_c_opti,
            }

    def u_sq_cost(self):
        try:
            U = self.opti_arrs["u"]
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before defining cost, use opti_setup()"
            )

        dt = self.t_end - self.t_start

        try:
            U_c = self.opti_arrs["u_c"]
            cost = dt * cas.sum2(
                (
                    4 * cas.sum1(U_c[:, :] ** 2)
                    + cas.sum1(U[:, :] ** 2)
                    + cas.sum1(U[1:-1, :] ** 2)
                )
                / (3 * self.N)
            )
        except AttributeError:
            cost = dt * cas.sum2(
                (cas.sum1(U[:, :] ** 2) + cas.sum1(U[1:-1, :] ** 2)) / self.N
            )
        self.cost = cost
        self.opti.minimize(cost)

    def apply_scheme(self):
        from .casadi import accelrestriction2casadi
        from .schemes import (
            euler_accel_restr,
            trapz_accel_restr,
            trapz_mod_accel_restr,
            hs_mod_accel_restr,
            hs_accel_restr,
            hs_half_x,
        )

        scheme = self.scheme
        try:
            N = self.N
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )
        T = self.t_end - self.t_start
        dynam_f_x = self.dyn_f_restr
        dynam_g_q = self.dyn_g_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        q_opti = self.opti_arrs["q"]
        v_opti = self.opti_arrs["v"]
        a_opti = self.opti_arrs["a"]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params
        if "hs" in scheme:
            x_c_opti = self.opti_arrs["x_c"]
            x_c_dot_opti = self.opti_arrs["x_d_c"]
            u_c_opti = self.opti_arrs["u_c"]
            a_c_opti = self.opti_arrs["a_c"]
            lam_c_opti = self.opti_arrs["lam_c"]
            if "mod" in scheme:
                from .schemes import hs_mod_half_x

                half_x = hs_mod_half_x
            else:
                from .schemes import hs_half_x

                half_x = hs_half_x

        # Dynamics Constraints:
        for ii in range(N + 1):
            self.opti.subject_to(
                dynam_f_x(
                    x_opti[ii, :],
                    x_dot_opti[ii, :],
                    u_opti[ii, :],
                    lam_opti[ii, :],
                    params,
                )
                == 0
            )
        if "hs" in scheme:
            for ii in range(N):
                self.opti.subject_to(
                    x_c_opti[ii, :]
                    == half_x(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        x_dot_opti[ii, :],
                        x_dot_opti[ii + 1, :],
                        T / N,
                    )
                )
                self.opti.subject_to(
                    dynam_f_x(
                        x_c_opti[ii, :],
                        x_c_dot_opti[ii, :],
                        u_c_opti[ii, :],
                        lam_c_opti[ii, :],
                        params,
                    )
                    == 0
                )
            if "parab" not in scheme:
                for ii in range(N):
                    self.opti.subject_to(
                        u_c_opti[ii, :] == (u_opti[ii, :] + u_opti[ii + 1, :]) / 2
                    )

        # Scheme Constraints
        restr_schemes = {
            #'euler': euler_accel_restr, #comprobar compatibilidad
            "trapz": trapz_accel_restr,
            "trapz_mod": trapz_mod_accel_restr,
            "hs": hs_accel_restr,
            "hs_mod": hs_mod_accel_restr,
            "hs_parab": hs_accel_restr,
            "hs_mod_parab": hs_mod_accel_restr,
        }
        n_q = self.n_q
        f_restr = restr_schemes[scheme]
        if "hs" in scheme:
            cas_accel_restr = accelrestriction2casadi(f_restr, n_q, n_q)
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        a_opti[ii, :],
                        a_opti[ii + 1, :],
                        T / N,
                        a_c_opti[ii, :],
                    )
                    == 0
                )
        else:
            cas_accel_restr = accelrestriction2casadi(f_restr, n_q)
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        a_opti[ii, :],
                        a_opti[ii + 1, :],
                        T / N,
                        [],
                    )
                    == 0
                )


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
        self.n_lambdas = 0


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
        self.n_lambdas = dynam_f_x.mx_in()[3].shape[0]


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
