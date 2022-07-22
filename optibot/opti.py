#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:03:53 2022

@author: Siro Moreno

Here we present some functions and classes designed to contain and interface
easily a casadi opti problem.

In order to create the object, the factory function Opti_problem must be called,
which depending on the parameters given will select the appropiate classes 
from which a new class must inherit. Then, it will return an instance
of the custom class created.

The class will inherit:
    - From _Opti_Problem, always. Contains methos common to all problems.
    
    - Depending on the scheme: from _Pseudospectral or _Equispaced. They
      contain the methods that change structure depending on wether the
      problem use a pseudospectral collocation or an equispaced scheme.
      
    - Depending on the physics: from _Explicit_Dynamics, _Implicit_Dynamics or
      _Function_Dynamics. First one will be used when the physics are passed
      as an instance of a Lagranges Method or Simplifified Lagranges Method
      object. Second will be used when they ar passed as an instance of
      Implicit Lagranges Method object. Third will be used when they are 
      passed as a function.
      
    - Depending on initial guess type: _Zero_init, _Lin_init or _Custom_init.
      They contain the function appropiate for the chosen kind of initial guess.
"""

from .symbolic import find_arguments
from .casadi import sympy2casadi, rhs_to_casadi_function
from .pseudospectral import (
    base_points,
    coll_points,
    matrix_D_bary,
    LG_end_p_fun_cas,
    LG_inv_diff_start_p_fun_cas,
    get_bary_extreme_f,
)

import casadi as cas
from numpy import array, linspace, expand_dims, ones
from sympy import symbols
from time import time
from functools import lru_cache

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
    dynamics residual functions. They have a blind argument "lamba" that allows them
    to share structure with implicit dinamic functions that require lagrange multipliers.
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
    silent: Bool, optional, default False
        if True, nothing will be printed. If verbose = True, silent will be ignored.

    Returns
    -------
    Casadi Function D(x, x', u, lamda, params) so that
        D = x'- F(x, u, params)
        
    Casadi Function D(x, x', u, lamda, params) so that
        D = x'[bottom] - G(x, u, params)
        
    Casadi Function D(q, q', q'', u, lamda, params) so that
        D = q'' - G(q, q', u, params)

    """
    if verbose:
        silent = False
    q_type = type(q_vars)
    if q_type == int:
        n_q = q_vars
    elif q_type == list or q_type == tuple:
        n_q = len(q_vars)
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
    n_u = F_cas_x.mx_in()[1].shape[0]
    n_p = F_cas_x.mx_in()[2].shape[0]
    n_lambdas = 0

    x_sym = cas.SX.sym("x", n_q * 2).T
    x_dot_sym = cas.SX.sym("x_dot", n_q * 2).T
    q_sym = cas.SX.sym("q", n_q).T
    q_dot_sym = cas.SX.sym("q_dot", n_q).T
    q_dot_dot_sym = cas.SX.sym("q_dot_dot", n_q).T
    u_sym = cas.SX.sym("u", n_u).T
    p_sym = cas.SX.sym("p", n_p)
    lam_sym = cas.SX.sym("lam", n_lambdas).T

    _F = F_cas_x(x_sym, u_sym, p_sym)
    _G = G_cas_q(q_sym, q_dot_sym, u_sym, p_sym)
    if _F.shape[0] != 1:
        _F = _F.T
    if _G.shape[0] != 1:
        _G = _G.T

    dynam_f_x = cas.Function(
        "dynamics_f_x",
        [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
        [x_dot_sym - _F],
        ["x", "x_dot", "u", "lamda", "params"],
        ["residue"],
    )

    dynam_g_x = cas.Function(
        "dynamics_g_x",
        [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
        [x_dot_sym[n_q:] - _F[n_q:]],
        ["x", "x_dot", "u", "lamda", "params"],
        ["residue"],
    )

    dynam_g_q = cas.Function(
        "dynamics_g_q",
        [q_sym, q_dot_sym, q_dot_dot_sym, u_sym, lam_sym, p_sym],
        [q_dot_dot_sym - _G],
        ["q", "q_dot", "q_dot_dot", "u", "lamda", "params"],
        ["residue"],
    )

    return dynam_f_x, dynam_g_x, dynam_g_q


# --- Pseudospectral functions


@lru_cache(maxsize=None)
def _get_cost_obj_trap_int(scheme, N):
    """ For a given pseudospectral scheme and number of collocation points,
    returns a function of values in said points that calculates a trapezoidal
    integration of the squared values.
    """
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


@lru_cache(maxsize=None)
def _get_cost_obj_trap_int_cas(scheme, N):
    """ For a given pseudospectral scheme and number of collocation points,
    returns a casadi function of values in said points that calculates a 
    trapezoidal integration of the squared values.
    """
    u_sym = cas.SX.sym("u", N)
    u_sympy = symbols(f"c0:{N}")
    fun = _get_cost_obj_trap_int(scheme, N)
    sympy_expr = fun(u_sympy)
    cas_expr = sympy2casadi(sympy_expr, u_sympy, cas.vertsplit(u_sym))
    cas_f = cas.Function("cost_func", [u_sym,], [cas_expr,])
    return cas_f


# --- Opti problem


class _Opti_Problem:
    """
    An object that contains a casadi opti problem.
    
    Use the methods in this order:
        
        problem.dynamic_setup()
        problem.opti_setup()
        problem.apply_scheme()
        
        additional restrictions and functions, such as:
            problem.u_sq_cost() [apply a u squared integral cost]
            problem.opti.subject_to(conditions)
            
        problem.simple_solve() or problem.chrono_solve()
        
    Important points and arrays of the opti problem generated after opti_setup()
    are stored at problem.opti_arrs and problem.opti_points
    
    Results obtained after solving are stored at problem.results
    """

    def __init__(
        self,
        LM,
        params,
        scheme="trapz",
        ini_guess="zero",
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
        self.t_start = t_start
        self.t_end = t_end
        self.verbose = verbose
        self.silent = silent

        try:
            self.LM.form_lagranges_equations()
        except AttributeError:
            pass

    def _ini_guess_start(self):
        """ Tries to find optimization arrays in the object
        """
        try:
            q_opti = self.opti_arrs["q"]
            v_opti = self.opti_arrs["v"]
            a_opti = self.opti_arrs["a"]
        except AttributeError:
            raise RuntimeError(
                "opti problem must be setup, use opti_setup() and apply_scheme()"
            )
        return q_opti, v_opti, a_opti

    def _save_results(self):
        """Saves results of optimization in dictionary 'results'
        """
        for key in self.opti_arrs.keys():
            opti_arr = self.opti_arrs[key]
            self.results[key] = self.sol.value(opti_arr)

    def simple_solve(self):
        """
        Calculate the solution of opti problem

        Raises
        ------
        RuntimeError
            If the opti problem has not been properly set up

        Returns
        -------
        None.

        """
        try:
            sol = self.opti.solve()
        except AttributeError:
            raise RuntimeError(
                "opti problem must be setup, use opti_setup() and apply_scheme()"
            )
        cpudt = None
        self.sol = sol
        self.results = {"cpudt": cpudt}
        self._save_results()

    def chrono_solve(self, solve_repetitions):
        """
        Calculate the solution of opti problem repetedly, measuring the 
        time required to do so.

        Parameters
        ----------
        solve_repetitions : int, optional. The default is 1.
            Number of times solve() is called.

        Raises
        ------
        RuntimeError
            If the opti problem has not been properly set up

        Returns
        -------
        None.

        """

        cput0 = time()
        try:
            for ii in range(solve_repetitions):
                sol = self.opti.solve()
        except AttributeError:
            raise RuntimeError(
                "opti problem must be setup, use opti_setup() and apply_scheme()"
            )
        cput1 = time()
        cpudt = (cput1 - cput0) / solve_repetitions
        self.sol = sol
        self.results = {"cpudt": cpudt, "cost": sol.value(self.cost)}
        self._save_results()


class _Pseudospectral:
    def opti_setup(
        self, col_points, precission=20,
    ):
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Points will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        col_points : int
            Number of collocation points
        precission : int, optional
            Precission decimals in collocation point computation.
            The default is 20.

        Raises
        ------
        NotImplementedError
            If the selected scheme is not yet available
        RuntimeError
            If mistakenly this function is run before dynamic_setup()

        Returns
        -------
        None.

        """

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        self.col_points = col_points

        opti = cas.Opti()
        if self.verbose:
            opts = {}
        else:
            opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        self.opti = opti

        opt_dict = {
            "LGL": [col_points,],
            "D2": [col_points,],
            "LG2": [col_points + 2,],
            "LGLm": [col_points + 2,],
            "LG": [col_points + 1,],
            "LGR": [col_points + 1,],
            "LGR_inv": [col_points + 1,],
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
        """
        Calculates a trapezoidal integration of u squared and sets it 
        as the optimization cost to minimize
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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
        """
        Applies the restrictions corresponding to the selected scheme to
        the opti variables.
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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


class _Pseudospectral_multi:
    def opti_setup(
        self, col_points, segment_num, precission=20,
    ):
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Points will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        col_points : int
            Number of collocation points in each segment
        segment_num : int
            Number of segments
        precission : int, optional
            Precission decimals in collocation point computation.
            The default is 20.

        Raises
        ------
        NotImplementedError
            If the selected scheme is not yet available
        RuntimeError
            If mistakenly this function is run before dynamic_setup()

        Returns
        -------
        None.

        """

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        self.col_points = col_points

        opti = cas.Opti()
        if self.verbose:
            opts = {}
        else:
            opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)
        self.opti = opti

        opt_dict = {
            "LGL_m": [col_points,],
            "D2_m": [col_points,],
            "LG2_m": [col_points + 2,],
            "LG_m": [col_points + 1,],
        }
        N = opt_dict[scheme][0]
        self.N = N

        D_mat = sympy2casadi(matrix_D_bary(N, scheme, precission), [], [])
        self.D_mat = D_mat

        try:
            if scheme in ["LGL_m", "LG_m"]:
                x_opti = opti.variable(N, 2 * self.n_q)
                x_dot_opti = 2 / (t_end - t_start) * D_mat @ x_opti
                q_opti = x_opti[:, : self.n_q]
                v_opti = x_opti[:, self.n_q :]
                a_opti = x_dot_opti[:, self.n_q :]

            elif scheme in [
                "LG2_m",
                "D2_m",
            ]:
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
        """
        Calculates a trapezoidal integration of u squared and sets it 
        as the optimization cost to minimize
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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
        """
        Applies the restrictions corresponding to the selected scheme to
        the opti variables.
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Point will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        segment_number : int
            Number of equal seegments in which solution is divided.
        

        Raises
        ------
        RuntimeError
            If mistakenly this function is run before dynamic_setup()

        Returns
        -------
        None.

        """

        N = segment_number
        self.N = N
        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        opti = cas.Opti()
        if self.verbose:
            p_opts = {
                "expand": True,
            }
        else:
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
        """
        Calculates a trapezoidal integration of u squared and sets it 
        as the optimization cost to minimize
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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
        except KeyError:
            cost = dt * cas.sum2(
                (cas.sum1(U[:, :] ** 2) + cas.sum1(U[1:-1, :] ** 2)) / self.N
            )
        self.cost = cost
        self.opti.minimize(cost)

    def apply_scheme(self):
        """
        Applies the restrictions corresponding to the selected scheme to
        the opti variables.
        
        Requires the functions dynamic_setup() and opti_setup(), in that order,
        to have been run prior.

        Raises
        ------
        RuntimeError
            If opti_setup() or dynamic_setup() have not ben run previously

        Returns
        -------
        None.

        """
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
                        a_opti[ii, :],
                        a_opti[ii + 1, :],
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
        """
        Creates and configures the functions that will be used in 
        physics restrictions.

        Parameters
        ----------
        u_vars : None, int or list of symbols. Default is None.
            Symbols of external control actions.
            If None, or int, symbols of the form u_ii where ii is a number
            will be assumed

        Returns
        -------
        None.

        """

        q_vars = list(self.LM.q)
        self.n_q = len(q_vars)

        try:
            dynam_f_x = self.LM.dynam_f_x
            dynam_g_q = self.LM.dynam_g_q
            _arguments = self.LM.arguments
            if not self.silent:
                print("Functions loaded from LM object")
        except AttributeError:
            RHS = self.LM.rhs
            _arguments = find_arguments(RHS, q_vars, u_vars)
            dynam_f_x, dynam_g_x, dynam_g_q = _get_f_g_funcs(
                RHS, q_vars, u_vars, verbose=self.verbose, silent=self.silent
            )
            self.LM.dynam_f_x = dynam_f_x
            self.LM.dynam_g_q = dynam_g_q
            self.LM.arguments = _arguments

        self.params_sym = _arguments[4]
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = dynam_f_x.mx_in()[2].shape[0]
        self.n_lambdas = 0


class _Implicit_Dynamics:
    def dynamic_setup(self, u_vars=None):
        """
        Creates and configures the functions that will be used in 
        physics restrictions.

        Parameters
        ----------
        u_vars : None, int or list of symbols. Default is None.
            Symbols of external control actions.
            If None, or int, symbols of the form u_ii where ii is a number
            will be assumed

        Returns
        -------
        None.

        """

        from .symbolic import q_2_x

        q_vars = list(self.LM.q)
        q_dot_vars = list(self.LM._qdots)
        x_vars = [q_2_x(ii, q_vars, q_dot_vars) for ii in q_vars + q_dot_vars]
        self.n_q = len(q_vars)
        silent = self.silent

        try:
            dynam_f_x = self.LM.dynam_f_x
            dynam_g_q = self.LM.dynam_g_q
            _arguments = self.LM.arguments
            if not silent:
                print("Functions loaded from LM object")
        except AttributeError:
            from .casadi import (
                implicit_dynamic_q_to_casadi_function,
                implicit_dynamic_x_to_casadi_function,
            )

            impl_x = self.LM.implicit_dynamics_x
            impl_q = self.LM.implicit_dynamics_q
            _arguments = find_arguments(impl_q, q_vars, separate_lambdas=True)

            if not silent:
                print("Generating F function")
            dynam_f_x = implicit_dynamic_x_to_casadi_function(
                impl_x, x_vars, verbose=self.verbose, silent=self.silent
            )
            if not silent:
                print("Generating G function")
            dynam_g_q = implicit_dynamic_q_to_casadi_function(
                impl_q, q_vars, verbose=self.verbose, silent=self.silent
            )
            self.LM.dynam_f_x = dynam_f_x
            self.LM.dynam_g_q = dynam_g_q
            self.LM.arguments = _arguments

        self.params_sym = _arguments[4]
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = dynam_f_x.mx_in()[2].shape[0]
        self.n_lambdas = dynam_f_x.mx_in()[3].shape[0]


class _Function_Dynamics:
    def dynamic_setup(self, func_kind, n_q, n_u, n_lambdas=0):
        """
        Creates and configures the functions that will be used in 
        physics restrictions.

        Parameters
        ----------
        func_kind : str in ['f_x', 'g_q', 'f_x_impl', 'g_q_impl']
            states the kind of function that was given as physics model:
                'f_x' if it was F so that x' = F(x, u, params)
                'f_x_imp' if it was F so that 0 = F(x, x', u, params)
                'g_q' if it was G so that q'' = G(q, q', u, params)
                'g_q_imp' if it was G so that 0 = G(q, q', q'', u, params)
        n_q : int
            number of q variables in the problem.
        n_u : int
            number of u variables in the problem.
        n_lambdas : int, optional
            number of lambda variables in the problem. The default is 0.

        Raises
        ------
        NotImplementedError
            if func_kind is not one of the above stated values.

        Returns
        -------
        None.

        """

        from .schemes import expand_G, reduce_F

        if n_lambdas != 0:
            raise NotImplementedError("Problems with lambdas are not implemented yet")

        n_p = len(self.params)

        x_sym = cas.SX.sym("x", n_q * 2).T
        x_dot_sym = cas.SX.sym("x_dot", n_q * 2).T
        q_sym = cas.SX.sym("q", n_q).T
        q_dot_sym = cas.SX.sym("q_dot", n_q).T
        q_dot_dot_sym = cas.SX.sym("q_dot_dot", n_q).T
        u_sym = cas.SX.sym("u", n_u).T
        p_sym = cas.SX.sym("p", n_p)
        lam_sym = cas.SX.sym("lam", n_lambdas).T

        if func_kind == "f_x":
            F_x = self.LM
            G_q = reduce_F(F_x, "casadi")

        elif func_kind == "g_q":
            G_q = self.LM
            F_x = expand_G(G_q, "casadi")

        elif func_kind == "f_x_impl":
            F_restr = self.LM
            x_restr = F_restr(x_sym, x_dot_sym, u_sym, p_sym)
            if x_restr.shape[0] != 1:
                x_restr = x_restr.T

            x_union = cas.horzcat(q_sym, q_dot_sym)
            x_dot_union = cas.horzcat(q_dot_sym, q_dot_dot_sym)

            q_restr = F_restr(x_union, x_dot_union, u_sym, p_sym)
            if q_restr.shape[0] != 1:
                q_restr = q_restr.T
            q_restr = q_restr[:n_q]

        elif func_kind == "g_q_impl":
            G_restr = self.LM
            q_restr = G_restr(q_sym, q_dot_sym, q_dot_dot_sym, u_sym, p_sym)
            if q_restr.shape[0] != 1:
                q_restr = q_restr.T

            x_restr = G_restr(x_sym[:n_q], x_sym[n_q:], x_dot_sym[n_q:], u_sym, p_sym)
            if x_restr.shape[0] != 1:
                x_restr = x_restr.T
            x_restr = cas.horzcat(x_restr, x_dot_sym[:n_q] - x_sym[n_q:])
        else:
            raise NotImplementedError(
                f"function kind {func_kind} not implemented. Implemented kinds are f_x, g_q, f_x_imp and g_q_impl"
            )

        if func_kind in ["f_x", "g_q"]:
            _F = F_x(x_sym, u_sym, p_sym)
            _G = G_q(q_sym, q_dot_sym, u_sym, p_sym)
            if _F.shape[0] != 1:
                _F = _F.T
            if _G.shape[0] != 1:
                _G = _G.T
            x_restr = x_dot_sym - _F
            q_restr = q_dot_dot_sym - _G

        dynam_f_x = cas.Function(
            "dynamics_f_x",
            [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
            [x_restr],
            ["x", "x_dot", "u", "lam", "params"],
            ["residue"],
        )

        dynam_g_q = cas.Function(
            "dynamics_g_q",
            [q_sym, q_dot_sym, q_dot_dot_sym, u_sym, lam_sym, p_sym],
            [q_restr],
            ["q", "q_dot", "q_dot_dot", "u", "lam", "params"],
            ["residue"],
        )
        self.params_sym = symbols(f"p_0:{n_p}")
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = n_u
        self.n_lambdas = n_lambdas
        self.n_q = n_q


class _Zero_init:
    def initial_guess(self):
        """
        Sets initial guess values for q, v and a as zeros.

        Returns
        -------
        None.

        """
        q_opti, v_opti, a_opti = self._ini_guess_start()
        self.opti.set_initial(q_opti, 0)
        if self.scheme not in ["LG2", "D2", "LGLm"]:
            self.opti.set_initial(v_opti, 0)
            if self.scheme_mode == "equispaced":
                self.opti.set_initial(a_opti, 0)


class _Lin_init:
    def initial_guess(self, q_s, q_e):
        """
        Sets initial guess values for q, v and a.
        q is a lineal interpolation between q_s and q_e
        v is a uniform value = (q_e - q_s)/T
        a is zero
        
        If scheme is an Hermite Simpson variation, inizialization is also
        applied to central point values.

        Parameters
        ----------
        q_s : list, array, or float if problem is 1-d
            Starting point.
        q_e : list, array, or float if problem is 1-d
            Ending point.

        Returns
        -------
        None.

        """
        q_opti, v_opti, a_opti = self._ini_guess_start()
        if self.scheme_mode == "equispaced":
            N = self.N + 1  # Number of segments
        else:
            N = self.N  # Number of Node Points
        T = self.t_end - self.t_start
        q_s = array(q_s, dtype=float)
        q_e = array(q_e, dtype=float)
        s_arr = linspace(0, 1, N)
        q_guess = expand_dims(q_s, 0) + expand_dims(s_arr, 1) * expand_dims(
            (q_e - q_s), 0
        )
        q_dot_guess = (q_e - q_s) * ones([N, 1]) / T
        self.opti.set_initial(q_opti, q_guess)
        if self.scheme not in ["LG2", "D2", "LGLm"]:
            self.opti.set_initial(v_opti, q_dot_guess)
            if self.scheme_mode == "equispaced":
                self.opti.set_initial(a_opti, 0)
                if "hs" in self.scheme:
                    self.opti.set_initial(
                        self.opti_arrs["q_c"], (q_guess[:-1, :] + q_guess[1:, :]) / 2
                    )
                    self.opti.set_initial(self.opti_arrs["v_c"], q_dot_guess[1:, :])
                    self.opti.set_initial(self.opti_arrs["a_c"], 0)


class _Custom_init:
    def initial_guess(self, q_guess, v_guess, a_guess, u_guess):
        """
        Sets initial guess values for q, v, a and u.

        If scheme is an Hermite Simpson variation, inizialization is also
        applied to central point values.
        
        Parameters
        ----------
        q_guess : numpy array
            Initial values of q array.
        v_guess : numpy array
            Initial values of v array.
        a_guess : numpy array
            Initial values of a array.
        u_guess : numpy array
            Initial values of u array.

        Returns
        -------
        None.

        """
        q_opti, v_opti, a_opti = self._ini_guess_start()
        u_opti = self.opti_arrs["u"]
        self.opti.set_initial(q_opti, q_guess)
        self.opti.set_initial(u_opti, u_guess)
        if self.scheme not in ["LG2", "D2", "LGLm"]:
            self.opti.set_initial(v_opti, v_guess)
            if self.scheme_mode == "equispaced":
                self.opti.set_initial(a_opti, 0)
                if "hs" in self.scheme:
                    self.opti.set_initial(
                        self.opti_arrs["q_c"], (q_guess[:-1, :] + q_guess[1:, :]) / 2
                    )
                    self.opti.set_initial(
                        self.opti_arrs["v_c"], (v_guess[:-1, :] + v_guess[1:, :]) / 2
                    )
                    self.opti.set_initial(
                        self.opti_arrs["a_c"], (a_guess[:-1, :] + a_guess[1:, :]) / 2
                    )
                    self.opti.set_initial(
                        self.opti_arrs["u_c"], (u_guess[:-1, :] + u_guess[1:, :]) / 2
                    )


def Opti_Problem(
    LM,
    params,
    scheme="trapz",
    ini_guess="zero",
    t_start=0,
    t_end=1,
    verbose=False,
    silent=True,
):
    """
    Creates an object that contains a casadi opti problem.
    
    Use the methods in this order:
        
        problem.dynamic_setup()
        problem.opti_setup()
        problem.apply_scheme()
        
        additional restrictions and functions, such as:
            problem.u_sq_cost() [apply a u squared integral cost]
            problem.opti.subject_to(conditions)
            
        problem.simple_solve() or problem.chrono_solve()
        
    Important points and arrays of the opti problem generated after opti_setup()
    are stored at problem.opti_arrs and problem.opti_points
    
    Results obtained after solving are stored at problem.results

    Parameters
    ----------
    LM : symbolic Lagranges Method object, or function
        If possible, use a symbolic Lagranges Method object. A function F so that
        x' = F(x, u, params) or G so that q'' = g(q, q', u, params) are also supported.
    params : list of numerical parameters
        Contains the values of the parameters of the problem
    scheme : str
        Discretization scheme. The default is "trapz". Acceptable values are:
            
            "trapz" : trapezoidal scheme 
            "trapz_mod": modified 2nd order compatible trapezoidal scheme 
            "hs": Hermite-Simpson scheme 
            "hs_mod": modified 2nd order compatible Hermite-Simpson scheme 
            "hs_parab": Hermite-Simpson scheme compatible with parabolic U
            "hs_mod_parab": 2nd order compatible Hermite-Simpson scheme with parabolic U
            
            "LG" Legendre-Gauss Collocation
            "LG_inv" LG Colocation with enpoint instead of startpoint as node point
            "LGR" Legendre-Gauss-Radau Collocation
            "LGR_inv" LGR with endpoint instead of start point as collocation point
            "LGL" Legendre-Gauss-Lobato Collocation
            "LGLm" 2nd order modified LGL that only uses interior points as collocation
            "LG2" 2nd order modified LG that adds endpoint as node point
            "D2" 2nd order modified LGL 
    ini_guess : ["zero", "lin", "custom"] The default is "zero".
        initial guess strategy for the optimization. Valid values are:
            "zero": All arrays initialised as zeroes.
            "lin": q arrays initialised as lineal interpolation between two points,
                   q' arrays initialised as constant speed where applicable
                   q'' arrays initialised as zeroes where applicable
            "custom": user defined arrays for q, v, a and u
    t_start : float, optional
        Initial time. The default is 0.
    t_end : float, optional
        End time. The default is 1.
    verbose : bool, optional
       Expanded information printed during operation. The default is False.
    silent : bool, optional
        If true, no information is printed during operation. The default is True.

    Raises
    ------
    NotImplementedError
        When trying to use options not implemented yet.

    Returns
    -------
    Optimization Object
        Contains methods for easier configuration of opti problems

    """
    from .symbolic import ImplicitLagrangesMethod, SimpLagrangesMethod
    from sympy.physics.mechanics import LagrangesMethod

    inherit = [_Opti_Problem]  # Methods common to all problems

    # Depending on the kind of object that contains the physics:
    if isinstance(LM, ImplicitLagrangesMethod):
        inherit.append(_Implicit_Dynamics)
        if verbose:
            print("Dynamics detected: Implicit Lagranges Method")
    elif isinstance(LM, (SimpLagrangesMethod, LagrangesMethod)):
        inherit.append(_Explicit_Dynamics)
        if verbose:
            print("Dynamics detected: Explicit Lagranges Method")
    elif callable(LM):  # Maybe too generic?
        inherit.append(_Function_Dynamics)
        if verbose:
            print("Dynamics detected: Function")
    else:
        ValueError(
            f"LM must be a Lagranges Method object or a function, not: {type(LM)}"
        )

    # Depending on the type of scheme used:
    if scheme in _implemented_equispaced_schemes:
        inherit.append(_Equispaced)
    elif scheme in _implemented_pseudospectral_schemes:
        inherit.append(_Pseudospectral)
    else:
        _v = _implemented_equispaced_schemes + _implemented_pseudospectral_schemes
        raise NotImplementedError(
            f"scheme {scheme} not implemented. Valid methods are {_v}."
        )

    # Depending on the inicialization type:
    if ini_guess == "zero":
        inherit.append(_Zero_init)
    elif ini_guess == "lin":
        inherit.append(_Lin_init)
    elif ini_guess == "custom":
        inherit.append(_Custom_init)
    else:
        raise NotImplementedError(
            f"Initial value mode {ini_guess} not implemented. Valid methods are 'zero', 'lin' and 'custom'."
        )

    class Adequate_Problem(*inherit):
        pass

    return Adequate_Problem(
        LM, params, scheme, ini_guess, t_start, t_end, verbose, silent,
    )
