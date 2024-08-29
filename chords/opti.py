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
    - From _Opti_Problem, always. Contains methods common to all problems.
    
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
    node_points,
    coll_points,
    matrix_D_bary,
    # end_p_fun_cas,
    # start_p_fun_cas,
    get_bary_extreme_f_cas,
    get_coll_indices_from_nodes,
)
from .pseudospectral import _implemented_schemes as _implemented_pseudospectral_schemes
from .pseudospectral import (
    _gauss_like_schemes,
    _gauss_inv_schemes,
    _gauss_2_schemes,
    _radau_like_schemes,
    _radau_inv_schemes,
    _lobato_like_schemes,
    _other_schemes,
    tau_to_t_points,
)
from .bu_pseudospectral import (
    BU_coll_points,
    BU_construction_points,
    Integration_Matrix,
    Extreme_Matrix,
    get_coll_indices,
)
from .bu_pseudospectral import (
    _implemented_schemes as _implemented_bottom_up_pseudospectral_schemes,
)
from .td_pseudospectral import TD_construction_points, matrix_D_nu, matrix_L
from .util import get_weights, is_integer

import casadi as cas
from numpy import array, linspace, expand_dims, ones, ndarray, alltrue, zeros
from numpy.linalg import matrix_power
from numpy import sum as np_sum
from sympy import symbols
from time import time
from functools import lru_cache

_implemented_equispaced_schemes = [
    "euler",
    "trapz",
    "trapz_mod",
    "trapz_n",
    "hs",
    "hs_mod",
    "hs_parab",
    "hs_mod_parab",
    "hsj",
    "hsj_parab",
    "hsj_parab_mod",
    "hsn",
    "hsn_parab",
]


_implemented_top_down_pseudospectral_schemes = [
    "TD_" + _sch for _sch in _implemented_bottom_up_pseudospectral_schemes
]

_implemented_bottom_up_pseudospectral_schemes = [
    "BU_" + _sch for _sch in _implemented_bottom_up_pseudospectral_schemes
]

_all_ps_schemes = (
    _implemented_bottom_up_pseudospectral_schemes
    + _implemented_top_down_pseudospectral_schemes
    + _implemented_pseudospectral_schemes
)

_all_implemented_schemes = _all_ps_schemes + _implemented_equispaced_schemes


def get_q_and_ders_names(order):
    q_and_ders_names = ["q", "v", "a", "jerk", "snap", "crackle", "pop"]
    for _ii in range(7, order + 1):
        q_and_ders_names.append(f"d{_ii}q")
    return q_and_ders_names[: order + 1]


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
def _get_cost_obj_trap_int(scheme, N, order=2, mode="u", squared=True):
    """For a given pseudospectral scheme and number of collocation points,
    returns a function of values in said points that calculates a trapezoidal
    integration of the values or squared values over a [-1, 1] tau domain.
    """
    t_arr = BU_construction_points(N, scheme, order) + [1.0]
    t_arr = [float(ii) for ii in t_arr]
    if mode in "ux":
        start_p_f = get_bary_extreme_f_cas(
            scheme, N, mode=mode, point="start", order=order
        )
        end_p_f = get_bary_extreme_f_cas(scheme, N, mode=mode, point="end", order=order)
    exp = 2 if squared else 1

    def obj_f(coefs):
        coef_list = list(coefs)
        if mode == "u":
            start_p = start_p_f(coefs)
            end_p = end_p_f(coefs)
            if scheme in ["LG", "JG", "LG2"]:
                coef_list = [start_p] + coef_list + [end_p]
            elif scheme in ["LGR", "JGR"]:
                coef_list = coef_list + [end_p]
            elif scheme in ["LGR_inv", "JGR_inv"]:
                coef_list = [start_p] + coef_list
            elif scheme in ["LGL", "JGL", "D2"]:
                pass
            else:
                raise NotImplementedError("unrecognized scheme")
        elif mode == "x":
            start_p = start_p_f(coefs)
            end_p = end_p_f(coefs)
            if scheme in ["LG", "JG"]:
                coef_list = coef_list + [end_p]
            elif scheme in ["LGR", "LGR_inv", "LGL", "LG2", "D2"]:
                pass
            else:
                raise NotImplementedError("unrecognized scheme")
        elif mode == "complete":
            pass
        else:
            raise ValueError("unrecognized mode")
        sum_res = 0
        for jj in range(len(t_arr) - 1):
            sum_res += (
                (coef_list[jj] ** exp + coef_list[jj + 1] ** exp)
                * (t_arr[jj + 1] - t_arr[jj])
                / 2
            )
        return sum_res

    return obj_f


@lru_cache(maxsize=None)
def _get_cost_obj_trap_int_cas(scheme, N, order=2, mode="u", squared=True):
    """For a given pseudospectral scheme and number of collocation points,
    returns a casadi function of values in said points that calculates a
    trapezoidal integration of the values or squares over a [-1, 1] tau domain.
    """
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    if mode == "x":
        if scheme in ["JG", "LG2"]:
            N_arr = N + 2
        elif scheme in ["LG", "LGR", "JGR", "LGR_inv", "JGR_inv"]:
            N_arr = N + 1
        elif scheme in ["LGL", "JGL", "D2"]:
            N_arr = N
        else:
            raise NotImplementedError("unrecognized scheme")
    elif mode == "u":
        N_arr = N
    elif mode == "complete":
        if scheme in ["JG", "LG2", "LG"]:
            N_arr = N + 2
        elif scheme in ["LGR", "JGR", "LGR_inv", "JGR_inv"]:
            N_arr = N + 1
        elif scheme in ["LGL", "JGL", "D2"]:
            N_arr = N
        else:
            raise NotImplementedError("unrecognized scheme")

    u_sym = cas.SX.sym("u", N_arr)
    u_sympy = symbols(f"c0:{N_arr}")
    fun = _get_cost_obj_trap_int(scheme, N, order, squared=squared)
    sympy_expr = fun(u_sympy)
    cas_expr = sympy2casadi(sympy_expr, u_sympy, cas.vertsplit(u_sym))
    cas_f = cas.Function(
        "cost_func",
        [
            u_sym,
        ],
        [
            cas_expr,
        ],
    )
    return cas_f


@lru_cache(maxsize=None)
def _get_cost_obj_quad_int_cas(scheme, N, order=2, squared=False, mode="u"):
    """For a given pseudospectral scheme and number of collocation points,
    returns a casadi function of values in said points that calculates a
    quadrature integration of the values or squares over a [-1, 1] tau domain.
    """
    if scheme[:3] == "BU_":
        scheme = scheme[3:]

    u_sym = cas.SX.sym("u", N)
    exp = 2 if squared else 1
    weights = get_weights(N, scheme, order, mode)
    cas_expr = cas.sum1(weights * u_sym**exp)
    cas_f = cas.Function(
        "cost_func",
        [
            u_sym,
        ],
        [
            cas_expr,
        ],
    )
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
            problem.quad_cost() [apply a quadrature integral cost]
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
        if scheme[:3] == 'ph_':
            ph_scheme = True
            scheme = scheme[3:]
        else:
            ph_scheme = False
            
        if scheme in _implemented_equispaced_schemes:
            self.scheme_mode = "equispaced"
        elif scheme in _implemented_pseudospectral_schemes:
            self.scheme_mode = "pseudospectral"
        elif scheme in _implemented_bottom_up_pseudospectral_schemes:
            self.scheme_mode = "bottom-up pseudospectral"
            scheme = scheme[3:]
        elif scheme in _implemented_top_down_pseudospectral_schemes:
            self.scheme_mode = "top-down pseudospectral"
            scheme = scheme[3:]
        else:
            raise NotImplementedError(
                f"scheme {scheme} not implemented. Valid methods are {_all_implemented_schemes}."
            )
        if ph_scheme:
            self.scheme_mode = 'ph ' + self.scheme_mode
            
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
        """Tries to find optimization arrays in the object"""
        try:
            q_opti = self.opti_arrs["q"]
            v_opti = self.opti_arrs["v"]
            if self.order > 1:
                a_opti = self.opti_arrs["a"]
            else:
                a_opti = None
        except AttributeError:
            raise RuntimeError(
                "opti problem must be setup, use opti_setup() and apply_scheme()"
            )
        return q_opti, v_opti, a_opti

    def _save_results(self, results=None):
        """Saves results of optimization in dictionary 'results'"""
        if results is None:
            results = {}
        self.results = results
        self.results["scheme_mode"] = self.scheme_mode
        self.results["n_q"] = self.n_q
        self.results["n_u"] = self.n_u
        self.results["solve_order"] = self.order
        self.results["params"] = self.params
        if self.scheme_mode == "equispaced":
            self.results["N"] = self.N
            self.results["scheme"] = self.scheme
        else:
            self.results["n_coll"] = self.n_coll
            if self.scheme_mode == "bottom-up pseudospectral":
                self.results["scheme"] = "BU_" + self.scheme
            elif self.scheme_mode == "top-down pseudospectral":
                self.results["scheme"] = "TD_" + self.scheme
            else:
                self.results["scheme"] = self.scheme

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

        if self.cost is None:
            cost_out = None
        else:
            cost_out = sol.value(self.cost)

        results = {
            "cpudt": cpudt,
            "iter_count": self.opti.stats()["iter_count"],
            "cost": cost_out,
            "opti_stats": self.opti.stats(),
        }
        self._save_results(results)

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

        if self.cost is None:
            cost_out = None
        else:
            cost_out = sol.value(self.cost)

        results = {
            "cpudt": cpudt,
            "iter_count": self.opti.stats()["iter_count"],
            "cost": cost_out,
            "opti_stats": self.opti.stats(),
        }
        self._save_results(results)


class _Pseudospectral:
    def opti_setup(
        self,
        n_coll,
        precission=16,
        tol=1e-16,
        p_opts=None,
        s_opts=None,
    ):
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Points will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        n_coll : int
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
        order = self.order
        self.n_coll = n_coll
        coll_index = get_coll_indices_from_nodes(scheme)
        self.coll_index = coll_index
        try:
            order = self.order
            n_q = self.n_q
            n_u = self.n_u

        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )
        if scheme in _gauss_2_schemes + [
            "D2",
        ]:
            assert order == 2

        opti = cas.Opti()
        if p_opts is None:
            if self.verbose:
                p_opts = {
                    "expand": True,
                }
            else:
                p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        if s_opts is None:
            s_opts = {
                "max_iter": 10000,
                "tol": tol,
            }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        if scheme in _gauss_like_schemes + _gauss_inv_schemes:
            n_nodes = n_coll + 1
        elif scheme in _radau_inv_schemes + _radau_like_schemes:
            n_nodes = n_coll + 1
        elif scheme in _gauss_2_schemes:
            n_nodes = n_coll + 2
        elif scheme in _lobato_like_schemes:
            n_nodes = n_coll
        elif scheme in _other_schemes:
            if scheme == "D2":
                n_nodes = n_coll
            else:
                raise NotImplementedError(
                    f"unrecognized scheme {scheme}. "
                    + "Implementede schemes are { _implemented_pseudospectral_schemes}"
                )
        else:
            raise NotImplementedError(
                f"unrecognized scheme {scheme}. "
                + "Implementede schemes are { _implemented_pseudospectral_schemes}"
            )

        collocation_points_tau = coll_points(n_coll, scheme, precission)
        node_points_tau = node_points(n_nodes, scheme, precission)
        arr_points_tau = node_points_tau
        n_arr = n_nodes

        if scheme in _gauss_like_schemes:
            n_arr = n_nodes + 1
            arr_points_tau = arr_points_tau + [
                1.0,
            ]
        elif scheme in _gauss_inv_schemes:
            n_arr = n_nodes + 1
            arr_points_tau = [
                -1.0,
            ] + arr_points_tau

        self.n_nodes = n_nodes
        self.n_arr = n_arr

        D_mat = array(matrix_D_bary(n_nodes, scheme, precission), dtype="float64")
        self.D_mat = D_mat

        x_opti = opti.variable(n_arr, order * n_q)
        x_dot_opti = opti.variable(n_arr, order * n_q)
        if scheme in _gauss_like_schemes:
            x_node_opti = x_opti[:-1, :]
            x_dot_node_opti = x_dot_opti[:-1, :]
        elif scheme in _gauss_inv_schemes:
            x_node_opti = x_opti[1:, :]
            x_dot_node_opti = x_dot_opti[1:, :]
        else:
            x_node_opti = x_opti
            x_dot_node_opti = x_dot_opti
        u_opti = opti.variable(n_coll, n_u)
        u_like_x_opti = u_opti  # in later steps extreme points are added

        q_and_ders = []
        for _ii in range(order):
            q_and_ders.append(x_opti[:, n_q * _ii : n_q * (_ii + 1)])
        q_and_ders.append(x_dot_opti[:, -n_q:])
        self.q_and_ders_opti = q_and_ders

        q_and_ders_node = []
        for _ii in range(order):
            q_and_ders_node.append(x_node_opti[:, n_q * _ii : n_q * (_ii + 1)])
        q_and_ders_node.append(x_dot_node_opti[:, -n_q:])
        self.q_and_ders_node_opti = q_and_ders_node

        q_and_ders_names = get_q_and_ders_names(order)
        self.q_and_ders_names = q_and_ders_names[: order + 1]
        lam_opti = opti.variable(n_coll, self.n_lambdas)

        tau_arr = array(arr_points_tau, dtype="float64")
        node_tau_arr = array(node_points_tau, dtype="float64")
        col_tau_arr = array(collocation_points_tau, dtype="float64")

        t_arr = tau_to_t_points(tau_arr, t_start, t_end)
        node_t_arr = tau_to_t_points(node_tau_arr, t_start, t_end)
        col_t_arr = tau_to_t_points(col_tau_arr, t_start, t_end)

        self.opti_arrs = {
            "x": x_opti,
            "x_node": x_node_opti,
            "x_like_u": x_node_opti[coll_index, :],
            "x_d": x_dot_opti,
            "x_d_node": x_dot_node_opti,
            "x_d_like_u": x_dot_node_opti[coll_index, :],
            "u": u_opti,
            "t": t_arr,
            "t_node": node_t_arr,
            "t_col": col_t_arr,
            "tau": tau_arr,
            "tau_node": node_tau_arr,
            "tau_col": col_tau_arr,
            "lam": lam_opti,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
        }
        if scheme in _radau_like_schemes + _lobato_like_schemes + [
            "D2",
        ]:
            self.opti_points["u_s"] = u_opti[0, :]
        else:
            start_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="start", order=order
            )
            self.opti_points["u_s"] = start_f(u_opti)
            u_like_x_opti = cas.vertcat(self.opti_points["u_s"], u_like_x_opti)

        if scheme in _radau_inv_schemes + _lobato_like_schemes + [
            "D2",
        ]:
            self.opti_points["u_e"] = u_opti[-1, :]
        else:
            end_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="end", order=order
            )
            self.opti_points["u_e"] = end_f(u_opti)
            u_like_x_opti = cas.vertcat(u_like_x_opti, self.opti_points["u_e"])

        self.opti_arrs["u_like_x"] = u_like_x_opti

        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            _arr = q_and_ders[_ii]
            _arr_node = q_and_ders_node[_ii]
            self.opti_arrs[_name] = _arr
            self.opti_arrs[_name + "_node"] = _arr_node
            self.opti_arrs[_name + "_like_u"] = _arr_node[coll_index, :]
            self.opti_points[_name + "_s"] = _arr[0, :]
            self.opti_points[_name + "_e"] = _arr[-1, :]

        # try:
        #     if scheme in ["LGL", "LG", "LGR", "LGR_inv"]:
        #         x_opti = opti.variable(N, 2 * self.n_q)
        #         x_dot_opti = 2 / (t_end - t_start) * D_mat @ x_opti
        #         q_opti = x_opti[:, : self.n_q]
        #         v_opti = x_opti[:, self.n_q :]
        #         a_opti = x_dot_opti[:, self.n_q :]

        #     elif scheme in ["LG2", "D2", "LGLm", "JG"]:
        #         q_opti = opti.variable(N, self.n_q)
        #         v_opti = 2 / (t_end - t_start) * D_mat @ q_opti
        #         a_opti = 2 / (t_end - t_start) * D_mat @ v_opti
        #         x_opti = cas.horzcat(q_opti, v_opti)
        #         x_dot_opti = cas.horzcat(v_opti, a_opti)
        #     else:
        #         raise NotImplementedError(
        #             f"scheme {scheme} not implemented in opti_setup."
        #         )
        # except AttributeError:
        #     raise RuntimeError(
        #         "Dynamics must be computed before opti setup, use dynamic_setup()"
        #     )

        # u_opti = opti.variable(n_coll, self.n_u)
        # u_like_x_opti = u_opti
        # tau_arr = array(node_points(N, scheme, precission), dtype="float64")
        # col_arr = array(coll_points(n_coll, scheme, precission), dtype="float64")
        # t_arr = ((t_end - t_start) * tau_arr + (t_start + t_end)) / 2
        # t_col_arr = ((t_end - t_start) * col_arr + (t_start + t_end)) / 2
        # lam_opti = opti.variable(n_coll, self.n_lambdas)

        # self.opti_arrs = {
        #     "x": x_opti,
        #     "x_d": x_dot_opti,
        #     "q": q_opti,
        #     "v": v_opti,
        #     "a": a_opti,
        #     "u": u_opti,
        #     "lam": lam_opti,
        #     "tau": tau_arr,
        #     "tau_col": col_arr,
        #     "t": t_arr,
        #     "t_col": t_col_arr,
        # }

    def quad_cost(self, arr, squared=False):
        """
        Calculates a quadrature integration of an array and sets it
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

        dt = self.t_end - self.t_start
        N_col = self.n_coll
        N_x = self.n_arr
        N_arr = arr.shape[0]

        if N_arr == N_col:
            mode = "u"
        elif N_arr == N_x:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of collocation points nor array x lenght"
            )

        f_u_cost = _get_cost_obj_quad_int_cas(
            self.scheme, N_arr, self.order, squared, mode
        )
        cost = dt * cas.sum2(f_u_cost(arr))

        self.cost = cost
        self.opti.minimize(cost)

    def u_sq_cost(self):
        """
        Calculates a quadrature integration of u squared and sets it
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
        u_opti = self.opti_arrs["u"]
        self.quad_cost(u_opti, squared=True)

    def trapz_cost(self, arr, squared=False):
        """
        Calculates a trapezoidal integration of an array and sets it
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

        dt = self.t_end - self.t_start
        n_arr = self.n_arr
        n_coll = self.n_coll
        N_arr_opt = arr.shape[0]

        if N_arr_opt == n_coll:
            mode = "u"
        elif N_arr_opt == n_arr:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of node nor collocation points"
            )

        f_u_cost = _get_cost_obj_trap_int_cas(
            self.scheme, self.n_coll, self.order, mode, squared
        )
        cost = dt * cas.sum2(f_u_cost(arr))

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
            n_coll = self.n_coll
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )

        dynam_f_x = self.dyn_f_restr
        # dynam_g_q = self.dyn_g_restr
        x_opti = self.opti_arrs["x"]
        x_node_opti = self.opti_arrs["x_node"]
        x_dot_opti = self.opti_arrs["x_d"]
        x_dot_node_opti = self.opti_arrs["x_d_node"]
        # q_opti = self.opti_arrs["q"]
        # v_opti = self.opti_arrs["v"]
        # a_opti = self.opti_arrs["a"]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params
        q_and_ders = self.q_and_ders_opti
        q_and_ders_node = self.q_and_ders_node_opti
        D_mat = self.D_mat
        t_start = self.t_start
        t_end = self.t_end
        n_nodes = self.n_nodes

        # --- Scheme constraints ---
        if scheme in _gauss_2_schemes + ["D2"]:
            q_opti_node = q_and_ders_node[0]
            v_opti_node = q_and_ders_node[1]
            a_opti_node = q_and_ders_node[2]

            v_calc = 2 / (t_end - t_start) * D_mat @ q_opti_node
            a_calc = 2 / (t_end - t_start) * D_mat @ v_opti_node
            self.opti.subject_to(v_opti_node == v_calc)
            self.opti.subject_to(a_opti_node == a_calc)
        else:
            x_d_calc = 2 / (t_end - t_start) * D_mat @ x_node_opti
            self.opti.subject_to(x_dot_node_opti == x_d_calc)

        # --- Scheme not-node end values constraints ---
        if scheme in _gauss_inv_schemes:
            f_start = get_bary_extreme_f_cas(
                scheme, n_nodes, mode="x", point="start", order=self.order
            )
            x_start = f_start(x_node_opti)
            self.opti.subject_to(x_opti[0, :] == x_start)
            highest_q_d_opti = q_and_ders[-1]
            highest_q_d_node_opti = q_and_ders_node[-1]
            hqd_start = f_start(highest_q_d_node_opti)
            self.opti.subject_to(highest_q_d_opti[0, :] == hqd_start)

        if scheme in _gauss_like_schemes:
            f_end = get_bary_extreme_f_cas(
                scheme, n_nodes, mode="x", point="end", order=self.order
            )
            x_end = f_end(x_node_opti)
            self.opti.subject_to(x_opti[-1, :] == x_end)
            highest_q_d_opti = q_and_ders[-1]
            highest_q_d_node_opti = q_and_ders_node[-1]
            hqd_end = f_end(highest_q_d_node_opti)
            self.opti.subject_to(highest_q_d_opti[-1, :] == hqd_end)

        # --- Collocation Constraints ---
        if scheme in _lobato_like_schemes + _radau_like_schemes + ["D2"]:
            i0 = 0
        else:
            i0 = 1

        for ii in range(n_coll):
            self.opti.subject_to(
                dynam_f_x(
                    x_opti[i0 + ii, :],
                    x_dot_opti[i0 + ii, :],
                    u_opti[ii, :],
                    lam_opti[ii, :],
                    params,
                )
                == 0
            )
        # if scheme == "LGL":
        #     for ii in range(N):
        #         self.opti.subject_to(
        #             dynam_f_x(
        #                 x_opti[ii, :],
        #                 x_dot_opti[ii, :],
        #                 u_opti[ii, :],
        #                 lam_opti[ii, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "LG":
        #     for ii in range(1, N):
        #         self.opti.subject_to(
        #             dynam_f_x(
        #                 x_opti[ii, :],
        #                 x_dot_opti[ii, :],
        #                 u_opti[ii - 1, :],
        #                 lam_opti[ii - 1, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "LGR":
        #     for ii in range(N - 1):
        #         self.opti.subject_to(
        #             dynam_f_x(
        #                 x_opti[ii, :],
        #                 x_dot_opti[ii, :],
        #                 u_opti[ii, :],
        #                 lam_opti[ii, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "LGR_inv":
        #     for ii in range(1, N):
        #         self.opti.subject_to(
        #             dynam_f_x(
        #                 x_opti[ii, :],
        #                 x_dot_opti[ii, :],
        #                 u_opti[ii - 1, :],
        #                 lam_opti[ii - 1, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "D2":
        #     for ii in range(N):
        #         self.opti.subject_to(
        #             dynam_g_q(
        #                 q_opti[ii, :],
        #                 v_opti[ii, :],
        #                 a_opti[ii, :],
        #                 u_opti[ii, :],
        #                 lam_opti[ii, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "LG2":
        #     for ii in range(1, N - 1):
        #         self.opti.subject_to(
        #             dynam_g_q(
        #                 q_opti[ii, :],
        #                 v_opti[ii, :],
        #                 a_opti[ii, :],
        #                 u_opti[ii - 1, :],
        #                 lam_opti[ii - 1, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "LGLm":
        #     for ii in range(1, N - 1):
        #         self.opti.subject_to(
        #             dynam_g_q(
        #                 q_opti[ii, :],
        #                 v_opti[ii, :],
        #                 a_opti[ii, :],
        #                 u_opti[ii - 1, :],
        #                 lam_opti[ii - 1, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # elif scheme == "JG":
        #     for ii in range(1, N - 1):
        #         self.opti.subject_to(
        #             dynam_g_q(
        #                 q_opti[ii, :],
        #                 v_opti[ii, :],
        #                 a_opti[ii, :],
        #                 u_opti[ii - 1, :],
        #                 lam_opti[ii - 1, :],
        #                 params,
        #             )
        #             == 0
        #         )
        # else:
        #     raise NotImplementedError(
        #         f"scheme {scheme} not implemented in apply_scheme."
        #     )


class _BU_Pseudospectral:
    def opti_setup(
        self,
        n_coll,
        precission=16,
        tol=1e-16,
        p_opts=None,
        s_opts=None,
    ):
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Points will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        n_coll : int
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

        try:
            order = self.order
            n_q = self.n_q
            n_u = self.n_u

        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        self.n_coll = n_coll
        self.precission = precission
        coll_index = get_coll_indices(scheme)
        self.coll_index = coll_index

        opti = cas.Opti()
        if p_opts is None:
            if self.verbose:
                p_opts = {
                    "expand": True,
                }
            else:
                p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        if s_opts is None:
            s_opts = {
                "max_iter": 10000,
                "tol": tol,
            }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        construction_points_tau = BU_construction_points(
            n_coll, scheme, order, precission
        )
        collocation_points_tau = BU_coll_points(n_coll, scheme, order, precission)
        n_arr = len(construction_points_tau) + 1
        self.n_arr = n_arr

        x_opti = opti.variable(n_arr, order * n_q)
        x_dot_opti = opti.variable(n_arr, order * n_q)
        u_opti = opti.variable(n_coll, n_u)
        u_like_x_opti = u_opti

        q_and_ders = []
        for _ii in range(order):
            q_and_ders.append(x_opti[:, n_q * _ii : n_q * (_ii + 1)])
        q_and_ders.append(x_dot_opti[:, -n_q:])

        q_and_ders_names = get_q_and_ders_names(order)
        self.q_and_ders_names = q_and_ders_names[: order + 1]
        lam_opti = opti.variable(n_coll, self.n_lambdas)

        tau_arr = array([-1.0] + construction_points_tau, dtype="float64")
        col_tau_arr = array(collocation_points_tau, dtype="float64")
        t_arr = tau_to_t_points(tau_arr, t_start, t_end)
        col_t_arr = tau_to_t_points(col_tau_arr, t_start, t_end)

        self.opti_arrs = {
            "x": x_opti,
            "x_like_u": x_opti[coll_index, :],
            "x_d": x_dot_opti,
            "x_d_like_u": x_dot_opti[coll_index, :],
            "u": u_opti,
            "t": t_arr,
            "t_col": col_t_arr,
            "tau": tau_arr,
            "tau_col": col_tau_arr,
            "lam": lam_opti,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
        }
        if scheme in _radau_like_schemes + _lobato_like_schemes:
            self.opti_points["u_s"] = u_opti[0, :]
        else:
            start_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="start", order=order
            )
            self.opti_points["u_s"] = start_f(u_opti)
            u_like_x_opti = cas.vertcat(self.opti_points["u_s"], u_like_x_opti)

        if scheme in _radau_inv_schemes + _lobato_like_schemes:
            self.opti_points["u_e"] = u_opti[-1, :]
        else:
            end_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="end", order=order
            )
            self.opti_points["u_e"] = end_f(u_opti)
            u_like_x_opti = cas.vertcat(u_like_x_opti, self.opti_points["u_e"])
        self.opti_arrs["u_like_x"] = u_like_x_opti

        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            _arr = q_and_ders[_ii]
            self.opti_arrs[_name] = _arr
            self.opti_arrs[_name + "_like_u"] = _arr[coll_index, :]
            self.opti_points[_name + "_s"] = _arr[0, :]
            self.opti_points[_name + "_e"] = _arr[-1, :]

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

        f_u_cost = _get_cost_obj_quad_int_cas(self.scheme, self.n_coll, squared=True)
        cost = dt * cas.sum2(f_u_cost(U))

        self.cost = cost
        self.opti.minimize(cost)

    def quad_cost(self, arr, squared=False):
        """
        Calculates a quadrature integration of an array and sets it
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

        dt = self.t_end - self.t_start
        N_col = self.n_coll
        N_x = self.n_arr
        N_arr = arr.shape[0]

        if N_arr == N_col:
            mode = "u"
        elif N_arr == N_x:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of collocation points nor array x lenght"
            )

        f_u_cost = _get_cost_obj_quad_int_cas(
            self.scheme, N_arr, self.order, squared, mode
        )
        cost = dt * cas.sum2(f_u_cost(arr))

        self.cost = cost
        self.opti.minimize(cost)

    def trapz_cost(self, arr, squared=False):
        """
        Calculates a trapezoidal integration of an array and sets it
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

        dt = self.t_end - self.t_start
        N = self.n_arr
        N_col = self.n_coll
        N_arr = arr.shape[0]

        if N_arr == N_col:
            mode = "u"
        elif N_arr == N:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of node nor collocation points"
            )

        f_u_cost = _get_cost_obj_trap_int_cas(
            self.scheme, self.n_coll, self.order, mode, squared
        )
        cost = dt * cas.sum2(f_u_cost(arr))

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
            n_arr = self.n_arr
            n_coll = self.n_coll
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )

        order = self.order
        q_and_ders_names = get_q_and_ders_names(order)
        h = self.t_end - self.t_start

        dynam_f_x = self.dyn_f_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        highest_q_d_opti = self.opti_arrs[q_and_ders_names[-1]]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params

        coll_index = self.coll_index
        q_and_ders_0 = []
        for ii in q_and_ders_names[:-1]:
            q_and_ders_0.append(self.opti_points[ii + "_s"])
        highest_q_d_opti_coll = highest_q_d_opti[coll_index, :]
        polynomial_data = cas.vertcat(*q_and_ders_0, highest_q_d_opti_coll)

        # ----- Extremes of highest derivative Constraints -----

        if scheme in _gauss_like_schemes + _radau_inv_schemes:
            start_mat = Extreme_Matrix(
                n_coll, scheme, "start", scheme_order=order, precission=self.precission
            )
            self.opti.subject_to(
                highest_q_d_opti[0, :] == start_mat @ highest_q_d_opti_coll
            )
        if scheme in _gauss_like_schemes + _radau_like_schemes:
            end_mat = Extreme_Matrix(
                n_coll, scheme, "end", scheme_order=order, precission=self.precission
            )
            self.opti.subject_to(
                highest_q_d_opti[-1, :] == end_mat @ highest_q_d_opti_coll
            )

        # ----- Scheme Constraints ----

        for ii in range(order):
            # Careful of not including the starting point!
            q_and_der_matrix = self.opti_arrs[q_and_ders_names[ii]][1:, :]
            I_mat = Integration_Matrix(
                n_coll, scheme, ii, h, scheme_order=order, precission=self.precission
            )
            self.opti.subject_to(q_and_der_matrix == I_mat @ polynomial_data)

        # ----- Dynamics Constraints ----

        indices = [ii for ii in range(n_arr)][coll_index]
        for ii, jj in enumerate(indices):
            # ii : collocation point counter
            # jj : index in array of x
            self.opti.subject_to(
                dynam_f_x(
                    x_opti[jj, :],
                    x_dot_opti[jj, :],
                    u_opti[ii, :],
                    lam_opti[ii, :],
                    params,
                )
                == 0
            )


class _TD_Pseudospectral:
    def opti_setup(
        self,
        n_coll,
        precission=16,
        tol=1e-16,
        p_opts=None,
        s_opts=None,
    ):
        """
        Creates and links the different opti variables to be used in the problem.
        Requires the function dynamic_setup() to have been run prior.
        Arrays will be accesible through self.opti_arrs dictionary.
        Points will be accesible through self.opti_points dictionary.

        Parameters
        ----------
        n_coll : int
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

        try:
            order = self.order
            n_q = self.n_q
            n_u = self.n_u

        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        self.n_coll = n_coll
        self.precission = precission
        coll_index = get_coll_indices(scheme)
        self.coll_index = coll_index

        opti = cas.Opti()
        if p_opts is None:
            if self.verbose:
                p_opts = {
                    "expand": True,
                }
            else:
                p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        if s_opts is None:
            s_opts = {
                "max_iter": 10000,
                "tol": tol,
            }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        construction_points_tau = TD_construction_points(
            n_coll, scheme, order, precission
        )
        collocation_points_tau = BU_coll_points(n_coll, scheme, order, precission)
        n_arr = len(construction_points_tau)
        self.n_arr = n_arr

        x_opti = opti.variable(n_arr, order * n_q)
        x_dot_opti = opti.variable(n_arr, order * n_q)
        u_opti = opti.variable(n_coll, n_u)
        u_like_x_opti = u_opti
        q_constr_opti = opti.variable(n_coll + order, n_q)

        q_and_ders = []
        for _ii in range(order):
            q_and_ders.append(x_opti[:, n_q * _ii : n_q * (_ii + 1)])
        q_and_ders.append(x_dot_opti[:, -n_q:])

        q_and_ders_names = get_q_and_ders_names(order)
        self.q_and_ders_names = q_and_ders_names[: order + 1]
        lam_opti = opti.variable(n_coll, self.n_lambdas)

        tau_arr = array(construction_points_tau, dtype="float64")
        col_tau_arr = array(collocation_points_tau, dtype="float64")
        t_arr = tau_to_t_points(tau_arr, t_start, t_end)
        col_t_arr = tau_to_t_points(col_tau_arr, t_start, t_end)

        self.opti_arrs = {
            "x": x_opti,
            "x_like_u": x_opti[coll_index, :],
            "x_d": x_dot_opti,
            "x_d_like_u": x_dot_opti[coll_index, :],
            "u": u_opti,
            "t": t_arr,
            "t_col": col_t_arr,
            "tau": tau_arr,
            "tau_col": col_tau_arr,
            "lam": lam_opti,
            "q_constr": q_constr_opti,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
        }
        if scheme in _radau_like_schemes + _lobato_like_schemes:
            self.opti_points["u_s"] = u_opti[0, :]
        else:
            start_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="start", order=order
            )
            self.opti_points["u_s"] = start_f(u_opti)
            u_like_x_opti = cas.vertcat(self.opti_points["u_s"], u_like_x_opti)

        if scheme in _radau_inv_schemes + _lobato_like_schemes:
            self.opti_points["u_e"] = u_opti[-1, :]
        else:
            end_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="end", order=order
            )
            self.opti_points["u_e"] = end_f(u_opti)
            u_like_x_opti = cas.vertcat(u_like_x_opti, self.opti_points["u_e"])
        self.opti_arrs["u_like_x"] = u_like_x_opti

        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            _arr = q_and_ders[_ii]
            self.opti_arrs[_name] = _arr
            self.opti_arrs[_name + "_like_u"] = _arr[coll_index, :]
            self.opti_points[_name + "_s"] = _arr[0, :]
            self.opti_points[_name + "_e"] = _arr[-1, :]

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

        f_u_cost = _get_cost_obj_quad_int_cas(self.scheme, self.n_coll, squared=True)
        cost = dt * cas.sum2(f_u_cost(U))

        self.cost = cost
        self.opti.minimize(cost)

    def quad_cost(self, arr, squared=False):
        """
        Calculates a quadrature integration of an array and sets it
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

        dt = self.t_end - self.t_start
        N_col = self.n_coll
        N_x = self.n_arr
        N_arr = arr.shape[0]

        if N_arr == N_col:
            mode = "u"
        elif N_arr == N_x:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of collocation points nor array x lenght"
            )

        f_u_cost = _get_cost_obj_quad_int_cas(
            self.scheme, N_arr, self.order, squared, mode
        )
        cost = dt * cas.sum2(f_u_cost(arr))

        self.cost = cost
        self.opti.minimize(cost)

    def trapz_cost(self, arr, squared=False):
        """
        Calculates a trapezoidal integration of an array and sets it
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

        dt = self.t_end - self.t_start
        N = self.n_arr
        N_col = self.n_coll
        N_arr = arr.shape[0]

        if N_arr == N_col:
            mode = "u"
        elif N_arr == N:
            mode = "x"
        else:
            raise ValueError(
                "Unrecognized shape of arr, not equal to number"
                + " of node nor collocation points"
            )

        f_u_cost = _get_cost_obj_trap_int_cas(
            self.scheme, self.n_coll, self.order, mode, squared
        )
        cost = dt * cas.sum2(f_u_cost(arr))

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
            n_arr = self.n_arr
            n_coll = self.n_coll
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )

        order = self.order
        q_and_ders_names = get_q_and_ders_names(order)
        h = self.t_end - self.t_start

        dynam_f_x = self.dyn_f_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        # highest_q_d_opti = self.opti_arrs[q_and_ders_names[-1]]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        q_constr_opti = self.opti_arrs["q_constr"]
        params = self.params

        coll_index = self.coll_index
        q_and_ders_0 = []
        for ii in q_and_ders_names[:-1]:
            q_and_ders_0.append(self.opti_points[ii + "_s"])
        # highest_q_d_opti_coll = highest_q_d_opti[coll_index, :]
        # polynomial_data = cas.vertcat(*q_and_ders_0, highest_q_d_opti_coll)

        D_nu = matrix_D_nu(order + n_coll)
        L = matrix_L(n_coll, scheme, order, self.precission)

        # ----- Scheme Constraints ----

        for ii in range(order + 1):
            q_and_der_matrix = self.opti_arrs[q_and_ders_names[ii]]
            deriv_matrix = (2 / h) ** ii * L @ matrix_power(D_nu, ii)
            self.opti.subject_to(q_and_der_matrix == deriv_matrix @ q_constr_opti)

        # ----- Dynamics Constraints ----

        indices = [ii for ii in range(n_arr)][coll_index]
        for ii, jj in enumerate(indices):
            # ii : collocation point counter
            # jj : index in array of x
            self.opti.subject_to(
                dynam_f_x(
                    x_opti[jj, :],
                    x_dot_opti[jj, :],
                    u_opti[ii, :],
                    lam_opti[ii, :],
                    params,
                )
                == 0
            )


class _Equispaced:
    def opti_setup(self, segment_number, tol=1e-16, p_opts=None, s_opts=None):
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
        if p_opts is None:
            if self.verbose:
                p_opts = {
                    "expand": True,
                }
            else:
                p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        if s_opts is None:
            s_opts = {
                "max_iter": 10000,
                "tol": tol,
            }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        try:
            order = self.order
            x_opti = opti.variable(N + 1, order * self.n_q)
            x_dot_opti = opti.variable(N + 1, order * self.n_q)
            u_opti = opti.variable(N + 1, self.n_u)
            q_and_ders = []
            for _ii in range(order):
                q_and_ders.append(x_opti[:, self.n_q * _ii : self.n_q * (_ii + 1)])
            q_and_ders.append(x_dot_opti[:, -self.n_q :])
            # q_opti = x_opti[:, : self.n_q]
            # v_opti = x_opti[:, self.n_q :]
            # a_opti = x_dot_opti[:, -self.n_q :]
            q_and_ders_names = ["q", "v", "a", "jerk", "snap", "crackle", "pop"]
            for _ii in range(7, order + 1):
                q_and_ders_names.append(f"d{_ii}q")
            self.q_and_ders_names = q_and_ders_names[: order + 1]

            t_arr = linspace(t_start, t_end, N + 1)
        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        lam_opti = opti.variable(N + 1, self.n_lambdas)

        self.opti_arrs = {
            "x": x_opti,
            "x_d": x_dot_opti,
            "x_like_u": x_opti,
            "x_d_like_u": x_dot_opti,
            # "q": q_opti,
            # "v": v_opti,
            # "a": a_opti,
            "u": u_opti,
            "u_like_x": u_opti,
            "t": t_arr,
            "t_col": t_arr,
            "lam": lam_opti,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
            # "q_s": q_opti[0, :],
            # "q_e": q_opti[-1, :],
            # "v_s": v_opti[0, :],
            # "v_e": v_opti[-1, :],
            # "a_s": a_opti[0, :],
            # "a_e": a_opti[-1, :],
        }

        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            _arr = q_and_ders[_ii]
            self.opti_arrs[_name] = _arr
            self.opti_arrs[_name + "_like_u"] = _arr
            self.opti_points[_name + "_s"] = _arr[0, :]
            self.opti_points[_name + "_e"] = _arr[-1, :]

        if "hs" in scheme:
            u_c_opti = opti.variable(N, self.n_u)
            x_c_opti = opti.variable(N, order * self.n_q)
            x_dot_c_opti = opti.variable(N, order * self.n_q)
            q_and_ders_c = []
            for _ii in range(order):
                q_and_ders_c.append(x_c_opti[:, self.n_q * _ii : self.n_q * (_ii + 1)])
            q_and_ders_c.append(x_dot_c_opti[:, -self.n_q :])
            # q_c_opti = x_c_opti[:, : self.n_q]
            # v_c_opti = x_c_opti[:, self.n_q :]
            # a_c_opti = x_dot_c_opti[:, self.n_q :]
            if "j" in scheme:
                t_c_arr = 0.6 * t_arr[:-1] + 0.4 * t_arr[1:]
            else:
                t_c_arr = (t_arr[:-1] + t_arr[1:]) / 2
            lam_c_opti = opti.variable(N, self.n_lambdas)
            self.opti_arrs = {
                **self.opti_arrs,
                "x_c": x_c_opti,
                "x_d_c": x_dot_c_opti,
                # "q_c": q_c_opti,
                # "v_c": v_c_opti,
                # "a_c": a_c_opti,
                "u_c": u_c_opti,
                "t_c": t_c_arr,
                "lam_c": lam_c_opti,
            }

            for _ii in range(order + 1):
                _name = q_and_ders_names[_ii]
                _arr = q_and_ders_c[_ii]
                self.opti_arrs[_name + "_c"] = _arr

            if scheme == "hsj_parab_mod":
                u_j_opti = opti.variable(N, self.n_u)
                self.opti_arrs["u_j"] = u_j_opti

    def quad_cost(self, arr, arr_c=None, squared=False):
        """
        Calculates a quadrature integration of an array and sets it
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

        dt = self.t_end - self.t_start
        if squared:
            arr = arr**2

        if arr_c is None:
            cost = dt * cas.sum2(
                (cas.sum1(arr[:, :]) + cas.sum1(arr[1:-1, :])) / self.N
            )
        else:
            if squared:
                arr_c = arr_c**2
            cost = dt * cas.sum2(
                (
                    4 * cas.sum1(arr_c[:, :])
                    + cas.sum1(arr[:, :])
                    + cas.sum1(arr[1:-1, :])
                )
                / (3 * self.N)
            )

        self.cost = cost
        self.opti.minimize(cost)

    def sq_cost(self, arr, arr_c=None):
        """
        Calculates a quadrature integration of an array squared and sets it
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

        dt = self.t_end - self.t_start
        arr_sq = arr**2
        if arr_c is None:
            cost = dt * cas.sum2(
                (cas.sum1(arr_sq[:, :]) + cas.sum1(arr_sq[1:-1, :])) / self.N
            )
        else:
            arr_c_sq = arr_c**2
            cost = dt * cas.sum2(
                (
                    4 * cas.sum1(arr_c_sq[:, :])
                    + cas.sum1(arr_sq[:, :])
                    + cas.sum1(arr_sq[1:-1, :])
                )
                / (3 * self.N)
            )

        self.cost = cost
        self.opti.minimize(cost)

    def u_sq_cost(self):
        """
        Calculates a quadrature integration of u squared and sets it
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
        U_sq = U**2

        if "parab" in self.scheme:
            U_c = self.opti_arrs["u_c"]
            if self.scheme == "hsj_parab_mod":
                U_c = self.opti_arrs["u_j"]
                U_j = self.opti_arrs["u_c"]  # OJO EN arrs están guardados al revés!
                # self.opti.subject_to(U_c == (25*U_j + 2*U[1:, :] - 3*U[:-1, :])/24)
                self.opti.subject_to(
                    U_j == (24 * U_c - 2 * U[1:, :] + 3 * U[:-1, :]) / 25
                )

            U_c_sq = U_c**2

            if "j" in self.scheme and self.scheme != "hsj_parab_mod":
                cost = dt * cas.sum2(
                    (
                        25 * cas.sum1(U_c_sq[:, :])
                        + 3 * cas.sum1(U_sq[:-1, :])
                        + 8 * cas.sum1(U_sq[1:, :])
                    )
                    / (18 * self.N)
                )
            else:
                cost = dt * cas.sum2(
                    (
                        4 * cas.sum1(U_c_sq[:, :])
                        + cas.sum1(U_sq[:, :])
                        + cas.sum1(U_sq[1:-1, :])
                    )
                    / (3 * self.N)
                )
        else:
            cost = dt * cas.sum2(
                (cas.sum1(U_sq[:, :]) + cas.sum1(U_sq[1:-1, :])) / self.N
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
        from .piecewise import (
            # euler_accel_restr,
            trapz_accel_restr,
            trapz_mod_accel_restr,
            hs_mod_accel_restr,
            hs_mod_half_x,
            hs_accel_restr,
            hs_half_x,
            hsj_accel_restr,
            hsj_half_x,
            generate_hs_m_funcs,
            generate_trapz_m_func,
        )

        scheme = self.scheme
        try:
            N = self.N
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )
        q_and_ders_names = self.q_and_ders_names
        T = self.t_end - self.t_start
        dynam_f_x = self.dyn_f_restr
        # dynam_g_q = self.dyn_g_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        # q_opti = self.opti_arrs["q"]
        # v_opti = self.opti_arrs["v"]
        highest_q_d_opti = self.opti_arrs[q_and_ders_names[-1]]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params
        if "hs" in scheme:
            x_c_opti = self.opti_arrs["x_c"]
            x_c_dot_opti = self.opti_arrs["x_d_c"]
            u_c_opti = self.opti_arrs["u_c"]
            # a_c_opti = self.opti_arrs["a_c"]
            highest_q_d_c_opti = self.opti_arrs[q_and_ders_names[-1] + "_c"]
            lam_c_opti = self.opti_arrs["lam_c"]

            if "j" in scheme:
                half_x = hsj_half_x
            elif "hsn" in scheme:
                half_x = generate_hs_m_funcs(self.order)[0]
            elif "mod" in scheme:
                half_x = hs_mod_half_x
            else:
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
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
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
                if "j" in scheme:
                    for ii in range(N):
                        self.opti.subject_to(
                            u_c_opti[ii, :]
                            == 0.6 * u_opti[ii, :] + 0.4 * u_opti[ii + 1, :]
                        )
                else:
                    for ii in range(N):
                        self.opti.subject_to(
                            u_c_opti[ii, :] == (u_opti[ii, :] + u_opti[ii + 1, :]) / 2
                        )

        # Scheme Constraints
        restr_schemes = {
            #'euler': euler_accel_restr, #comprobar compatibilidad
            "trapz": trapz_accel_restr,
            "trapz_mod": trapz_mod_accel_restr,
            "trapz_n": generate_trapz_m_func(self.order),
            "hs": hs_accel_restr,
            "hs_mod": hs_mod_accel_restr,
            "hs_parab": hs_accel_restr,
            "hs_mod_parab": hs_mod_accel_restr,
            "hsj": hsj_accel_restr,
            "hsj_parab": hsj_accel_restr,
            "hsj_parab_mod": hsj_accel_restr,
            "hsn": generate_hs_m_funcs(self.order)[1],
            "hsn_parab": generate_hs_m_funcs(self.order)[1],
        }
        n_q = self.n_q
        f_restr = restr_schemes[scheme]
        if "hs" in scheme:
            cas_accel_restr = accelrestriction2casadi(
                f_restr, n_q, n_q, order=self.order
            )
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
                        T / N,
                        highest_q_d_c_opti[ii, :],
                    )
                    == 0
                )
        else:
            cas_accel_restr = accelrestriction2casadi(f_restr, n_q, order=self.order)
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
                        T / N,
                        [],
                    )
                    == 0
                )





class _multi_pseudospectral:
    def _check_structure_consistency(
        self, t_knots_arr, point_structure,
    ):
        """Initial structural checks to ensure compatibilty of given data"""

        #n_q = self.n_q
        

        # t_knot_arr checks
        t_knots_arr = array(t_knots_arr, dtype="float64")
        assert len(t_knots_arr.shape) == 1
        n_segments = t_knots_arr.shape[0] + 1
        self.n_segments = n_segments
        t_start = self.t_start
        t_end = self.t_end
        assert alltrue(t_knots_arr > t_start)
        assert alltrue(t_knots_arr < t_end)
        assert alltrue(t_knots_arr[:-1] <= t_knots_arr[1:])  # is sorted

        # point_structures checks
        for element in point_structure:
            assert is_integer(element), f'{element} in point_structure should be int'
        point_structure = array(point_structure, dtype="int")
        assert len(point_structure.shape) == 1
        assert point_structure.shape[0] == n_segments

        self.point_structure = point_structure
        return t_knots_arr, point_structure
    
    def opti_setup(
            self, 
            t_knots_arr,
            point_structure,
            precission=16,
            tol=1e-16,
            p_opts=None,
            s_opts=None
        ):
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
        try:
            n_q = self.n_q
            n_u = self.n_u
        except AttributeError:
            raise RuntimeError(
                "Dynamics must be computed before opti setup, use dynamic_setup()"
            )

        scheme = self.scheme
        t_start = self.t_start
        t_end = self.t_end
        order = self.order
        
        t_knots_arr, point_structure = self._check_structure_consistency(
            t_knots_arr,
            point_structure
        )
        n_segments = self.n_segments
        n_coll_total = np_sum(point_structure)
        #Total Number of points:
        N = n_coll_total + n_segments + 1
        self.N = N
        self.n_coll_total = n_coll_total
        
        opti = cas.Opti()
        if p_opts is None:
            if self.verbose:
                p_opts = {
                    "expand": True,
                }
            else:
                p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
        if s_opts is None:
            s_opts = {
                "max_iter": 10000,
                "tol": tol,
            }  # investigate how to make it work adding 'linear_solver' : "MA27"}
        opti.solver("ipopt", p_opts, s_opts)
        self.opti = opti

        # ---- X opti setup ----
        
        x_opti = opti.variable(N, order * self.n_q)
        x_dot_opti = opti.variable(N, order * self.n_q)
        
        x_knot_opti_list = []
        x_coll_opti_list = []
        x_dot_knot_opti_list = []
        x_dot_coll_opti_list = []
        
        ii_inf = 1
        for seg_ii in range(n_segments):
            ii_sup = ii_inf + point_structure[seg_ii]
            x_coll_opti_list.append(x_opti[ii_inf:ii_sup,:])
            x_dot_coll_opti_list.append(x_dot_opti[ii_inf:ii_sup,:])
            x_knot_opti_list.append(x_opti[ii_sup,:])
            x_dot_knot_opti_list.append(x_dot_opti[ii_sup,:])
            
        #Last knot is not a knot but the end extreme:
        x_knot_opti_list.pop(-1)
        x_dot_knot_opti_list.pop(-1)
        
        x_like_u_opti = cas.vertcat(*x_coll_opti_list)
        x_dot_like_u_opti = cas.vertcat(*x_dot_coll_opti_list)
        
        # ---- U opti setup ----
        
        u_like_x_opti = opti.variable(N, order * n_u)
        
        u_knot_opti_list = []
        u_coll_opti_list = []
        
        ii_inf = 1
        for seg_ii in range(n_segments):
            ii_sup = ii_inf + point_structure[seg_ii]
            u_coll_opti_list.append(u_like_x_opti[ii_inf:ii_sup,:])
            u_knot_opti_list.append(u_like_x_opti[ii_sup,:])
            
        #Last knot is not a knot but the end extreme:
        u_knot_opti_list.pop(-1)
        
        u_opti = cas.vertcat(*u_coll_opti_list)
        
        #---- time arrays setup ----
        
        t_knots_and_extremes_arr = zeros(n_segments+1, dtype = 'float64')
        t_knots_and_extremes_arr[0] = t_start
        t_knots_and_extremes_arr[1:-1] = t_knots_arr
        t_knots_and_extremes_arr[-1] = t_end
        self.t_knots_and_extremes_arr = t_knots_and_extremes_arr
        h_arr = t_knots_and_extremes_arr[1:] - t_knots_and_extremes_arr[:-1]
        self.h_arr = h_arr
        
        tau_coll_list = []
        tau_coll_arr = []
        t_coll_list = []
        t_coll_arr = []
        t_arr = [t_start,]
        t_coll_since_knot_list = []
        for seg_ii in range(n_segments):
            _col_tau = coll_points(point_structure[seg_ii], scheme, precission, order)
            _col_t = tau_to_t_points(
                _col_tau,
                t_knots_and_extremes_arr[seg_ii],
                t_knots_and_extremes_arr[seg_ii+1]
                )
            tau_coll_list.append(array(_col_tau, dtype = 'float64'))
            t_coll_list.append(array(_col_t, dtype = 'float64'))
            tau_coll_arr += _col_tau
            t_coll_arr += _col_t
            t_arr += _col_t
            t_arr.append(t_knots_and_extremes_arr[seg_ii+1])
            _col_t_since_knot = tau_to_t_points(
                _col_tau,
                0,
                h_arr[seg_ii]
                )
            t_coll_since_knot_list.append(array(_col_t_since_knot, dtype = 'float64'))
        
        t_coll_arr = array(t_coll_arr, dtype = 'float64')
        tau_coll_arr = array(tau_coll_arr, dtype = 'float64')
        t_arr = array(t_arr, dtype = 'float64')
        
        
        # ---- lambdas ----
        
        lam_opti = opti.variable(N, self.n_lambdas)



        # ---- distribution of data in self ----
        
        self.opti_arrs = {
            "x": x_opti,
            "x_like_u": x_like_u_opti,
            "x_d": x_dot_opti,
            "x_d_like_u": x_dot_like_u_opti,
            "u": u_opti,
            "u_like_x": u_like_x_opti,
            "t": t_arr,
            "t_col": t_coll_arr,
            "tau_col": tau_coll_arr,
            "t_knot":t_knots_arr,
            "t_knot_ext": t_knots_and_extremes_arr,
            "lam": lam_opti,
        }
        
        self.opti_lists = {
            "x_col" : x_coll_opti_list,
            "x_knot" : x_knot_opti_list,
            "x_dot_col" : x_dot_coll_opti_list,
            "x_dot_knot" : x_dot_knot_opti_list,
            "u_col" : u_coll_opti_list,
            "u_knot" : u_knot_opti_list,
            "t_col" : t_coll_list,
            "tau_col" : tau_coll_list,
            "t_col_since_knot": t_coll_since_knot_list,
        }

        self.opti_points = {
            "x_s": x_opti[0, :],
            "x_e": x_opti[-1, :],
            "x_d_s": x_dot_opti[0, :],
            "x_d_e": x_dot_opti[-1, :],
            "u_s": u_like_x_opti[0, :],
            "u_e": u_like_x_opti[-1, :],
        }
        
        # ---- q and its derivatives as isolated arrays ----
        
        q_and_ders = []
        q_and_ders_like_u = []
        for _ii in range(order):
            q_and_ders.append(x_opti[:, n_q * _ii : n_q * (_ii + 1)])
            q_and_ders_like_u.append(x_like_u_opti[:, n_q * _ii : n_q * (_ii + 1)])
        q_and_ders.append(x_dot_opti[:, -n_q:])
        q_and_ders_like_u.append(x_dot_like_u_opti[:, -n_q:])
        
        q_and_ders_names = get_q_and_ders_names(order)
        self.q_and_ders_names = q_and_ders_names
        
        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            _arr = q_and_ders[_ii]
            self.opti_arrs[_name] = _arr
            self.opti_arrs[_name + "_like_u"] = q_and_ders_like_u[_ii]
            self.opti_points[_name + "_s"] = _arr[0, :]
            self.opti_points[_name + "_e"] = _arr[-1, :]
            
        for _ii in range(order + 1):
            _name = q_and_ders_names[_ii]
            self.opti_lists[_name + '_col'] = []
            self.opti_lists[_name + '_knot'] = []
            
        for seg_ii in range(n_segments):
            _x_coll_segment = x_coll_opti_list[seg_ii]
            _x_dot_coll_segment = x_dot_coll_opti_list[seg_ii]
            for _ii in range(order):
                _name = q_and_ders_names[_ii]
                _key_name = _name + '_col'
                self.opti_lists[_key_name].append(
                    _x_coll_segment[:, n_q * _ii : n_q * (_ii + 1)]
                    )
            _name = q_and_ders_names[-1]
            _key_name = _name + '_col'
            self.opti_lists[_key_name].append(
                _x_dot_coll_segment[:, -n_q:]
                )
            
        for seg_ii in range(n_segments-1):
            _x_knot_segment = x_knot_opti_list[seg_ii]
            _x_dot_knot_segment = x_dot_knot_opti_list[seg_ii]
            for _ii in range(order):
                _name = q_and_ders_names[_ii]
                _key_name = _name + '_knot'
                self.opti_lists[_key_name].append(
                    _x_knot_segment[:, n_q * _ii : n_q * (_ii + 1)]
                    )
            _name = q_and_ders_names[-1]
            _key_name = _name + '_knot'
            self.opti_lists[_key_name].append(
                _x_dot_knot_segment[:, -n_q:]
                )

    def quad_cost(self, arr, arr_c=None, squared=False):
        """
        Calculates a quadrature integration of an array and sets it
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

        dt = self.t_end - self.t_start
        if squared:
            arr = arr**2

        if arr_c is None:
            cost = dt * cas.sum2(
                (cas.sum1(arr[:, :]) + cas.sum1(arr[1:-1, :])) / self.N
            )
        else:
            if squared:
                arr_c = arr_c**2
            cost = dt * cas.sum2(
                (
                    4 * cas.sum1(arr_c[:, :])
                    + cas.sum1(arr[:, :])
                    + cas.sum1(arr[1:-1, :])
                )
                / (3 * self.N)
            )

        self.cost = cost
        self.opti.minimize(cost)

    def sq_cost(self, arr, arr_c=None):
        """
        Calculates a quadrature integration of an array squared and sets it
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

        dt = self.t_end - self.t_start
        arr_sq = arr**2
        if arr_c is None:
            cost = dt * cas.sum2(
                (cas.sum1(arr_sq[:, :]) + cas.sum1(arr_sq[1:-1, :])) / self.N
            )
        else:
            arr_c_sq = arr_c**2
            cost = dt * cas.sum2(
                (
                    4 * cas.sum1(arr_c_sq[:, :])
                    + cas.sum1(arr_sq[:, :])
                    + cas.sum1(arr_sq[1:-1, :])
                )
                / (3 * self.N)
            )

        self.cost = cost
        self.opti.minimize(cost)

    def u_sq_cost(self):
        """
        Calculates a quadrature integration of u squared and sets it
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
        U_sq = U**2

        if "parab" in self.scheme:
            U_c = self.opti_arrs["u_c"]
            if self.scheme == "hsj_parab_mod":
                U_c = self.opti_arrs["u_j"]
                U_j = self.opti_arrs["u_c"]  # OJO EN arrs están guardados al revés!
                # self.opti.subject_to(U_c == (25*U_j + 2*U[1:, :] - 3*U[:-1, :])/24)
                self.opti.subject_to(
                    U_j == (24 * U_c - 2 * U[1:, :] + 3 * U[:-1, :]) / 25
                )

            U_c_sq = U_c**2

            if "j" in self.scheme and self.scheme != "hsj_parab_mod":
                cost = dt * cas.sum2(
                    (
                        25 * cas.sum1(U_c_sq[:, :])
                        + 3 * cas.sum1(U_sq[:-1, :])
                        + 8 * cas.sum1(U_sq[1:, :])
                    )
                    / (18 * self.N)
                )
            else:
                cost = dt * cas.sum2(
                    (
                        4 * cas.sum1(U_c_sq[:, :])
                        + cas.sum1(U_sq[:, :])
                        + cas.sum1(U_sq[1:-1, :])
                    )
                    / (3 * self.N)
                )
        else:
            cost = dt * cas.sum2(
                (cas.sum1(U_sq[:, :]) + cas.sum1(U_sq[1:-1, :])) / self.N
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
        from .piecewise import (
            # euler_accel_restr,
            trapz_accel_restr,
            trapz_mod_accel_restr,
            hs_mod_accel_restr,
            hs_mod_half_x,
            hs_accel_restr,
            hs_half_x,
            hsj_accel_restr,
            hsj_half_x,
            generate_hs_m_funcs,
            generate_trapz_m_func,
        )

        scheme = self.scheme
        try:
            N = self.N
        except AttributeError:
            raise RuntimeError(
                "opti must be set up before applying constraints, use opti_setup()"
            )
        q_and_ders_names = self.q_and_ders_names
        T = self.t_end - self.t_start
        dynam_f_x = self.dyn_f_restr
        # dynam_g_q = self.dyn_g_restr
        x_opti = self.opti_arrs["x"]
        x_dot_opti = self.opti_arrs["x_d"]
        # q_opti = self.opti_arrs["q"]
        # v_opti = self.opti_arrs["v"]
        highest_q_d_opti = self.opti_arrs[q_and_ders_names[-1]]
        u_opti = self.opti_arrs["u"]
        lam_opti = self.opti_arrs["lam"]
        params = self.params
        if "hs" in scheme:
            x_c_opti = self.opti_arrs["x_c"]
            x_c_dot_opti = self.opti_arrs["x_d_c"]
            u_c_opti = self.opti_arrs["u_c"]
            # a_c_opti = self.opti_arrs["a_c"]
            highest_q_d_c_opti = self.opti_arrs[q_and_ders_names[-1] + "_c"]
            lam_c_opti = self.opti_arrs["lam_c"]

            if "j" in scheme:
                half_x = hsj_half_x
            elif "hsn" in scheme:
                half_x = generate_hs_m_funcs(self.order)[0]
            elif "mod" in scheme:
                half_x = hs_mod_half_x
            else:
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
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
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
                if "j" in scheme:
                    for ii in range(N):
                        self.opti.subject_to(
                            u_c_opti[ii, :]
                            == 0.6 * u_opti[ii, :] + 0.4 * u_opti[ii + 1, :]
                        )
                else:
                    for ii in range(N):
                        self.opti.subject_to(
                            u_c_opti[ii, :] == (u_opti[ii, :] + u_opti[ii + 1, :]) / 2
                        )

        # Scheme Constraints
        
        
        if scheme in _radau_like_schemes + _lobato_like_schemes:
            self.opti_points["u_s"] = u_opti[0, :]
        else:
            start_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="start", order=order
            )
            self.opti_points["u_s"] = start_f(u_opti)
            u_like_x_opti = cas.vertcat(self.opti_points["u_s"], u_like_x_opti)

        if scheme in _radau_inv_schemes + _lobato_like_schemes:
            self.opti_points["u_e"] = u_opti[-1, :]
        else:
            end_f = get_bary_extreme_f_cas(
                scheme, n_coll, mode="u", point="end", order=order
            )
            self.opti_points["u_e"] = end_f(u_opti)
            u_like_x_opti = cas.vertcat(u_like_x_opti, self.opti_points["u_e"])
            
            
        restr_schemes = {
            #'euler': euler_accel_restr, #comprobar compatibilidad
            "trapz": trapz_accel_restr,
            "trapz_mod": trapz_mod_accel_restr,
            "trapz_n": generate_trapz_m_func(self.order),
            "hs": hs_accel_restr,
            "hs_mod": hs_mod_accel_restr,
            "hs_parab": hs_accel_restr,
            "hs_mod_parab": hs_mod_accel_restr,
            "hsj": hsj_accel_restr,
            "hsj_parab": hsj_accel_restr,
            "hsj_parab_mod": hsj_accel_restr,
            "hsn": generate_hs_m_funcs(self.order)[1],
            "hsn_parab": generate_hs_m_funcs(self.order)[1],
        }
        n_q = self.n_q
        f_restr = restr_schemes[scheme]
        if "hs" in scheme:
            cas_accel_restr = accelrestriction2casadi(
                f_restr, n_q, n_q, order=self.order
            )
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
                        T / N,
                        highest_q_d_c_opti[ii, :],
                    )
                    == 0
                )
        else:
            cas_accel_restr = accelrestriction2casadi(f_restr, n_q, order=self.order)
            for ii in range(N):
                self.opti.subject_to(
                    cas_accel_restr(
                        x_opti[ii, :],
                        x_opti[ii + 1, :],
                        highest_q_d_opti[ii, :],
                        highest_q_d_opti[ii + 1, :],
                        T / N,
                        [],
                    )
                    == 0
                )
                
# class _Custom:
#     def _check_structure_consistency(
#         self, t_knots_arr, point_structure_q, n_v, n_a, point_structure_v
#     ):
#         """Initial structural checks to ensure compatibilty of given data"""

#         n_q = self.n_q
#         if n_v is None:
#             n_v = n_q
#         self.n_v = n_v
#         if n_a is None:
#             n_a = n_q
#         self.n_a = n_a

#         # t_knot_arr checks
#         t_knots_arr = array(t_knots_arr, dtype="float64")
#         assert len(t_knots_arr.shape) == 1
#         n_segments = t_knots_arr.shape[0] + 1
#         self.n_segments = n_segments
#         t_start = self.t_start
#         t_end = self.t_end
#         assert alltrue(t_knots_arr > t_start)
#         assert alltrue(t_knots_arr < t_end)
#         assert alltrue(t_knots_arr[:-1] <= t_knots_arr[1:])  # is sorted

#         # point_structures checks
#         assert type(point_structure_q) == ndarray
#         assert point_structure_q.dtype == int

#         if point_structure_v is None:
#             point_structure_v = point_structure_q
#         else:
#             assert type(point_structure_v) == ndarray
#             assert point_structure_v.dtype == int

#         q_shape = point_structure_q.shape
#         assert q_shape[1] == n_q
#         assert q_shape[0] == n_segments
#         v_shape = point_structure_v.shape
#         assert v_shape[1] == n_v
#         assert v_shape[0] == n_segments

#         self.point_structure_q = point_structure_q
#         self.point_structure_v = point_structure_v
#         return t_knots_arr, point_structure_v, n_v, n_a

#     def opti_setup(
#         self,
#         t_knots_arr,
#         local_scheme,
#         point_structure_q,
#         tol=1e-16,
#         n_v=None,
#         n_a=None,
#         point_structure_v=None,
#         p_opts=None,
#         s_opts=None,
#     ):
#         """
#          Creates and links the different opti variables to be used in the problem.
#          Requires the function dynamic_setup() to have been run prior.
#          Arrays will be accesible through self.opti_arrs dictionary.
#          Point will be accesible through self.opti_points dictionary.

#          Parameters
#          ----------
#          t_knots_arr : array or list of floats
#              values of time at the knot points.
#          local_scheme: str
#              Discretization scheme applied to each segment. Default names imply
#              a "derivative" approach, in which q polynomials are modeled and its
#              derivatives obtained. Add "_int" to the end of any scheme name to
#              use an "integral" approach, in which highest order derivatives
#              polynomials are modeled and its integrals obtained.
#              Acceptable values are:
#                  "LG" Legendre-Gauss Collocation
#                  "LG_inv" LG Colocation with enpoint instead of startpoint as node point
#                  "LGR" Legendre-Gauss-Radau Collocation
#                  "LGR_inv" LGR with endpoint instead of start point as collocation point
#                  "LGL" Legendre-Gauss-Lobato Collocation
#                  "LGLm" 2nd order modified LGL that only uses interior points as collocation
#                  "LG2" 2nd order modified LG that adds endpoint as node point
#                  "D2" 2nd order modified LGL
#                  "JG" 2nd order Jacobi-Gauss collocation
#          point_structure_q: array of int
#              Matrix containing the number of collocation points on
#              each segment for each component of the configuration.
#              For example, por M components and L segments:
#                  [[N_q1_s1, N_q2_s1, ... , N:qM_s1],
#                   [N_q1_s2, N_q2_s2, ... , N:qM_s2],
#                   ...,
#                   [N_q1_sL, N_q2_sL, ... , N:qM_sL]]
#          n_v : None or int, default None
#              number of velocity variables. If None, will be equal to
#              the number of configuration variables.
#          n_a : None or int, default None
#             number of acceleration variables. If None, will be equal to
#             the number of configuration variables.
#         point_structure_v: None or array of int, default None
#             If None, will be equal to point_structure_q.
#             Matrix containing the number of collocation points on
#             each segment for each component of the velocity.
#             For example, por M components and L segments:
#                 [[N_v1_s1, N_v2_s1, ... , N:vM_s1],
#                  [N_v1_s2, N_v2_s2, ... , N:vM_s2],
#                  ...,
#                  [N_v1_sL, N_v2_sL, ... , N:vM_sL]]


#          Raises
#          ------
#          RuntimeError
#              If mistakenly this function is run before dynamic_setup()

#          Returns
#          -------
#          None.

#         """
#         try:
#             n_q = self.n_q
#             n_u = self.n_u
#         except AttributeError:
#             raise RuntimeError(
#                 "Dynamics must be computed before opti setup, use dynamic_setup()"
#             )

#         assert local_scheme in _all_ps_schemes

#         t_knots_arr, point_structure_v, n_v, n_a = self._check_structure_consistency(
#             t_knots_arr, point_structure_q, n_v, n_a, point_structure_v
#         )

#         n_segments = self.n_segments
#         self.local_scheme = local_scheme

#         opti = cas.Opti()
#         if p_opts is None:
#             if self.verbose:
#                 p_opts = {
#                     "expand": True,
#                 }
#             else:
#                 p_opts = {"expand": True, "ipopt.print_level": 0, "print_time": 0}
#         if s_opts is None:
#             s_opts = {
#                 "max_iter": 10000,
#                 "tol": tol,
#             }  # investigate how to make it work adding 'linear_solver' : "MA27"}
#         opti.solver("ipopt", p_opts, s_opts)
#         self.opti = opti
#         x_opti = []
#         u_opti = []
#         for segment in range(n_segments):
#             x_opti_line = []
#             u_opti_line = []
#             for coordinate in range(n_q):
#                 n_coll = point_structure_q[segment, coordinate]
#                 opt_dict = {
#                     "LGL": [
#                         n_coll,
#                     ],
#                     "D2": [
#                         n_coll,
#                     ],
#                     "LG2": [
#                         n_coll + 2,
#                     ],
#                     "LGLm": [
#                         n_coll + 2,
#                     ],
#                     "LG": [
#                         n_coll + 1,
#                     ],
#                     "LGR": [
#                         n_coll + 1,
#                     ],
#                     "LGR_inv": [
#                         n_coll + 1,
#                     ],
#                     "JG": [
#                         n_coll + 2,
#                     ],
#                 }
#                 N = opt_dict[local_scheme][0]
#                 D_mat = sympy2casadi(
#                     matrix_D_bary(N, local_scheme, precission=16), [], []
#                 )

#                 try:
#                     if scheme in ["LGL", "LG"]:
#                         x_opti = opti.variable(N, 2 * self.n_q)
#                         x_dot_opti = 2 / (t_end - t_start) * D_mat @ x_opti
#                         q_opti = x_opti[:, : self.n_q]
#                         v_opti = x_opti[:, self.n_q :]
#                         a_opti = x_dot_opti[:, self.n_q :]

#                     elif scheme in ["LG2", "D2", "LGLm", "JG"]:
#                         q_opti = opti.variable(N, self.n_q)
#                         v_opti = 2 / (t_end - t_start) * D_mat @ q_opti
#                         a_opti = 2 / (t_end - t_start) * D_mat @ v_opti
#                         x_opti = cas.horzcat(q_opti, v_opti)
#                         x_dot_opti = cas.horzcat(v_opti, a_opti)
#                     else:
#                         raise NotImplementedError(
#                             f"scheme {scheme} not implemented in opti_setup."
#                         )
#                 except AttributeError:
#                     raise RuntimeError(
#                         "Dynamics must be computed before opti setup, use dynamic_setup()"
#                     )

#                 u_opti = opti.variable(n_coll, self.n_u)
#                 tau_arr = array(node_points(N, scheme, precission), dtype="float64")
#                 col_arr = array(
#                     coll_points(n_coll, scheme, precission), dtype="float64"
#                 )
#                 t_arr = ((t_end - t_start) * tau_arr + (t_start + t_end)) / 2
#                 t_col_arr = ((t_end - t_start) * col_arr + (t_start + t_end)) / 2
#                 lam_opti = opti.variable(n_coll, self.n_lambdas)

#         self.opti_arrs = {
#             "x": x_opti,
#             "x_d": x_dot_opti,
#             "q": q_opti,
#             "v": v_opti,
#             "a": a_opti,
#             "u": u_opti,
#             "lam": lam_opti,
#             "tau": tau_arr,
#             "tau_col": col_arr,
#             "t": t_arr,
#             "t_col": t_col_arr,
#         }

#         if scheme == "LGinv":
#             start_p_func = LG_inv_diff_start_p_fun_cas(N)
#             x_start = start_p_func(x_opti)
#             x_dot_start = start_p_func(x_dot_opti)
#             q_start = start_p_func(q_opti)
#             v_start = start_p_func(v_opti)
#             a_start = start_p_func(a_opti)
#         else:
#             x_start = x_opti[0, :]
#             x_dot_start = x_dot_opti[0, :]
#             q_start = q_opti[0, :]
#             v_start = v_opti[0, :]
#             a_start = a_opti[0, :]

#         if scheme == "LG":
#             end_p_func = LG_end_p_fun_cas(N)
#             x_end = end_p_func(x_opti)
#             x_dot_end = end_p_func(x_dot_opti)
#             q_end = end_p_func(q_opti)
#             v_end = end_p_func(v_opti)
#             a_end = end_p_func(a_opti)
#         else:
#             x_end = x_opti[-1, :]
#             x_dot_end = x_dot_opti[-1, :]
#             q_end = q_opti[-1, :]
#             v_end = v_opti[-1, :]
#             a_end = a_opti[-1, :]

#         self.opti_points = {
#             "x_s": x_start,
#             "x_e": x_end,
#             "x_d_s": x_dot_start,
#             "x_d_e": x_dot_end,
#             "q_s": q_start,
#             "q_e": q_end,
#             "v_s": v_start,
#             "v_e": v_end,
#             "a_s": a_start,
#             "a_e": a_end,
#         }

#     def u_sq_cost(self):
#         """
#         Calculates a trapezoidal integration of u squared and sets it
#         as the optimization cost to minimize

#         Requires the functions dynamic_setup() and opti_setup(), in that order,
#         to have been run prior.

#         Raises
#         ------
#         RuntimeError
#             If opti_setup() or dynamic_setup() have not ben run previously

#         Returns
#         -------
#         None.

#         """
#         try:
#             U = self.opti_arrs["u"]
#         except AttributeError:
#             raise RuntimeError(
#                 "opti must be set up before defining cost, use opti_setup()"
#             )

#         dt = self.t_end - self.t_start

#         try:
#             U_c = self.opti_arrs["u_c"]
#             cost = dt * cas.sum2(
#                 (
#                     4 * cas.sum1(U_c[:, :] ** 2)
#                     + cas.sum1(U[:, :] ** 2)
#                     + cas.sum1(U[1:-1, :] ** 2)
#                 )
#                 / (3 * self.N)
#             )
#         except KeyError:
#             cost = dt * cas.sum2(
#                 (cas.sum1(U[:, :] ** 2) + cas.sum1(U[1:-1, :] ** 2)) / self.N
#             )
#         self.cost = cost
#         self.opti.minimize(cost)

#     def apply_scheme(self):
#         """
#         Applies the restrictions corresponding to the selected scheme to
#         the opti variables.

#         Requires the functions dynamic_setup() and opti_setup(), in that order,
#         to have been run prior.

#         Raises
#         ------
#         RuntimeError
#             If opti_setup() or dynamic_setup() have not ben run previously

#         Returns
#         -------
#         None.

#         """
#         from .casadi import accelrestriction2casadi
#         from .piecewise import (
#             euler_accel_restr,
#             trapz_accel_restr,
#             trapz_mod_accel_restr,
#             hs_mod_accel_restr,
#             hs_accel_restr,
#             hs_half_x,
#         )

#         scheme = self.scheme
#         try:
#             N = self.N
#         except AttributeError:
#             raise RuntimeError(
#                 "opti must be set up before applying constraints, use opti_setup()"
#             )
#         T = self.t_end - self.t_start
#         dynam_f_x = self.dyn_f_restr
#         dynam_g_q = self.dyn_g_restr
#         x_opti = self.opti_arrs["x"]
#         x_dot_opti = self.opti_arrs["x_d"]
#         q_opti = self.opti_arrs["q"]
#         v_opti = self.opti_arrs["v"]
#         a_opti = self.opti_arrs["a"]
#         u_opti = self.opti_arrs["u"]
#         lam_opti = self.opti_arrs["lam"]
#         params = self.params
#         if "hs" in scheme:
#             x_c_opti = self.opti_arrs["x_c"]
#             x_c_dot_opti = self.opti_arrs["x_d_c"]
#             u_c_opti = self.opti_arrs["u_c"]
#             a_c_opti = self.opti_arrs["a_c"]
#             lam_c_opti = self.opti_arrs["lam_c"]
#             if "mod" in scheme:
#                 from .piecewise import hs_mod_half_x

#                 half_x = hs_mod_half_x
#             else:
#                 from .piecewise import hs_half_x

#                 half_x = hs_half_x

#         # Dynamics Constraints:
#         for ii in range(N + 1):
#             self.opti.subject_to(
#                 dynam_f_x(
#                     x_opti[ii, :],
#                     x_dot_opti[ii, :],
#                     u_opti[ii, :],
#                     lam_opti[ii, :],
#                     params,
#                 )
#                 == 0
#             )
#         if "hs" in scheme:
#             for ii in range(N):
#                 self.opti.subject_to(
#                     x_c_opti[ii, :]
#                     == half_x(
#                         x_opti[ii, :],
#                         x_opti[ii + 1, :],
#                         a_opti[ii, :],
#                         a_opti[ii + 1, :],
#                         T / N,
#                     )
#                 )
#                 self.opti.subject_to(
#                     dynam_f_x(
#                         x_c_opti[ii, :],
#                         x_c_dot_opti[ii, :],
#                         u_c_opti[ii, :],
#                         lam_c_opti[ii, :],
#                         params,
#                     )
#                     == 0
#                 )
#             if "parab" not in scheme:
#                 for ii in range(N):
#                     self.opti.subject_to(
#                         u_c_opti[ii, :] == (u_opti[ii, :] + u_opti[ii + 1, :]) / 2
#                     )

#         # Scheme Constraints
#         restr_schemes = {
#             #'euler': euler_accel_restr, #comprobar compatibilidad
#             "trapz": trapz_accel_restr,
#             "trapz_mod": trapz_mod_accel_restr,
#             "hs": hs_accel_restr,
#             "hs_mod": hs_mod_accel_restr,
#             "hs_parab": hs_accel_restr,
#             "hs_mod_parab": hs_mod_accel_restr,
#         }
#         n_q = self.n_q
#         f_restr = restr_schemes[scheme]
#         if "hs" in scheme:
#             cas_accel_restr = accelrestriction2casadi(f_restr, n_q, n_q)
#             for ii in range(N):
#                 self.opti.subject_to(
#                     cas_accel_restr(
#                         x_opti[ii, :],
#                         x_opti[ii + 1, :],
#                         a_opti[ii, :],
#                         a_opti[ii + 1, :],
#                         T / N,
#                         a_c_opti[ii, :],
#                     )
#                     == 0
#                 )
#         else:
#             cas_accel_restr = accelrestriction2casadi(f_restr, n_q)
#             for ii in range(N):
#                 self.opti.subject_to(
#                     cas_accel_restr(
#                         x_opti[ii, :],
#                         x_opti[ii + 1, :],
#                         a_opti[ii, :],
#                         a_opti[ii + 1, :],
#                         T / N,
#                         [],
#                     )
#                     == 0
#                 )


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
        self.order = 2


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
        self.order = 2


class _Function_Dynamics:
    def dynamic_setup(self, func_kind, n_q, n_u, n_lambdas=0, order=2):
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
        order : int, default 2
            differential order of the problem. For example, if the function
            is: q''' = G(q, q', q'', u, params), order must be set to 3

        Raises
        ------
        NotImplementedError
            if func_kind is not one of the above stated values.

        Returns
        -------
        None.

        """

        from .piecewise import expand_G, reduce_F

        if n_lambdas != 0:
            raise NotImplementedError("Problems with lambdas are not implemented yet")

        n_p = len(self.params)

        x_sym = cas.SX.sym("x", n_q * order).T
        x_dot_sym = cas.SX.sym("x_dot", n_q * order).T
        # q_sym = cas.SX.sym("q", n_q).T
        # q_dot_sym = cas.SX.sym("q_dot", n_q).T
        # q_dot_dot_sym = cas.SX.sym("q_dot_dot", n_q).T
        q_and_ders = []
        q_and_ders_names = []
        for _ii in range(order + 1):
            _name = "q" + (_ii) * "_dot"
            q_and_ders.append(cas.SX.sym(_name, n_q).T)
            q_and_ders_names.append(_name)
        u_sym = cas.SX.sym("u", n_u).T
        p_sym = cas.SX.sym("p", n_p)
        lam_sym = cas.SX.sym("lam", n_lambdas).T

        if func_kind == "f_x":
            F_x = self.LM
            G_q = reduce_F(F_x, "casadi", order=order)

        elif func_kind == "g_q":
            G_q = self.LM
            F_x = expand_G(G_q, "casadi", order=order)

        elif func_kind == "f_x_impl":
            F_restr = self.LM
            x_restr = F_restr(x_sym, x_dot_sym, u_sym, lam_sym, p_sym)
            if x_restr.shape[0] != 1:
                x_restr = x_restr.T

            x_union = cas.horzcat(*q_and_ders[:-1])
            x_dot_union = cas.horzcat(*q_and_ders[1:])

            q_restr = F_restr(x_union, x_dot_union, u_sym, lam_sym, p_sym)
            if q_restr.shape[0] != 1:
                q_restr = q_restr.T
            q_restr = q_restr[-n_q:]

        elif func_kind == "g_q_impl":
            G_restr = self.LM
            q_restr = G_restr(*q_and_ders, u_sym, lam_sym, p_sym)
            if q_restr.shape[0] != 1:
                q_restr = q_restr.T

            x_list = []
            x_dot_list = []
            for _ii in range(order):
                _jj = _ii + 1
                x_list.append(x_sym[n_q * _ii : n_q * _jj])
                x_dot_list.append(x_dot_sym[n_q * _ii : n_q * _jj])

            x_restr = G_restr(*x_list, x_dot_list[-1], u_sym, lam_sym, p_sym)
            if x_restr.shape[0] != 1:
                x_restr = x_restr.T
            x_d_difs = [x_dot_list[_ii] - x_list[_ii + 1] for _ii in range(order - 1)]
            x_restr = cas.horzcat(x_restr, *x_d_difs)
        else:
            raise NotImplementedError(
                f"function kind {func_kind} not implemented. Implemented kinds are f_x, g_q, f_x_imp and g_q_impl"
            )

        if func_kind in ["f_x", "g_q"]:
            _F = F_x(x_sym, u_sym, p_sym)
            _G = G_q(*q_and_ders[:-1], u_sym, p_sym)
            if _F.shape[0] != 1:
                _F = _F.T
            if _G.shape[0] != 1:
                _G = _G.T
            x_restr = x_dot_sym - _F
            q_restr = q_and_ders[-1] - _G

        dynam_f_x = cas.Function(
            "dynamics_f_x",
            [x_sym, x_dot_sym, u_sym, lam_sym, p_sym],
            [x_restr],
            ["x", "x_dot", "u", "lam", "params"],
            ["residue"],
        )

        dynam_g_q = cas.Function(
            "dynamics_g_q",
            q_and_ders + [u_sym, lam_sym, p_sym],
            [q_restr],
            q_and_ders_names + ["u", "lam", "params"],
            ["residue"],
        )
        self.params_sym = symbols(f"p_0:{n_p}")
        self.dyn_f_restr = dynam_f_x
        self.dyn_g_restr = dynam_g_q
        self.n_u = n_u
        self.n_lambdas = n_lambdas
        self.n_q = n_q
        self.order = order


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
        if self.scheme not in ["LG2", "D2", "LGLm", "JG"]:
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
        elif self.scheme_mode in [
            "bottom-up pseudospectral",
            "top-down pseudospectral",
            "pseudospectral",
        ]:
            N = self.n_arr  # Number of points in X
        else:
            raise NotImplementedError(
                f"Linear init not implemented for scheme mode {self.scheme_mode}"
            )
        T = self.t_end - self.t_start
        q_s = array(q_s, dtype="float64")
        q_e = array(q_e, dtype="float64")
        s_arr = linspace(0, 1, N)
        q_guess = expand_dims(q_s, 0) + expand_dims(s_arr, 1) * expand_dims(
            (q_e - q_s), 0
        )
        q_dot_guess = (q_e - q_s) * ones([N, 1]) / T
        self.opti.set_initial(q_opti, q_guess)
        if self.scheme not in ["LG2", "D2", "LGLm", "JG"]:
            self.opti.set_initial(v_opti, q_dot_guess)
            if self.order > 1 and self.scheme_mode == "equispaced":
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
        if self.scheme not in ["LG2", "D2", "LGLm", "JG"]:
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

                Piecewise Schemes:
            "trapz" : trapezoidal scheme
            "trapz_mod": modified 2nd order compatible trapezoidal scheme
            "hs": Hermite-Simpson scheme
            "hs_mod": modified 2nd order compatible Hermite-Simpson scheme
            "hs_parab": Hermite-Simpson scheme compatible with parabolic U
            "hs_mod_parab": 2nd order compatible Hermite-Simpson scheme with parabolic U
            "hsn" : Hermite-Simpson scheme adapted to order N
            "hsn_parab" : Hermite-Simpson scheme adapted to order N with parabolic U

                Usual Pseudospectral Schemes:
            "LG" Legendre-Gauss Collocation
            "LG_inv" LG Colocation with enpoint instead of startpoint as node point
            "LGR" Legendre-Gauss-Radau Collocation
            "LGR_inv" LGR with endpoint instead of start point as collocation point
            "LGL" Legendre-Gauss-Lobato Collocation
            "LGLm" 2nd order modified LGL that only uses interior points as collocation
            "LG2" 2nd order modified LG that adds endpoint as node point
            "D2" 2nd order modified LGL
            "JG" 2nd order Jacobi-Gauss collocation

                Bottom-Up Pseudospectral Schemes:
            "BU_LG" Legendre-Gauss Collocation
            "BU_LG_inv" LG Colocation with enpoint instead of startpoint as node point
            "BU_LGR" Legendre-Gauss-Radau Collocation
            "BU_LGR_inv" LGR with endpoint instead of start point as collocation point
            "BU_LGL" Legendre-Gauss-Lobato Collocation
            "BU_JG" 2nd order Jacobi-Gauss collocation
            "BU_JGR" 2nd order Jacobi-Gauss-Radau collocation
            "BU_JGR_inv" JGR with endpoint instead of start point as collocation point
            "BU_JGL" 2nd order Jacobi-Gauss-Lobatto collocation

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
    elif scheme in _implemented_bottom_up_pseudospectral_schemes:
        inherit.append(_BU_Pseudospectral)
    elif scheme in _implemented_top_down_pseudospectral_schemes:
        inherit.append(_TD_Pseudospectral)
    elif scheme[:3] == 'ph_':
        inherit.append(_multi_pseudospectral)
    else:
        raise NotImplementedError(
            f"scheme {scheme} not implemented. Valid methods are {_all_implemented_schemes}."
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

    problem = Adequate_Problem(
        LM,
        params,
        scheme,
        ini_guess,
        t_start,
        t_end,
        verbose,
        silent,
    )
    problem.cost = None
    return problem
