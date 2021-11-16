#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:11:43 2021

@author: Siro Moreno
"""

from sympy import legendre_poly, symbols, expand, zeros, lambdify
from functools import lru_cache
from numpy import array, piecewise, linspace


# --- Generating Collocation Points ---


@lru_cache
def LG(N, precission=20):
    return [ii.evalf(n=precission) for ii in legendre_poly(N, polys=True).real_roots()]


@lru_cache
def LGR(N, precission=20):
    pol = legendre_poly(N, polys=True) + legendre_poly(N - 1, polys=True)
    return [ii.evalf(n=precission) for ii in pol.real_roots()]


@lru_cache
def LGL(N, precission=20):
    root_list = [
        ii.evalf(n=precission)
        for ii in legendre_poly(N - 1, polys=True).diff().real_roots()
    ]
    return (
        [-1.0,] + root_list + [1.0,]
    )


@lru_cache
def LGLm(N, precission=20):
    return LGL(N + 2, precission)[1:-1]


@lru_cache
def LG2(N, precission=20):
    return [-1] + LG(N - 2, precission) + [1]


@lru_cache
def coll_points(N, scheme, precission=20):
    """
    Generates a list of len N with values of tau for collocation points

    Parameters
    ----------
    N : int
        Number of collocation points.
    scheme : str
        Scheme name. Supported values are:
            'LG'
            'LGR'
            'LGR_inv'
            'LGL'
            'LGLm'
            'LG2'
            'D2'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    coll_points : list
        list of collocation points
    """
    if scheme == "LG":
        return LG(N, precission)
    elif scheme == "LGR":
        return LGR(N, precission)
    elif scheme == "LGR_inv":
        return [-ii for ii in LGR(N, precission)[::-1]]
    elif scheme in ["LGL", "D2"]:
        return LGL(N, precission)
    elif scheme == "LGLm":
        return LGLm(N, precission)
    elif scheme == "LG2":
        return LG(N, precission)
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: LG, LGR, LGR_inv, LGL, LGLm, LG2, D2"
        )


@lru_cache
def base_points(N, scheme, precission=20):
    """
    Generates a list of len N with values of tau for lagrange base points

    Parameters
    ----------
    N : int
        Number of collocation points.
    scheme : str
        Scheme name. Supported values are:
            'LG'
            'LG_inv'
            'LGR'
            'LGR_inv'
            'LGL'
            'LGLm'
            'LG2'
            'D2'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    base_points : list
        list of base points
    """
    if scheme == "LG":
        return [-1.0] + LG(N, precission)
    elif scheme == "LG_inv":
        return LG(N, precission) + [1.0]
    elif scheme == "LGR":
        return LGR(N, precission) + [1.0]
    elif scheme == "LGR_inv":
        return [-1.0] + [-ii for ii in LGR(N, precission)[::-1]]
    elif scheme in ["LGL", "D2"]:
        return LGL(N, precission)
    elif scheme == "LGLm":
        return [-1.0] + LGLm(N, precission) + [1.0]
    elif scheme == "LG2":
        return LG2(N, precission)
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: LG, LG_inv LGR, LGR_inv, LGL, LGLm, LG2, D2"
        )


# --- Symbolic Lagrange Polynomials ---


#### CREDIT: https://gist.github.com/folkertdev/084c53887c49a6248839 ####
#### Snippet adapted from Folkert de Vries ###

from operator import mul
from functools import reduce, lru_cache


# convenience functions
_product = lambda *args: reduce(mul, *(list(args) + [1]))


# this product may be reusable (when creating many functions on the same domain)
# therefore, cache the result
@lru_cache
def lag_pol(labels, j):
    x = symbols("x")

    def gen(labels, j):
        current = labels[j]
        for m in labels:
            if m == current:
                continue
            yield (x - m) / (current - m)

    return expand(_product(gen(labels, j)))


def lagrangePolynomial(xs, ys):
    """
    Generates a symbolic polynomial that goes through all (x,y) given points

    Parameters
    ----------
    xs : iterable of floats or symbols
        independent coordinates of the points
    ys : iterable of floats or symbols
        dependent coordinates of the points

    Returns
    -------
    total : SymPy symbolic polynomial
        DESCRIPTION.

    """
    # based on https://en.wikipedia.org/wiki/Lagrange_polynomial#Example_1
    total = 0

    # use tuple, needs to be hashable to cache
    xs = tuple(xs)

    for j, current in enumerate(ys):
        t = current * lag_pol(xs, j)
        total += t

    return total


# --- Barycentric Coordinates and Derivation Matrices ---


def _v_sum(t_arr, i):
    """
    Generates the coefficient V for barycentric coordinates

    Parameters
    ----------
    t_arr : iterable of floats
        values of t
    i : int
        index of current point.

    Returns
    -------
    v_i : float
        coefficient V.

    """
    n = len(t_arr)
    prod_coef = [ii for ii in range(n)]
    prod_coef.pop(i)
    v_i = 1.0
    for jj in prod_coef:
        v_i *= t_arr[i] - t_arr[jj]
    return 1.0 / v_i


@lru_cache
def v_coef(N, i, scheme, precission=20):
    """
    Generates the coefficient V for barycentric coordinates

    Parameters
    ----------
    N : int
        Number of base points
    i : int
        index of current point
    scheme : str
        Scheme name. Supported values are:
            'LG'
            'LG_inv'
            'LGR'
            'LGR_inv'
            'LGL'
            'LGLm'
            'LG2'
            'D2'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    v_i : float
        coefficient V.

    """
    taus = base_points(N, scheme, precission)
    return _v_sum(taus, i)


@lru_cache
def matrix_D_bary(N, scheme, precission=20):
    """
    Generates the Derivation Matrix for the given scheme from
    barycentric coordinates

    Parameters
    ----------
    N : int
        Number of base points
    scheme : str
        Scheme name. Supported values are:
            'LG'
            'LG_inv'
            'LGR'
            'LGR_inv'
            'LGL'
            'LGLm'
            'LG2'
            'D2'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    v_i : float
        coefficient V.

    """
    taus = base_points(N, scheme, precission)
    M = zeros(N)
    v_arr = [v_coef(N, ii, scheme, precission) for ii in range(N)]
    for i in range(N):
        j_range = [j for j in range(N)]
        j_range.pop(i)
        for j in j_range:
            M[i, j] = (v_arr[j] / v_arr[i]) / (taus[i] - taus[j])
    for j in range(N):
        M[j, j] = -sum(M[j, :])
    return M


def bary_poly(t_arr, y_arr):
    """
    Generates a numeric function of t that corresponds to the polynomial
    that passes through the points (t, x) using the barycentric formula

    Parameters
    ----------
    t_arr : iterable of floats
        values of t
    y_arr : iterable of floats
        values of y

    Returns
    -------
    polynomial : Function F(t)
        polynomial numerical function

    """
    t = symbols("t")
    n = len(t_arr)
    v_arr = [_v_sum(t_arr, ii) for ii in range(n)]
    sup = 0
    for i in range(n):
        sup += v_arr[i] * y_arr[i] / (t - t_arr[i])
    inf = 0
    for i in range(n):
        inf += v_arr[i] / (t - t_arr[i])
    poly_fun = lambdify([t,], sup / inf)

    def new_poly(t):
        t = array(t, dtype="float64")
        cond_list = [t == t_i for t_i in t_arr]
        func_list = list(y_arr)
        func_list.append(poly_fun)
        return piecewise(t, cond_list, func_list)

    return new_poly


# --- Extreme points of LG scheme ---


@lru_cache
def LG_end_p_fun(N, precission=20):
    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.subs(x, 1)
    return lambdify(coefs, res)


@lru_cache
def LG_diff_end_p_fun(N, precission=20):
    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.diff(x).subs(x, 1)
    return lambdify(coefs, res)


@lru_cache
def LG_inv_start_p_fun(N, precission=20):
    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG_inv", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.subs(x, 0)
    return lambdify(coefs, res)


@lru_cache
def LG_inv_diff_start_p_fun(N, precission=20):
    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG_inv", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.diff(x).subs(x, 0)
    return lambdify(coefs, res)


@lru_cache
def LG_end_p_fun_cas(N, precission=20):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.subs(x, 1)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


@lru_cache
def LG_diff_end_p_fun_cas(N, precission=20):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.diff(x).subs(x, 1)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


@lru_cache
def LG_inv_start_p_fun_cas(N, precission=20):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG_inv", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.subs(x, 0)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


@lru_cache
def LG_inv_diff_start_p_fun_cas(N, precission=20):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = base_points(N, "LG_inv", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.diff(x).subs(x, 0)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


# --- Interpolations and dynamic errors ---


def find_der_polyline(x_n, xp, yp):
    from numpy import searchsorted, where

    n = searchsorted(xp, x_n)
    n = where(n - 1 > 0, n - 1, 0)
    deriv_arr = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    return deriv_arr[n]


def get_pol_u(scheme, uu):
    N = len(uu)
    taus = coll_points(N, scheme)
    pol_u = bary_poly(taus, uu)
    return pol_u


def get_pol_x(scheme, qq, vv, t0, t1):
    N = len(qq)
    tau_x = base_points(N, scheme)
    qq_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ qq
    vv_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ vv
    qq_d_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ qq_d

    pol_q = bary_poly(tau_x, qq)
    pol_v = bary_poly(tau_x, vv)
    pol_q_d = bary_poly(tau_x, qq_d)
    pol_v_d = bary_poly(tau_x, vv_d)
    pol_q_d_d = bary_poly(tau_x, qq_d_d)
    return pol_q, pol_v, pol_q_d, pol_v_d, pol_q_d_d


def extend_x_arrays(qq, vv, scheme):
    N = len(qq)
    if scheme == "LG":
        tau_x = base_points(N, scheme) + [1]
        endp_f = LG_end_p_fun(N)
        qq_1 = float(endp_f(*qq))
        vv_1 = float(endp_f(*vv))
        qq = array(list(qq) + [qq_1,], dtype="float64")
        vv = array(list(vv) + [vv_1,], dtype="float64")
    elif scheme == "LG_inv":
        tau_x = [-1] + base_points(N, scheme)
        startp_f = LG_inv_start_p_fun(N)
        qq_1 = float(startp_f(*qq))
        vv_1 = float(startp_f(*vv))
        qq = array(list(qq) + [qq_1,], dtype="float64")
        vv = array(list(vv) + [vv_1,], dtype="float64")
    else:
        tau_x = base_points(N, scheme)
    return tau_x, qq, vv


def extend_u_array(uu, scheme, N):
    tau_u = base_points(N, scheme)
    if scheme == "LG2":
        uu = array([uu[0]] + list(uu) + [uu[-1]], dtype="float64")
    elif scheme == "LG":
        tau_u = tau_u + [1]
        uu = array([uu[0]] + list(uu) + [uu[-1]], dtype="float64")
    elif scheme == "LG_inv":
        tau_u = [-1] + tau_u
        uu = array([uu[0]] + list(uu) + [uu[-1]], dtype="float64")
    elif scheme == "LGLm":
        uu = array([uu[0]] + list(uu) + [uu[-1]], dtype="float64")
    return tau_u, uu


def get_hermite_x(qq, vv, aa, tau_x, t0, t1):
    from scipy.interpolate import CubicHermiteSpline as hermite

    coll_p = t0 + (1 + array(tau_x, dtype="float64")) * (t1 - t0) / 2
    her_q = hermite(coll_p, qq, vv)
    her_v = hermite(coll_p, vv, aa)
    her_q_d = her_q.derivative()
    her_v_d = her_v.derivative()
    her_q_d_d = her_q_d.derivative()
    return her_q, her_v, her_q_d, her_v_d, her_q_d_d


def dynamic_error_pseudospectral(
    qq,
    vv,
    uu,
    scheme,
    t0,
    t1,
    u_interp="pol",
    x_interp="pol",
    g_func=lambda q, v, u, p: u,
):
    """
    Generates arrays of equispaced points with values of dynamic error.
    
    If x(t) = [q(t), v(t)], and the physics equation states that x' = F(x, u),
    which is equivalent to [q', v'] = [v , G(q, v, u)] we can define the 
    dynamic errors at a point t as:
        dyn_q_err = q'(t) - v(t)
        dyn_v_err = v'(t) - G(q(t), v(t), u(t))
        dyn_2_err = q''(t) - G(q(t), v(t), u(t))
        
    'x_interp' and 'u_interp' define the way in which we interpolate the values
    of q, v and u between the given points.

    Parameters
    ----------
    qq : Numpy Array, shape = (W, N)
        Values known of q(t)
    vv : Numpy Array, shape = (W, N)
        Values known of v(t)
    uu : Numpy Array, shape = (Y, [Z])
        Values known of x(t)
    scheme : str, optional
        Pseudospectral cheme used in the optimization.
        Acceptable values are:
            'LG'
            'LG_inv'
            'LGR'
            'LGR_inv'
            'LGL'
            'LGLm'
            'LG2'
            'D2'
    t0 : float
        starting time of interval of analysis
    t1 : float
        ending time of interval of analysis
    u_interp :  string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "smooth": 3d order spline interpolation
    x_interp : string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "Hermite": Hermite's 3d order spline interpolation
    g_func : Function of (q, v, u, params)
        A function of a dynamic sistem, so that
            v' = g(q, v, u, params)

    Raises
    ------
    NameError
        When an unsupported value for scheme, x_interp or u_interp is used.

    Returns
    -------
    err_q : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error q'(t) - v(t).
    err_v : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error v'(t) - G(q(t), v(t), u(t)).
    err_2 : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error q''(t) - G(q(t), v(t), u(t)).

    """
    from scipy.interpolate import CubicHermiteSpline as hermite
    from numpy import interp, gradient, zeros_like

    N = len(qq)
    scheme_opts = ["LG", "LG_inv", "LGR", "LGR_inv", "LGL", "D2", "LG2", "LGLm"]
    if scheme not in scheme_opts:
        NameError(f"Invalid scheme.\n valid options are {scheme_opts}")
    t_arr = linspace(-1, 1, 1000)
    if u_interp == "pol":
        pol_u = get_pol_u(scheme, N, uu)
        u_arr = pol_u(t_arr)
    elif u_interp == "lin":
        tau_u, uu = extend_u_array(uu, scheme, N)
        u_arr = interp(t_arr, tau_u, uu)
    elif u_interp == "smooth":
        tau_u, uu = extend_u_array(uu, scheme, N)
        uu_dot = gradient(uu, tau_u)
        u_arr = hermite(tau_u, uu, uu_dot)(t_arr)
    else:
        raise NameError(
            'Invalid interpolation method for u.\n valid options are "pol", "lin", "smooth"'
        )

    if x_interp == "pol":
        tau_x = base_points(N, scheme)
        pol_q, pol_v, pol_q_d, pol_v_d, pol_q_d_d = get_pol_x(scheme, qq, vv, t0, t1)
        q_arr = pol_q(t_arr)
        v_arr = pol_v(t_arr)
        q_arr_d = pol_q_d(t_arr)
        v_arr_d = pol_v_d(t_arr)
        q_arr_d_d = pol_q_d_d(t_arr)
    elif x_interp == "lin":
        tau_x, qq, vv = extend_x_arrays(qq, vv, scheme)
        q_arr = interp(t_arr, tau_x, qq)
        v_arr = interp(t_arr, tau_x, vv)
        coll_p = t0 + (1 + array(tau_x, dtype="float64")) * (t1 - t0) / 2
        t_arr_lin = linspace(t0, t1, 1000)
        q_arr_d = find_der_polyline(t_arr_lin, coll_p, qq)
        v_arr_d = find_der_polyline(t_arr_lin, coll_p, vv)
        q_arr_d_d = zeros_like(q_arr)
    elif x_interp == "Hermite":
        tau_x, qq, vv = extend_x_arrays(qq, vv, scheme)
        aa = g_func(qq, vv, uu)
        her_q, her_v, her_q_d, her_v_d, her_q_d_d = get_hermite_x(
            qq, vv, aa, tau_x, t0, t1
        )
        t_arr_lin = linspace(t0, t1, 1000)
        q_arr = her_q(t_arr_lin)
        v_arr = her_v(t_arr_lin)
        q_arr_d = her_q_d(t_arr_lin)
        v_arr_d = her_v_d(t_arr_lin)
        q_arr_d_d = her_q_d_d(t_arr_lin)
    else:
        raise NameError(
            'Invalid interpolation method for x.\n valid options are "pol", "lin", "Hermite"'
        )

    err_q = q_arr_d - v_arr
    err_v = v_arr_d - g_func(q_arr, v_arr, u_arr)
    err_2 = q_arr_d_d - g_func(q_arr, v_arr, u_arr)

    return err_q, err_v, err_2
