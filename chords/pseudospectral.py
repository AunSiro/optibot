#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:11:43 2021

@author: Siro Moreno

Here we define functions needed to operate with pseudospectral collocations
schemes. In order to keep the best accuracy in interpolations, barycentric
formulas are constructed.
"""

from sympy import legendre_poly, jacobi_poly, symbols, expand, zeros, lambdify
from functools import lru_cache
from numpy import (
    array,
    piecewise,
    linspace,
    expand_dims,
    squeeze,
    zeros_like,
    eye,
    arange,
    cos,
    pi,
)
from numpy import sum as npsum
from numba import njit
from .numpy import combinefunctions, store_results
from .piecewise import interp_2d


# --- Generating Collocation Points ---

_gauss_like_schemes = ["LG", "JG", "CG"]
_gauss_inv_schemes = [_sch + "_inv" for _sch in _gauss_like_schemes]
_gauss_2_schemes = [_sch + "2" for _sch in _gauss_like_schemes] + ["LGLm"]
_radau_like_schemes = ["LGR", "JGR", "CGR"]
_radau_inv_schemes = [_sch + "_inv" for _sch in _radau_like_schemes]
_lobato_like_schemes = ["LGL", "JGL", "CGL"]
_other_schemes = ["D2"]
_implemented_schemes = (
    _gauss_like_schemes
    + _gauss_inv_schemes
    + _gauss_2_schemes
    + _radau_like_schemes
    + _radau_inv_schemes
    + _lobato_like_schemes
    + _other_schemes
)


@lru_cache(maxsize=2000)
@store_results
def LG(N, precission=16):
    return [ii.evalf(n=precission) for ii in legendre_poly(N, polys=True).real_roots()]


@lru_cache(maxsize=2000)
@store_results
def LGR(N, precission=16):
    pol = legendre_poly(N, polys=True) + legendre_poly(N - 1, polys=True)
    return [ii.evalf(n=precission) for ii in pol.real_roots()]


@lru_cache(maxsize=2000)
@store_results
def LGL(N, precission=16):
    root_list = [
        ii.evalf(n=precission)
        for ii in legendre_poly(N - 1, polys=True).diff().real_roots()
    ]
    return (
        [
            -1.0,
        ]
        + root_list
        + [
            1.0,
        ]
    )


@lru_cache(maxsize=2000)
@store_results
def LGLm(N, precission=16):
    return LGL(N + 2, precission)[1:-1]


@lru_cache(maxsize=2000)
@store_results
def LG2(N, precission=16):
    return [-1] + LG(N - 2, precission) + [1]


@lru_cache(maxsize=2000)
@store_results
def JG(N, order=2, precission=16):
    return [
        ii.evalf(n=precission)
        for ii in jacobi_poly(N, order - 1, 0, polys=True).real_roots()
    ]


@lru_cache(maxsize=2000)
@store_results
def JGR(N, order=2, precission=16):
    return [-1.0] + [
        ii.evalf(n=precission)
        for ii in jacobi_poly(N - 1, order - 1, 1, polys=True).real_roots()
    ]


@lru_cache(maxsize=2000)
@store_results
def JGR_inv(N, order=2, precission=16):
    return [
        ii.evalf(n=precission)
        for ii in jacobi_poly(N - 1, order, 0, polys=True).real_roots()
    ] + [1.0]


@lru_cache(maxsize=2000)
@store_results
def JGL(N, order=2, precission=16):
    return (
        [-1.0]
        + [
            ii.evalf(n=precission)
            for ii in jacobi_poly(N - 2, order, 1, polys=True).real_roots()
        ]
        + [1.0]
    )


@lru_cache(maxsize=2000)
@store_results
def JG2(N, precission=16):
    return [-1] + JG(N - 2, precission) + [1]


def CG(N):
    theta = pi * (2 * arange(N, dtype="float64") + 1) / N / 2
    return list(cos(theta)[::-1])


def CGL(N):
    theta = pi * (arange(0, N, dtype="float64")) / (N - 1)
    return list(cos(theta)[::-1])


def CGR(N):
    theta = pi * (2 * arange(N, dtype="float64")) / (2 * N - 1)
    return list(-cos(theta))


@lru_cache(maxsize=2000)
def coll_points(N, scheme, precission=16, order=2):
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
            'JG'
    precission: int, default 16
        number of decimal places of precission

    Returns
    -------
    coll_points : list
        list of collocation points
    """
    if scheme in ["LG", "LG_inv", "LG2"]:
        return LG(N, precission)
    elif scheme == "LGR":
        return LGR(N, precission)
    elif scheme == "LGR_inv":
        return [-ii for ii in LGR(N, precission)[::-1]]
    elif scheme in ["LGL", "D2"]:
        return LGL(N, precission)
    elif scheme == "LGLm":
        return LGLm(N, precission)
    elif scheme in ["JG", "JG_inv", "JG2"]:
        return JG(N, order, precission)
    elif scheme == "JGR":
        return JGR(N, order, precission)
    elif scheme == "JGR_inv":
        return JGR_inv(N, order, precission)
    elif scheme == "JGL":
        return JGL(N, order, precission)
    elif scheme in ["CG", "CG_inv", "CG2"]:
        return CG(N)
    elif scheme == "CGR":
        return CGR(N)
    elif scheme == "CGR_inv":
        return [-ii for ii in CGR(N)[::-1]]
    elif scheme == "CGL":
        return CGL(N)
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are:{_implemented_schemes}"
        )


@lru_cache(maxsize=2000)
def node_points(N, scheme, precission=16):
    """
    Generates a list of len N with values of tau for lagrange node points

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
            'JG'
    precission: int, default 16
        number of decimal places of precission

    Returns
    -------
    node_points : list
        list of node points
    """

    if scheme in (_gauss_like_schemes + _radau_inv_schemes):
        coll_p = coll_points(N - 1, scheme, precission, order=2)
        return [-1.0] + coll_p
    elif scheme in (_gauss_inv_schemes + _radau_like_schemes):
        coll_p = coll_points(N - 1, scheme, precission, order=2)
        return coll_p + [1.0]
    elif scheme in _lobato_like_schemes:
        coll_p = coll_points(N, scheme, precission, order=2)
        return coll_p
    elif scheme in _gauss_2_schemes:
        coll_p = coll_points(N - 2, scheme, precission, order=2)
        return [-1.0] + coll_p + [1.0]
    elif scheme in _other_schemes:
        if scheme in [
            "D2",
        ]:
            return LGL(N, precission)
        else:
            raise NotImplementedError(
                f"scheme {scheme} in category 'others' is not yet implemented"
            )
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


def get_coll_indices_from_nodes(scheme):
    """
    returns a slice that can be used to separate collocation points from
    an array that includes first and last point

    Parameters
    ----------
    scheme : str
        the scheme used.

    Raises
    ------
    NotImplementedError
        When the scheme is not recognized.

    Returns
    -------
    coll_index : slice
        slice to be used as index in the array to extract the collocation points.

    """
    if scheme in (_gauss_like_schemes + _radau_inv_schemes):
        coll_index = slice(1, None)
    elif scheme in (_gauss_inv_schemes + _radau_like_schemes):
        coll_index = slice(None, -1)
    elif scheme in _lobato_like_schemes:
        coll_index = slice(None, None)
    elif scheme in _gauss_2_schemes:
        coll_index = slice(1, -1)
    elif scheme in _other_schemes:
        if scheme in [
            "D2",
        ]:
            coll_index = slice(None, None)
        else:
            raise NotImplementedError(
                f"scheme {scheme} in category 'others' is not yet implemented"
            )
    else:
        raise NotImplementedError(
            f"Scheme {scheme} not implemented yet, valid schemes are: {_implemented_schemes}"
        )
    return coll_index


# --- Symbolic Lagrange Polynomials ---


#### CREDIT: https://gist.github.com/folkertdev/084c53887c49a6248839 ####
#### Snippet adapted from Folkert de Vries ###

from operator import mul
from functools import reduce, lru_cache


# convenience functions
_product = lambda *args: reduce(mul, *(list(args) + [1]))


# this product may be reusable (when creating many functions on the same domain)
# therefore, cache the result
@lru_cache(maxsize=2000)
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
    # noded on https://en.wikipedia.org/wiki/Lagrange_polynomial#Example_1
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


@lru_cache(maxsize=None)
@store_results
def v_coef(N, i, scheme, precission=16, order=2):
    """
    Generates the coefficient V for barycentric coordinates for
    Polynomials constructed over node points.

    Parameters
    ----------
    N : int
        Number of node points
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
    precission: int, default 16
        number of decimal places of precission

    Returns
    -------
    v_i : float
        coefficient V.

    """
    if order > 2:
        raise ValueError(
            "You are trying to calculate a barycentric polynomial"
            + " over node points for a differential order larger than 2"
            + ", but node points are only defined or order up to 2."
        )
    taus = node_points(N, scheme, precission)
    return _v_sum(taus, i)


@lru_cache(maxsize=None)
@store_results
def v_coef_coll(N, i, scheme, precission=16, order=2):
    """
    Generates the coefficient V for barycentric coordinates for
    Polynomials constructed over collocation points.

    Parameters
    ----------
    N : int
        Number of node points
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    v_i : float
        coefficient V.

    """
    taus = coll_points(N, scheme, precission, order)
    return _v_sum(taus, i)


@lru_cache(maxsize=2000)
@store_results
def matrix_D_bary(N, scheme, precission=16):
    """
    Generates the Derivation Matrix for the given scheme from
    barycentric coordinates

    Parameters
    ----------
    N : int
        Number of node points
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
    precission: int, default 16
        number of decimal places of precission

    Returns
    -------
    M : NumPy Array
        Derivation Matrix.

    """
    from numpy import zeros

    taus = node_points(N, scheme, precission)
    M = zeros((N, N), dtype="float64")
    v_arr = [v_coef(N, ii, scheme, precission) for ii in range(N)]
    for i in range(N):
        j_range = [j for j in range(N)]
        j_range.pop(i)
        for j in j_range:
            M[i, j] = (v_arr[j] / v_arr[i]) / (taus[i] - taus[j])
    for j in range(N):
        M[j, j] = -sum(M[j, :])
    return M


# def bary_poly(t_arr, y_arr):
#     """
#     Generates a numeric function of t that corresponds to the polynomial
#     that passes through the points (t, y) using the barycentric formula

#     Parameters
#     ----------
#     t_arr : iterable of floats
#         values of t
#     y_arr : iterable of floats
#         values of y

#     Returns
#     -------
#     polynomial : Function F(t)
#         polynomial numerical function

#     """
#     t = symbols("t")
#     n = len(t_arr)
#     v_arr = [_v_sum(t_arr, ii) for ii in range(n)]
#     sup = 0
#     for i in range(n):
#         sup += v_arr[i] * y_arr[i] / (t - t_arr[i])
#     inf = 0
#     for i in range(n):
#         inf += v_arr[i] / (t - t_arr[i])
#     poly_fun = lambdify(
#         [
#             t,
#         ],
#         sup / inf,
#     )

#     def new_poly(t):
#         t = array(t, dtype="float64")
#         cond_list = [t == t_i for t_i in t_arr]
#         func_list = list(y_arr)
#         func_list.append(poly_fun)
#         return piecewise(t, cond_list, func_list)

#     return new_poly


def bary_poly(t_arr, y_arr):
    """
    Generates a numeric function of t that corresponds to the polynomial
    that passes through the points (t, y) using the barycentric formula

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
    # t = symbols("t")
    t_arr = array(t_arr, dtype="float64")
    y_arr = array(y_arr, dtype="float64")
    n = len(t_arr)
    v_arr = [_v_sum(t_arr, ii) for ii in range(n)]
    v_arr = array(v_arr, dtype="float64")
    v_y_arr = v_arr * y_arr
    t_dict = {t_arr[ii]: y_arr[ii] for ii in range(n)}

    # @njit
    def poly_fun(t=0.0):
        t_sub_arr = t - t_arr
        # sup = 0.0
        # for i in range(n):
        #     sup += v_arr[i] * y_arr[i] / (t - t_arr[i])
        sup = npsum(v_y_arr / t_sub_arr)
        # inf = 0
        # for i in range(n):
        #     inf += v_arr[i] / (t - t_arr[i])
        inf = npsum(v_arr / t_sub_arr)
        return sup / inf

    @lru_cache(maxsize=2000)
    def poly_fun_con(t):
        if t in t_dict:
            result = t_dict[t]
        else:
            result = poly_fun(t)
        return result

    def new_poly(t):
        t_arr = array(t, dtype="float64")
        if t_arr.size > 1:
            result = zeros_like(t_arr)
            for ii, tt in enumerate(t_arr):
                result[ii] = poly_fun_con(tt)
        else:
            result = poly_fun_con(float(t))
        return result

    return new_poly


def bary_poly_2d(t_arr, y_arr):
    """
    Generates a numeric function of t that corresponds to the polynomials
    that passes through the points (t, y) using the barycentric formula
    for each column n in y_arr(:, n)

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
    if len(y_arr.shape) == 1:
        return bary_poly(t_arr, y_arr)

    dim = y_arr.shape[-1]
    pols = [bary_poly(t_arr, y_arr[:, ii]) for ii in range(dim)]
    return combinefunctions(*pols)


@lru_cache(maxsize=2000)
@store_results
def unit_Lag_pol(N, scheme, n, kind="q", precission=16):
    """
    Generate a barycentric numeric Lagrange polynomial over N node points.
    L_n(x_i) = 0 for i != n
    L_n(x_n) = 1

    Parameters
    ----------
    N : Int
        Number of node points.
    scheme : str
        Name of pseudospectral scheme.
    n : int
        number for which L(x_n) = 1.
    kind : str: "q" or "u", optional
        Whether the node points are the collocation points ("u") or the node
        points of the given scheme ("q"). The default is "q".
    precission : int, optional
        Precission in colloction point calculation. The default is 20.

    Returns
    -------
    Function
        Barycentric Lagrange Polynomial L_n(x).

    """
    assert kind in ["q", "u"]
    if kind == "q":
        x = node_points(N, scheme, precission)
    else:
        x = coll_points(N, scheme, precission)

    y = zeros(N)
    y[n] = 1
    return bary_poly(x, y)


@lru_cache(maxsize=2000)
@store_results
def vector_interpolator(
    N_from, N_to, scheme_from, scheme_to, n, kind="q", precission=16
):
    """
    Generates a vector that multiplied by the q coordinates gives the value
    at the n-th point of a different scheme

    Parameters
    ----------
    N_from : Int
        DESCRIPTION.
    N_to : Int
        DESCRIPTION.
    scheme_from : str
        DESCRIPTION.
    scheme_to : str
        DESCRIPTION.
    n : int
        DESCRIPTION.
    kind : str "q" or "u", optional
        DESCRIPTION. The default is 'q'.
    precission : int, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    vec : numpy array
        DESCRIPTION.

    """

    assert kind in ["q", "u"]
    vec = zeros(N_from)
    if N_from == N_to and scheme_from == scheme_to:
        vec[n] = 1
    else:
        if kind == "q":
            node_points_to = node_points(N_to, scheme_to, precission)
        else:
            node_points_to = coll_points(N_to, scheme_to, precission)
        point_to = node_points_to[n]
        for ii in range(N_from):
            Lag_pol_ii = unit_Lag_pol(N_from, scheme_from, ii, kind, precission)
            vec[ii] = Lag_pol_ii(point_to)

    return vec


# --- Extreme points of LG scheme ---


def get_bary_extreme_f(scheme, N, mode="u", point="start", order=2):
    """
    Create a function that calculates the value of a polynomial at
    an extreme point when given the value at construction points.

    Parameters
    ----------
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
            'JG'
            'JGR'
            'JGR_inv'
            'JGL'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
    N : int
        Number of points that construct the polynomial
    mode : {'u', 'x'}
        u polynomials are constructed on collocation points, while x polynomials
        are constructed on node points
    point : {'start', 'end'}
        which point is to be calculated
    order :
        for Jacobi points, differential order of the scheme

    Returns
    -------
    Function(values)
        A function that will calculate the value at the asked point when the
        value of the construction points are [values]

    """

    if point == "start":
        if mode == "u":
            if scheme in ["LGL", "D2", "LGR", "JGR", "JGL", "CGL", "CGR"]:
                return lambda coefs: coefs[0]
        elif mode == "x":
            if scheme in [
                "LGL",
                "D2",
                "LGR",
                "LGR_inv",
                "LG",
                "LG2",
                "LGLm",
                "JG",
                "CG",
                "CGR",
                "CGR_inv",
                "CGL",
            ]:
                return lambda coefs: coefs[0]
        else:
            raise ValueError(f"Invalid mode {mode}, accepted are u and x")
    elif point == "end":
        if mode == "u":
            if scheme in ["LGL", "D2", "LGR_inv", "JGR_inv", "JGL", "CGL", "CGR_inv"]:
                return lambda coefs: coefs[-1]
        elif mode == "x":
            if scheme in [
                "LGL",
                "D2",
                "LGR",
                "LGR_inv",
                "LG_inv",
                "LG2",
                "LGLm",
                "JG",
                "JG_inv",
                "CG",
                "CGR",
                "CGR_inv",
                "CGL",
            ]:
                return lambda coefs: coefs[-1]
        else:
            raise ValueError(f"Invalid mode {mode}, accepted are u and x")
    else:
        raise ValueError(f"Invalid point {point}, accepted are start and end")

    p = 1 if point == "end" else -1
    v_gen = v_coef_coll if mode == "u" else v_coef
    precission = 16

    v_arr = [v_gen(N, ii, scheme, precission, order) for ii in range(N)]
    t_arr = coll_points(N, scheme, precission, order)
    # Barycentric Formula: Sup: Superior, Inf:Inferior
    sup = []
    for i in range(N):
        sup.append(float(v_arr[i] / (p - t_arr[i])))
    inf = 0
    for i in range(N):
        inf += v_arr[i] / (p - t_arr[i])
    inf = float(inf)

    def extpoint(coefs):
        numsup = 0
        for ii in range(N):
            numsup += coefs[ii] * sup[ii]
        return numsup / inf

    return extpoint


# @lru_cache(maxsize=2000)
# def LG_end_p_fun(N, precission=16):
#     coefs = symbols(f"c_0:{N}")
#     taus = node_points(N, "LG", precission)
#     x = symbols("x")
#     pol_lag = lagrangePolynomial(taus, coefs)
#     res = pol_lag.subs(x, 1)
#     return lambdify(coefs, res)


@lru_cache(maxsize=2000)
@store_results
def LG_diff_end_p_fun(N, precission=16):
    coefs = symbols(f"c_0:{N}")
    taus = node_points(N, "LG", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.diff(x).subs(x, 1)
    return lambdify(coefs, res)


# @lru_cache(maxsize=2000)
# def LG_inv_start_p_fun(N, precission=16):
#     coefs = symbols(f"c_0:{N}")
#     taus = node_points(N, "LG_inv", precission)
#     x = symbols("x")
#     pol_lag = lagrangePolynomial(taus, coefs)
#     res = pol_lag.subs(x, 0)
#     return lambdify(coefs, res)


@lru_cache(maxsize=2000)
@store_results
def LG_inv_diff_start_p_fun(N, precission=16):
    coefs = symbols(f"c_0:{N}")
    taus = node_points(N, "LG_inv", precission)
    x = symbols("x")
    pol_lag = lagrangePolynomial(taus, coefs)
    res = pol_lag.diff(x).subs(x, 0)
    return lambdify(coefs, res)


@lru_cache(maxsize=2000)
@store_results
def LG_end_p_fun_cas(N, precission=16):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    x_cas = SX.sym("x", N)
    x_sympy = symbols(f"c0:{N}")
    fun = get_bary_extreme_f("LG", N, mode="x", point="end")
    sympy_expr = fun(x_sympy)
    cas_expr = sympy2casadi(sympy_expr, x_sympy, vertsplit(x_cas))
    cas_f = Function(
        "x_poly_endpoint",
        [
            x_cas,
        ],
        [
            cas_expr,
        ],
    )
    return cas_f


@lru_cache(maxsize=2000)
@store_results
def LG_diff_end_p_fun_cas(N, precission=16):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = node_points(N, "LG", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.diff(x).subs(x, 1)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


@lru_cache(maxsize=2000)
@store_results
def LG_inv_start_p_fun_cas(N, precission=16):
    _f = LG_end_p_fun_cas(N, precission)

    def cas_f(x_cas):
        return _f(x_cas[::-1, :])

    return cas_f


@lru_cache(maxsize=2000)
@store_results
def LG_inv_diff_start_p_fun_cas(N, precission=16):
    from casadi import SX, vertsplit, Function
    from .casadi import sympy2casadi

    coefs = symbols(f"c_0:{N}")
    taus = node_points(N, "LG_inv", precission)
    pol_lag = lagrangePolynomial(taus, coefs)
    x = symbols("x")
    res = pol_lag.diff(x).subs(x, 0)
    x_cas = SX.sym("x", N)
    res_cas = sympy2casadi(res, coefs, vertsplit(x_cas))
    return Function("dynamics_x", [x_cas], [res_cas])


# --- Interpolations and dynamic errors ---


def find_der_polyline(x_n, xp, yp):
    """
    Generates a lineal interpolation that passes through  points xp,yp
    Then returns values of slope at points x_n
    """
    from numpy import searchsorted, where

    n = searchsorted(xp, x_n)
    n = where(n - 1 > 0, n - 1, 0)
    dim = len(yp.shape)
    if dim == 1:
        deriv_arr = (yp[1:] - yp[:-1]) / (xp[1:] - xp[:-1])
    elif dim == 2:
        deriv_arr = (yp[1:] - yp[:-1]) / expand_dims(xp[1:] - xp[:-1], 1)
    return deriv_arr[n]


def get_pol_u(scheme, uu):
    """
    Generates a numerical function of a polynomial for interpolating
    u, valid in tau = (-1, 1)
    """
    N = len(uu)
    taus = array(coll_points(N, scheme), dtype="float")
    pol_u = bary_poly_2d(taus, uu)
    return pol_u


def get_pol_x(scheme, qq, vv, t0, t1):
    """
    Generates numerical functions of polynomials for interpolating
    x and its derivatives, valid in tau = (-1, 1)

    Parameters
    ----------
    scheme : str
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
    qq : Numpy Array
        Values known of q(t)
    vv : Numpy Array
        Values known of v(t)
    t0 : float
        starting time of interval of analysis
    t1 : float
        ending time of interval of analysis

    Returns
    -------
    pol_q : function(tau)
       polynomial interpolation of q(tau)
    pol_v : function(tau)
       polynomial interpolation of v(tau)
    pol_q_d : function(tau)
       polynomial interpolation of q'(tau)
    pol_v_d : function(tau)
       polynomial interpolation of v'(tau)
    pol_q_d_d : function(tau)
       polynomial interpolation of q''(tau)

    """
    qq = array(qq)
    N = qq.shape[0]
    tau_x = array(node_points(N, scheme), dtype="float")
    qq_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ qq
    vv_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ vv
    qq_d_d = 2 / (t1 - t0) * matrix_D_bary(N, scheme) @ qq_d

    pol_q = bary_poly_2d(tau_x, qq)
    pol_v = bary_poly_2d(tau_x, vv)
    pol_q_d = bary_poly_2d(tau_x, qq_d)
    pol_v_d = bary_poly_2d(tau_x, vv_d)
    pol_q_d_d = bary_poly_2d(tau_x, qq_d_d)
    return pol_q, pol_v, pol_q_d, pol_v_d, pol_q_d_d


def extend_x_arrays(qq, vv, scheme):
    """
    In the case that the scheme doesn't consider either extreme point as a node point,
    the value of q and v at said point is calculated and added to the arrays.
    A modified tau list compatible is also calculated.
    If both extremes are node points for the given scheme, unmodified arrays
    of q and v are returned along with the usual tau list.

    Parameters
    ----------
    qq : Numpy Array
        Values known of q(t)
    vv : Numpy Array
        Values known of v(t)
    scheme : str
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'

    Returns
    -------
    tau_x : list
        Tau values coherent with the new q and v arrays
    new_qq : Numpy Array
        Values known of q(t)
    new_vv : Numpy Array
        Values known of v(t)

    """
    N = len(qq)
    if scheme in ["LG", "CG"]:
        tau_x = node_points(N, scheme) + [1]
        endp_f = get_bary_extreme_f(scheme, N, mode="x", point="end")
        qq_1 = array(endp_f(qq), dtype="float")
        vv_1 = array(endp_f(vv), dtype="float")
        new_qq = array(
            list(qq)
            + [
                qq_1,
            ],
            dtype="float64",
        )
        new_vv = array(
            list(vv)
            + [
                vv_1,
            ],
            dtype="float64",
        )
    elif scheme in ["LG_inv", "CG_inv"]:
        tau_x = [-1] + node_points(N, scheme)
        startp_f = get_bary_extreme_f(scheme, N, mode="x", point="start")
        qq_1 = array(startp_f(qq), dtype="float")
        vv_1 = array(startp_f(vv), dtype="float")
        new_qq = array(
            list(qq)
            + [
                qq_1,
            ],
            dtype="float64",
        )
        new_vv = array(
            list(vv)
            + [
                vv_1,
            ],
            dtype="float64",
        )
    else:
        tau_x = node_points(N, scheme)
        new_qq = qq
        new_vv = vv
    return tau_x, new_qq, new_vv


def extend_u_array(uu, scheme, N, order=2):
    """
    In the case that the scheme doesn't consider either extreme point as a
    collocation point, the value of u at said points is extrapolated by
    duplicating the nearest known value and added to the array.
    A modified tau list compatible is also calculated.
    If both extremes are collocation points for the given scheme, unmodified
    array of u are returned along with the usual tau list.

    Parameters
    ----------
    uu : Numpy Array
        Values known of u(t)
    scheme : str
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'

    Returns
    -------
    tau_u : list
        Tau values coherent with the new u arrays
    new_uu : Numpy Array
        Values known of u(t)

    """
    tau_u = coll_points(N, scheme, order)
    n_col = uu.shape[0]
    uu_0 = get_bary_extreme_f(scheme, n_col, mode="u", point="start", order=order)(uu)
    uu_e = get_bary_extreme_f(scheme, n_col, mode="u", point="end", order=order)(uu)
    if scheme in ["LG2", "LG", "JG", "LG_inv", "LGLm", "CG", "CG_inv"]:
        tau_u = [-1.0] + tau_u + [1.0]
        new_uu = array([uu_0] + list(uu) + [uu_e], dtype="float64")
    elif scheme in ["LGR", "JGR", "CGR"]:
        tau_u = tau_u + [1.0]
        new_uu = array(list(uu) + [uu_e], dtype="float64")
    elif scheme in ["LGR_inv", "JGR_inv", "CGR_inv"]:
        tau_u = [-1.0] + tau_u
        new_uu = array([uu_0] + list(uu), dtype="float64")
    elif scheme in ["LGL", "D2", "JGL", "CGL"]:
        new_uu = uu
    else:
        raise ValueError("Unrecognized scheme")
    return tau_u, new_uu


def get_hermite_x(qq, vv, aa, tau_x, t0, t1):
    """
    Returns Scipy hermite functions that interpolate q, v, q', v', q'' in
    the interval t = (t0, t1)
    """
    from scipy.interpolate import CubicHermiteSpline as hermite

    coll_p = t0 + (1 + array(tau_x, dtype="float64")) * (t1 - t0) / 2
    her_q = hermite(coll_p, qq, vv)
    her_v = hermite(coll_p, vv, aa)
    her_q_d = her_q.derivative()
    her_v_d = her_v.derivative()
    her_q_d_d = her_q_d.derivative()
    return her_q, her_v, her_q_d, her_v_d, her_q_d_d


def try_array_f(function):
    def try_f(q, v, u, params):
        try:
            f_out = function(q, v, u, params)
        except:
            f_out = []
            for ii in range(q.shape[0]):
                _out = function(q[ii], v[ii], u[ii], params)
                _out = squeeze(_out)
                f_out.append(_out)
            f_out = array(f_out)
        return f_out

    return try_f


def interpolations_pseudospectral(
    qq,
    vv,
    uu,
    scheme,
    t0,
    t1,
    u_interp="pol",
    x_interp="pol",
    g_func=lambda q, v, u, p: u,
    params=None,
    n_interp=5000,
):
    """
    Generates arrays of equispaced points with values of interpolations.

    x(t) = [q(t), v(t)], and the physics equation states that x' = F(x, u),
    which is equivalent to [q', v'] = [v , G(q, v, u)]

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
    scheme : str
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
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
            q'' = g(q, q', u, params)
    params : list or None, default None
        Physical problem parameters to be passed to F
    n_interp : int, default 5000
        number of interpolation points

    Raises
    ------
    NameError
        When an unsupported value for scheme, x_interp or u_interp is used.

    Returns
    -------
    q_arr, q_arr_d, v_arr, v_arr_d, q_arr_d_d, u_arr : Numpy array, shape = (n_interp, N)
        equispaced values of interpolations.
    """
    from scipy.interpolate import CubicHermiteSpline as hermite
    from numpy import interp, gradient, zeros_like

    if scheme not in _implemented_schemes:
        NameError(f"Invalid scheme.\n valid options are {_implemented_schemes}")

    if params is None:
        params = []

    N = len(qq)
    tau_arr = linspace(-1, 1, n_interp)

    g_func = try_array_f(g_func)

    if u_interp == "pol":
        pol_u = get_pol_u(scheme, uu)
        u_arr = pol_u(tau_arr)
    elif u_interp == "lin":
        tau_u, uu = extend_u_array(uu, scheme, N)
        if len(uu.shape) == 1:
            u_arr = interp(tau_arr, tau_u, uu)
        elif len(uu.shape) == 2:
            u_arr = interp_2d(tau_arr, tau_u, uu)
        else:
            raise ValueError(
                f"U has {len(uu.shape)} dimensions, values accepted are 1 and 2"
            )
    elif u_interp == "smooth":
        tau_u, uu = extend_u_array(uu, scheme, N)
        uu_dot = gradient(uu, tau_u)
        u_arr = hermite(tau_u, uu, uu_dot)(tau_arr)
    else:
        raise NameError(
            'Invalid interpolation method for u.\n valid options are "pol", "lin", "smooth"'
        )

    if x_interp == "pol":
        tau_x = node_points(N, scheme)
        pol_q, pol_v, pol_q_d, pol_v_d, pol_q_d_d = get_pol_x(scheme, qq, vv, t0, t1)
        q_arr = pol_q(tau_arr)
        v_arr = pol_v(tau_arr)
        q_arr_d = pol_q_d(tau_arr)
        v_arr_d = pol_v_d(tau_arr)
        q_arr_d_d = pol_q_d_d(tau_arr)
    elif x_interp == "lin":
        tau_x, qq, vv = extend_x_arrays(qq, vv, scheme)
        if len(qq.shape) == 1:
            q_arr = interp(tau_arr, tau_x, qq)
            v_arr = interp(tau_arr, tau_x, vv)
        elif len(qq.shape) == 2:
            q_arr = interp_2d(tau_arr, tau_x, qq)
            v_arr = interp_2d(tau_arr, tau_x, vv)
        else:
            raise ValueError(
                f"q has {len(qq.shape)} dimensions, values accepted are 1 and 2"
            )

        coll_p = t0 + (1 + array(tau_x, dtype="float64")) * (t1 - t0) / 2
        t_arr_lin = linspace(t0, t1, n_interp)
        q_arr_d = find_der_polyline(t_arr_lin, coll_p, qq)
        v_arr_d = find_der_polyline(t_arr_lin, coll_p, vv)
        q_arr_d_d = zeros_like(q_arr)
    elif x_interp == "Hermite":
        tau_x, qq, vv = extend_x_arrays(qq, vv, scheme)
        aa = g_func(qq, vv, uu, params)
        her_q, her_v, her_q_d, her_v_d, her_q_d_d = get_hermite_x(
            qq, vv, aa, tau_x, t0, t1
        )
        t_arr_lin = linspace(t0, t1, n_interp)
        q_arr = her_q(t_arr_lin)
        v_arr = her_v(t_arr_lin)
        q_arr_d = her_q_d(t_arr_lin)
        v_arr_d = her_v_d(t_arr_lin)
        q_arr_d_d = her_q_d_d(t_arr_lin)
    else:
        raise NameError(
            'Invalid interpolation method for x.\n valid options are "pol", "lin", "Hermite"'
        )
    return q_arr, q_arr_d, v_arr, v_arr_d, q_arr_d_d, u_arr


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
    params=None,
    n_interp=5000,
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
    scheme : str
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
            'JG'
            'CG'
            'CG_inv'
            'CGR'
            'CGR_inv'
            'CGL'
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
            q'' = g(q, q', u, params)
    params : list or None, default None
        Physical problem parameters to be passed to F
    n_interp : int, default 5000
        number of interpolation points

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
        equispaced values of dynamic error q''(t) - G(q(t), q'(t), u(t)).

    """
    if params is None:
        params = []
    q_arr, q_arr_d, v_arr, v_arr_d, q_arr_d_d, u_arr = interpolations_pseudospectral(
        qq,
        vv,
        uu,
        scheme,
        t0,
        t1,
        u_interp,
        x_interp,
        g_func,
        params,
        n_interp,
    )
    g_func = try_array_f(g_func)
    err_q = q_arr_d - v_arr
    err_v = v_arr_d - g_func(q_arr, v_arr, u_arr, params)
    err_2 = q_arr_d_d - g_func(q_arr, q_arr_d, u_arr, params)

    return err_q, err_v, err_2
