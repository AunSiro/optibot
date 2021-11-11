#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 12:11:43 2021

@author: Siro Moreno
"""

from sympy import legendre_poly, symbols, expand, zeros, lambdify
from functools import lru_cache
from numpy import array, piecewise

# --- Generating Collocation Points
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
