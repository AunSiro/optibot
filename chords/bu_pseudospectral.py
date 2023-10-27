#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:35:30 2023

@author: Siro Moreno

Here we define functions needed to operate with bottom-up pseudospectral
collocations schemes. In order to keep the best accuracy in interpolations, 
barycentric formulas are constructed.
"""

from .pseudospectral import LG, LGL, LGR, JG, JGR, JGR_inv, JGL, bary_poly, bary_poly_2d
from .util import gauss_rep_integral, poly_integral, poly_integral_2d
from functools import lru_cache
from numpy import zeros, array, concatenate
from sympy import jacobi_poly
from math import factorial


_implemented_schemes = ["LG", "LGL", "LGR", "LGR_inv", "JG", "JGR", "JGR_inv", "JGL"]


@lru_cache(maxsize=2000)
def BU_coll_points(N, scheme, order=2, precission=20):
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
            'JGR'
            'JGR_inv'
            'JGL'
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
    elif scheme == "JG":
        return JG(N, order, precission)
    elif scheme == "JGR":
        return JGR(N, order, precission)
    elif scheme == "JGR_inv":
        return JGR_inv(N, order, precission)
    elif scheme == "JGL":
        return JGL(N, order, precission)
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


@lru_cache(maxsize=2000)
def BU_construction_points(N, scheme, order=2, precission=20):
    """
    Generates a list of with values of tau for construction points for schemes
    with N collocation points. Construction points are the collocation points,
    excluding the initial point if it is a collocation point, and
    adding the final point if it is not a collocation point.

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
            'JGR'
            'JGR_inv'
            'JGL'
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    constr_points : list
        list of construction points
    """
    if scheme == "LG":
        return LG(N, precission) + [1.0]
    elif scheme == "LGR":
        return LGR(N, precission)[1:] + [1.0]
    elif scheme == "LGR_inv":
        return [-ii for ii in LGR(N, precission)[::-1]]
    elif scheme in ["LGL", "D2"]:
        return LGL(N, precission)[1:]
    elif scheme == "JG":
        return JG(N, order, precission) + [1.0]
    elif scheme == "JGR":
        return JGR(N, order, precission)[1:] + [1.0]
    elif scheme == "JGR_inv":
        return JGR_inv(N, order, precission)
    elif scheme == "JGL":
        return JGL(N, order, precission)[1:]
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


def get_coll_indices(scheme):
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
    if scheme in ["LG", "JG"]:
        coll_index = slice(1, -1)
    elif scheme in ["LGR", "JGR"]:
        coll_index = slice(None, -1)
    elif scheme in ["LGR_inv", "JG_inv"]:
        coll_index = slice(1, None)
    elif scheme in ["LGL", "JGL"]:
        coll_index = slice(None, None)
    else:
        raise NotImplementedError(f"Scheme {scheme} not implemented yet")
    return coll_index


@lru_cache(maxsize=2000)
def BU_unit_Lag_pol(N, scheme, n, order=2, precission=20):
    """
    Generate a barycentric numeric Lagrange polynomial over N node points.
    L_n(x_i) = 0 for i != n
    L_n(x_n) = 1

    Parameters
    ----------
    N : Int
        Number of Collocation points.
    scheme : str
        Name of pseudospectral scheme.
    n : int
        number for which L(x_n) = 1.
    order: int, default 1
        When Jacobi type schemes are used, the order of derivation for
        which they are optimized
    precission : int, optional
        Precission in colloction point calculation. The default is 20.

    Returns
    -------
    Function
        Barycentric Lagrange Polynomial L_n(x) for x in [-1,1].

    """
    x = BU_coll_points(N, scheme, order, precission)

    y = zeros(N)
    y[n] = 1
    return bary_poly(x, y)


def tau_to_t_points(points, t0, tf):
    """
    converts a point or series of points from tau in [-1,1] to t in [t0, tf]

    Parameters
    ----------
    points : number, list, array or tuple
        points in tau
    t0 : float
        initial t
    tf : float
        final t

    Returns
    -------
    new_points : number, list, array or tuple
        points in t

    """
    points_arr = array(points)
    h = tf - t0
    new_points = t0 + h * (points_arr + 1) / 2
    if type(points) == list:
        new_points = list(new_points)
    elif type(points) == tuple:
        new_points = tuple(new_points)
    elif type(points) == float:
        new_points = float(new_points)
    return new_points


def tau_to_t_function(f, t0, tf):
    """
    converts a function from f(tau), tau in [-1,1] to f(t), t in [t0, tf]

    Parameters
    ----------
    f : function
        function of tau: f(tau)
    t0 : float
        initial t
    tf : float
        final t

    Returns
    -------
    new_f : function
        function of t: f(t)

    """
    h = tf - t0

    def new_F(t):
        tau = 2 * (t - t0) / h - 1
        return f(tau)

    try:
        old_docstring = str(f.__doc__)
    except:
        old_docstring = "function of tau"
    try:
        old_f_name = str(f.__name__)
    except:
        old_f_name = "Unnamed Function"

    new_docstring = f"""
    This is a time t based version of function {old_f_name}.
    This expanded function is designed to operate with time t: F(t)
    While the old function was designed for tau: F(tau)
    Old function documentation:
    """
    new_docstring += old_docstring
    new_F.__doc__ = new_docstring
    new_F.__name__ = old_f_name + " of tau"
    return new_F


@lru_cache(maxsize=2000)
def BU_unit_Lag_pol_t(N, scheme, n, t0, tf, order=2, precission=20):
    """
    Generate a barycentric numeric Lagrange polynomial over N node points.
    This will create a function for use with t in [t0, tf].
    L_n(x_i) = 0 for i != n
    L_n(x_n) = 1

    Parameters
    ----------
    N : Int
        Number of Collocation points.
    scheme : str
        Name of pseudospectral scheme.
    n : int
        number for which L(x_n) = 1.
    order: int, default 1
        When Jacobi type schemes are used, the order of derivation for
        which they are optimized
    precission : int, optional
        Precission in colloction point calculation. The default is 20.

    Returns
    -------
    Function
        Barycentric Lagrange Polynomial L_n(t) for t in [t0, tf].

    """
    L_tau = BU_unit_Lag_pol(N, scheme, n, order, precission)

    return tau_to_t_function(L_tau, t0, tf)


@lru_cache(maxsize=2000)
def _Lag_integ(
    N_coll,
    n_lag_pol,
    scheme,
    deriv_order,
    t_constr_index,
    scheme_order=2,
    precission=20,
):
    """
    Calculate a definite integral of a Lagrange polynomial to use in the
    definition of an integration matrix.

    Parameters
    ----------
    N_coll : int
        Number of collocation points
    n_lag_pol : int between 0 and N_coll-1
        current lagrange polynomial will be 1 for tau_{n_lag_pol}, and 0 for
        all the other collocation points tau_i for i = 0 ... N_coll-1
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
            'JGR'
            'JGR_inv'
            'JGL'
    deriv_order : int
        derivation level for which the element in the matrix is being
        calculated. For example:
            for q, deriv_order = 0
            for v, deriv_order = 1
    t_constr_index : int
        index of the construction point for which the element is being calculated
    scheme_order : int, optional
        Differential order of the problem. The default is 2.
    precission : int, optional
        Precission in colloction point calculation. The default is 20.

    Returns
    -------
    integral : float
        result of the numerical integration.

    """
    constr_points = BU_construction_points(N_coll, scheme, scheme_order, precission)
    tf = constr_points[t_constr_index]
    lag_pol = BU_unit_Lag_pol(N_coll, scheme, n_lag_pol, scheme_order, precission)
    integration_order = scheme_order - deriv_order
    integral = gauss_rep_integral(lag_pol, -1, tf, N_coll - 1, integration_order)
    return integral


@lru_cache(maxsize=2000)
def Integration_Matrix(N_coll, scheme, deriv_order, h, scheme_order=2, precission=20):
    assert (
        deriv_order < scheme_order
    ), "derivation order must be smaller than differential order of the problem"
    assert deriv_order >= 0
    assert scheme_order >= 1
    constr_points = BU_construction_points(N_coll, scheme, scheme_order, precission)
    n_t = len(constr_points)
    M = scheme_order
    N = N_coll
    P = deriv_order
    matrix = zeros([n_t, M + N])
    for t_index in range(n_t):
        t = tau_to_t_points(constr_points[t_index], 0, h)
        for ii in range(M - P):
            matrix[t_index, P + ii] = t**ii / factorial(ii)
        for ii in range(N):
            matrix[t_index, M + ii] = (h / 2) ** (M - P) * _Lag_integ(
                N_coll, ii, scheme, P, t_index, scheme_order, precission
            )
    return matrix


def Polynomial_interpolations_BU(xx_dot, x_0, uu, scheme, scheme_order, t0, tf, N_coll):
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    n_q = len(x_0) // scheme_order
    highest_der = xx_dot[:, -n_q:]
    coll_index = get_coll_indices(scheme)
    highest_der_col = highest_der[coll_index, :]
    n_col = highest_der_col.shape[0]

    q_and_ders = []
    for _ii in range(scheme_order):
        q_and_ders.append(x_0[n_q * _ii : n_q * (_ii + 1)])
    # q_and_ders = array(q_and_ders, dtype = float)

    # polynomial_data = concatenate((q_and_ders, highest_der_col), axis= 0)

    coll_points = tau_to_t_points(BU_coll_points(n_col, scheme, scheme_order), t0, tf)

    u_poly = bary_poly_2d(coll_points, uu)
    highest_der_poly = bary_poly_2d(coll_points, highest_der_col)

    q_and_der_polys = [
        highest_der_poly,
    ]
    for ii in range(scheme_order):
        _prev_poly = q_and_der_polys[-1]
        _new_poly = poly_integral_2d(
            _prev_poly, n_col - 1 + ii, t0, tf, q_and_ders[scheme_order - ii - 1]
        )
        q_and_der_polys.append(_new_poly)

    return u_poly, q_and_der_polys[::-1]


def interpolations_BU_pseudospectral(
    xx_dot,
    x_0,
    uu,
    scheme,
    t0,
    tf,
    u_interp="pol",
    x_interp="pol",
    g_func=lambda q, v, u, p: u,
    params=None,
    n_interp=5000,
):
    pass
