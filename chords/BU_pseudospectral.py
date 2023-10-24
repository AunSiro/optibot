#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:35:30 2023

@author: Siro Moreno

Here we define functions needed to operate with bottom-up pseudospectral
collocations schemes. In order to keep the best accuracy in interpolations, 
barycentric formulas are constructed.
"""

from .pseudospectral import LG, LGL, LGR, JG, bary_poly
from .util import gauss_rep_integral
from functools import lru_cache
from numpy import zeros
from sympy import jacobi_poly


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
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


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
        Barycentric Lagrange Polynomial L_n(x).

    """
    x = BU_coll_points(N, scheme, order, precission)

    y = zeros(N)
    y[n] = 1
    return bary_poly(x, y)
