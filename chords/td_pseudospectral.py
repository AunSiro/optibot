#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:04:21 2024

@author: Siro Moreno

Here we define functions needed to operate with top-down pseudospectral
collocations schemes. In order to keep the best accuracy in interpolations, 
barycentric formulas are constructed.


"""

from .util import Lag_pol_2d
from .numpy import store_results
from numpy import array
from .bu_pseudospectral import (
    BU_coll_points,
    _gauss_like_schemes,
    _radau_like_schemes,
    _radau_inv_schemes,
    _lobato_like_schemes,
    _other_schemes,
    _implemented_schemes,
)
from functools import lru_cache
from .pseudospectral import matrix_D_bary


@lru_cache(maxsize=2000)
def TD_construction_points(N, scheme, order=2, precission=16):
    """
    Generates a list of with values of tau for construction points for schemes
    with N collocation points. Construction points are the collocation points,
    adding the initial point if it is not a collocation point, and
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
            'CG'
            'CGR'
            'CGR_inv'
            'CGL'
    order: int
        differential order of the problem for jacobi points, default 2
    precission: int, default 20
        number of decimal places of precission

    Returns
    -------
    constr_points : list
        list of construction points
    """
    coll_p = BU_coll_points(N, scheme, order, precission)
    if scheme in _gauss_like_schemes:
        return [-1 - 0] + coll_p + [1.0]
    elif scheme in _radau_like_schemes:
        return coll_p[1:] + [1.0]
    elif scheme in _radau_inv_schemes:
        return [-1.0] + coll_p
    elif scheme in _lobato_like_schemes:
        return coll_p
    elif scheme in _other_schemes:
        if scheme in []:
            pass
        else:
            raise NotImplementedError(
                f"scheme {scheme} in category 'others' is not yet implemented"
            )
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


@lru_cache(maxsize=2000)
@store_results
def matrix_D_nu(size):
    return matrix_D_bary(size, "CGL")


@lru_cache(maxsize=2000)
@store_results
def matrix_L(N, scheme, order=2, precission=16):
    constr_points = TD_construction_points(N, scheme, order, precission)
    constr_points = array(constr_points, dtype="float64")
    l = N + order
    polynomial = Lag_pol_2d(l, "CGL")
    return polynomial(constr_points)
