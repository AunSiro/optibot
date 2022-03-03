#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 18:03:53 2022

@author: smorenom
"""

from casadi import Opti

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


class Opti_Problem:
    def __init__(
        self, scheme="trapz", ini_guess="zero", solve_repetitions=1, t_start=0, t_end=1
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
        self.scheme = scheme
        self.ini_guess = ini_guess
        self.solve_repetitions = solve_repetitions
        self.t_start = t_start
        self.t_end = t_end
