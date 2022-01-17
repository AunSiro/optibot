# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:08:53 2022

@author: Siro Moreno
"""

from.schemes import interpolated_array, interpolated_array_derivative
from scipy.optimize import root, minimize
from numpy import (
    zeros,
    append,
    concatenate,
    linspace,
    expand_dims,
    interp,
    array,
    sum,
    abs,
)
from scipy.interpolate import CubicHermiteSpline as hermite
from copy import copy
import warnings

def dynamic_error(
    x_arr,
    u_arr,
    t_end,
    params,
    F,
    X_dot=None,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params={},
    n_interp=2000,
):
    """
    Generate arrays of equispaced points with values of dynamic error.
    
    If x(t) = [q(t), v(t)], and the physics equation states that x' = F(x, u),
    which is equivalent to [q', v'] = [v , G(q, v, u)] we can define the 
    dynamic errors at a point t as:
        dyn_q_err = q'(t) - v(t)
        dyn_v_err = v'(t) - G(q(t), v(t), u(t))
        dyn_2_err_a = q''(t) - G(q(t), v(t), u(t))
        dyn_2_err_b = q''(t) - G(q(t), q'(t), u(t))
        
    'scheme' and 'u_scheme' define the way in which we interpolate the values
    of q, v and u between the given points.
        
    It is assumed that X and U start at t = 0 and are equispaced in time
    in the interval (0, t_end).
    

    Parameters
    ----------
    x_arr : Numpy Array, shape = (W, 2N)
        Values known of x(t)
    u_arr : Numpy Array, shape = (W, [Y])
        Values known of u(t)
    t_end : float
        ending time of interval of analysis
    params : list
        Physical problem parameters to be passed to F
    F : Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)
        if X_dot is None and F is not, F will be used to calculate X'
    X_dot : Numpy Array, optional, shape = (W, 2N), default = None
        Known values of X'
        if X_dot is None, F will be used to calculate X'
    scheme : str, optional
        Scheme to be used in the X interpolation. The default is "hs_scipy".
        Acceptable values are:
            "trapz" : trapezoidal scheme compatible interpolation (not lineal!)
            "trapz_mod": modified trapezoidal scheme compatible interpolation (not lineal!)
            "hs_scipy": 3d order polynomial that satisfies continuity in x(t) and x'(t)
            "hs": Hermite-Simpson scheme compatible interpolation
            "hs_mod": modified Hermite-Simpson scheme compatible interpolation
            "hs_parab": Hermite-Simpson scheme compatible interpolation with parabolic U
            "hs_mod_parab": modified Hermite-Simpson scheme compatible interpolation with parabolic U
    u_scheme : string, optional
        Model of the interpolation that must be used. The default is "lin".
        Acceptable values are:
            "lin": lineal interpolation
            "parab": parabolic interpolation, requires central points array
            as scheme params[0]
    scheme_params :dict, optional
        Aditional parameters of the scheme. The default is {}.
    n_interp : int, optional
        Number of interpolation points. The default is 2000.

    Returns
    -------
    dyn_err_q : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error q'(t) - v(t).
    dyn_err_v : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error v'(t) - G(q(t), v(t), u(t)).
    dyn_err_2_a : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error q''(t) - G(q(t), v(t), u(t)).
    dyn_err_2_b : Numpy array, shape = (n_interp, N)
        equispaced values of dynamic error q''(t) - G(q(t), q'(t), u(t)).

    """
    if "parab" in scheme and u_scheme != "parab":
        warnings.warn(
            "You are currently using a u-parabolic interpolation for x with a lineal interpolation of u"
        )
    if "parab" in u_scheme and "parab" not in scheme:
        warnings.warn(
            "You are currently using a parabolic interpolation for u with a non u-parabolic interpolation of x"
        )
    N = x_arr.shape[0] - 1
    dim = x_arr.shape[1] // 2
    h = t_end / N
    t_interp = linspace(0, t_end, n_interp)
    x_interp, u_interp = interpolated_array(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        F=F,
        scheme=scheme,
        u_scheme=u_scheme,
        scheme_params=scheme_params,
    )
    x_dot_interp = interpolated_array_derivative(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        F=F,
        scheme=scheme,
        order=1,
        scheme_params=scheme_params,
    )
    x_dot_dot_interp = interpolated_array_derivative(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        F=F,
        scheme=scheme,
        order=2,
        scheme_params=scheme_params,
    )
    f_arr_a = zeros([n_interp, dim])
    f_arr_b = zeros([n_interp, dim])
    for ii in range(n_interp):
        f_arr_a[ii, :] = F(x_interp[ii], u_interp[ii], params)[dim:]
        x_q = x_interp[ii].copy()
        x_q[dim:] = x_dot_interp[ii, :dim]
        f_arr_b[ii, :] = F(x_q, u_interp[ii], params)[dim:]
    dyn_err_q = x_dot_interp[:, :dim] - x_interp[:, dim:]
    dyn_err_v = x_dot_interp[:, dim:] - f_arr_a
    dyn_err_2_a = x_dot_dot_interp[:, :dim] - f_arr_a
    dyn_err_2_b = x_dot_dot_interp[:, :dim] - f_arr_b
    return dyn_err_q, dyn_err_v, dyn_err_2_a, dyn_err_2_b
