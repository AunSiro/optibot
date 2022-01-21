# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:08:53 2022

@author: Siro Moreno
"""

from .schemes import interpolated_array, interpolated_array_derivative
from scipy.optimize import root, minimize
from numpy import (
    zeros,
    zeros_like,
    append,
    concatenate,
    linspace,
    expand_dims,
    interp,
    array,
    sum,
    abs,
    max,
    sqrt,
    trapz,
)
from numpy.linalg import inv
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
    if "parab" in scheme and u_scheme == "lin":
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


def dynamic_error_implicit(
    x_arr,
    u_arr,
    t_end,
    params,
    F,
    M,
    lambda_arr=None,
    X_dot=None,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params={},
    n_interp=2000,
):
    """
    Generate arrays of equispaced points with values of dynamic error.
    
    If x(t) = [q(t), v(t)], and the dynamics can be written as:
            | M    A_c|   | q''  |   |f_d (q, v, u)|
            |         | @ |      | = |             |
            |m_cd   0 |   |lambda|   |f_dc(q, v)   |,
    The first equation is:
        M @ q'' + A_c @ lambda = f_d (q, v, u)
    we define F(q, v, u, lambda) as F = f_d - A_c @ lambda
    we can define G(q, v, u, lambda) = M^-1 @ F(q, v, u, lambda)
    so that the 1st equation plus the definition of v(t) as q'(t)
    is equivalent to [q', v'] = [v , G(x, u)] 
    
    we can define the dynamic errors at a point t as:
        dyn_q_err = q'(t) - v(t)
        dyn_v_err = v'(t) - G(x(t), u(t))
        dyn_2_err_a = q''(t) - G(x(t), u(t))
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
    F : Function of (x, u, lambda, params)
        A function of a dynamic sistem, so that
            G(q, v, u, lambda) = M^-1 @ F(x, u, lambda)
    M : Function of (x, params)
        Calculates the numerical value of the mass matrix at a given configuration
    lambda_arr : Numpy Array
        If the problem has restrictions, they are taken into account on the 
        lagrangian through the term: A_c @ lambda
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
    N = x_arr.shape[0]
    dim = x_arr.shape[1] // 2
    h = t_end / (N - 1)

    def G(x, u, lambdas, params):
        m_inv = inv(M(x, params))
        return (m_inv @ F(x, u, lambdas, params)).T

    if X_dot is None:
        X_dot = zeros_like(x_arr)
        for ii in range(N):
            if lambda_arr is None:
                lambdas = None
            else:
                lambdas = lambda_arr[ii]
            X_dot[ii, :dim] = x_arr[ii, dim:]
            X_dot[ii, dim:] = G(x_arr[ii], u_arr[ii], lambdas, params)

    t_interp = linspace(0, t_end, n_interp)
    x_interp, u_interp = interpolated_array(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        X_dot=X_dot,
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
        X_dot=X_dot,
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
        X_dot=X_dot,
        scheme=scheme,
        order=2,
        scheme_params=scheme_params,
    )
    if lambda_arr is None:
        lambda_interp = None
    else:
        lambda_interp = interp(linspace(0, t_end, n_interp), t_interp, lambda_arr)

    f_arr_a = zeros([n_interp, dim])
    f_arr_b = zeros([n_interp, dim])
    for ii in range(n_interp):
        if lambda_interp is None:
            lambdas = None
        else:
            lambdas = lambda_interp[ii]
        f_arr_a[ii, :] = G(x_interp[ii], u_interp[ii], lambdas, params)
        x_q = x_interp[ii].copy()
        x_q[dim:] = x_dot_interp[ii, :dim]
        f_arr_b[ii, :] = G(x_q, u_interp[ii], lambdas, params)
    dyn_err_q = x_dot_interp[:, :dim] - x_interp[:, dim:]
    dyn_err_v = x_dot_interp[:, dim:] - f_arr_a
    dyn_err_2_a = x_dot_dot_interp[:, :dim] - f_arr_a
    dyn_err_2_b = x_dot_dot_interp[:, :dim] - f_arr_b
    return dyn_err_q, dyn_err_v, dyn_err_2_a, dyn_err_2_b


def arr_mod(x):
    x_1 = sum(x * x, axis=1)
    return sqrt(x_1)


def arr_sum(x):
    return sum(abs(x), axis=1)


def arr_max(x):
    return max(abs(x), axis=1)


def arr_abs_integr_vert(t_arr, x):
    errors = trapz(abs(x), t_arr, axis=0)
    return errors
