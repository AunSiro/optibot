#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:52:34 2021

@author: Siro Moreno
"""

from scipy.optimize import root
from numpy import zeros, append, concatenate, linspace, expand_dims, interp, array
from scipy.interpolate import CubicHermiteSpline as hermite
from copy import copy
import functools


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def is2d(x):
    try:
        shape = x.shape
        if len(shape) == 2:
            return True
        else:
            return False
    except AttributeError:
        return False


def vec_len(x):
    if type(x) == int or type(x) == float:
        return 1
    try:
        return len(x)
    except TypeError:
        if x.size == 1:
            return 1
        else:
            if x.shape[0] == 1 and len(x.shape) == 2:
                return x.shape[1]
            else:
                return x.shape[0]


def interp_2d(t_array, old_t_array, Y):
    new_Y_len = t_array.shape[0]
    new_Y_width = Y.shape[-1]
    new_Y = zeros([new_Y_len, new_Y_width])
    for ii in range(new_Y_width):
        new_Y[:, ii] = interp(t_array, old_t_array, Y[:, ii])
    return new_Y


def extend_array(x):
    apppendix = expand_dims(x[-1], axis=0)
    return append(x, apppendix, 0)


def expand_F(F, mode="numpy"):
    """
    Expands a function F(x,u,params) that returns accelerations,
    so that the new function return accelerations and velocities.

    Parameters
    ----------
    F : function of (x, u, params)
        A function of a dynamic sistem, so that
            v' = F(x, u, params),
            q' = v
    mode : str: 'numpy' o 'casadi', optional
        Wether the function is a numpy or a casadi function.
        The default is "numpy".

    Returns
    -------
    Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)

    """
    old_docstring = str(F.__doc__)
    old_f_name = str(F.__name__)
    if not mode in ["numpy", "casadi"]:
        raise NameError(f"Unrecognized mode: {mode}")
    if mode == "numpy":

        def new_F(x, u, params):
            x = array(x)
            u = array(u)
            axnum = len(x.shape) - 1
            if axnum >= 2:
                raise ValueError(
                    f"Array X must have dimension 1 or 2, but has {len(x.shape)} instead"
                )
            if len(u.shape) >= 3:
                raise ValueError(
                    f"Array U must have dimension 1 or 2, but has {len(u.shape)} instead"
                )
            a = F(x, u, params)
            dim = x.shape[-1] // 2
            if axnum == 1:
                v = x[:, dim:]
            else:
                v = x[dim:]
            if is_iterable(a):
                new_a = array(a)
            else:
                new_a = array([a,])
            if len(new_a.shape) != len(v.shape):
                if new_a.shape[0] == v.shape[0]:
                    new_a = expand_dims(new_a, axis=1)
                elif new_a.shape[-1] == v.shape[-1]:
                    new_a = expand_dims(new_a, axis=0)
            assert new_a.shape == v.shape
            res = concatenate((v, new_a), axnum)
            return res

    elif mode == "casadi":
        from casadi import horzcat, DM

        def new_F(x, u, params):
            a = F(x, u, params)
            x = horzcat(x)
            x_transposed = False
            if x.shape[-1] == 1:
                # As x must always contain at least one q and one v,
                # it can never be a vertical vector of width 1,
                # considering time as the vertical dimension.
                x = x.T
                x_transposed = True
            if x.shape[0] == 1 and DM(a).shape[0] != 1:
                a = a.T
            dim = x.shape[-1] // 2
            v = array(x)[:, dim:]
            res = horzcat(v, a)
            if x_transposed and res.shape[0] == 1:
                res = res.T
                # If the input was x as a vertical array of
                # width 1, we must return a result
                # dimensionally consistend.
            return res

    new_docstring = f"""
    This is an expanded version of function {old_f_name}.
    This expanded function is designed to describe a dinamic sistem so that:
        x' = F(x, u, params)
    While the old function was:
        v' = F(x, u, params),
        q' = v
    Old function documentation:
    """
    new_docstring += old_docstring
    new_F.__doc__ = new_docstring

    return new_F


# --- Integration Steps ---


def euler_step(x, u, F, dt, params):
    return x + dt * F(x, u, params)


def rk4_step(x, u, F, dt, params):
    k1 = F(x, u, params)
    k2 = F(x + dt / 2 * k1, u, params)
    k3 = F(x + dt / 2 * k2, u, params)
    k4 = F(x + dt * k3, u, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def trapz_opti_step(x_n, x, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    res = x + dt / 2 * (f + f_n) - x_n
    return res


def trapz_step(x, u, u_n, F, dt, params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(trapz_opti_step, x_0, (x, u, u_n, F, dt, params))
    return x_n.x


def trapz_mod_opti_step(x_n, x, u, u_n, F, dt, params):
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    res = copy(x)
    res[dim:] = x[dim:] + dt / 2 * (f + f_n) - x_n[dim:]
    res[:dim] = x[:dim] + dt * x[dim:] + dt ** 2 / 6 * (f_n + 2 * f) - x_n[:dim]
    return res


def trapz_mod_step(x, u, u_n, F, dt, params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(trapz_mod_opti_step, x_0, (x, u, u_n, F, dt, params))
    return x_n.x


def hs_opti_step(x_n, x, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    u_c = (u + u_n) / 2
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    f_c = F(x_c, u_c, params)
    res = x + dt / 6 * (f + 4 * f_c + f_n) - x_n
    return res


def hs_step(x, u, u_n, F, dt, params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_opti_step, x_0, (x, u, u_n, F, dt, params))
    return x_n.x


def hs_mod_opti_step(x_n, x, u, u_n, F, dt, params):
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    u_c = (u + u_n) / 2
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt ** 2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 6 * (f + 4 * f_c + f_n) - v_n
    res[:dim] = q + dt * v + dt ** 2 / 6 * (f + 2 * f_c) - q_n
    return res


def hs_mod_step(x, u, u_n, F, dt, params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_mod_opti_step, x_0, (x, u, u_n, F, dt, params))
    return x_n.x


def hs_parab_opti_step(x_n, x, u, u_n, F, dt, params, scheme_params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    u_c = scheme_params
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    f_c = F(x_c, u_c, params)
    res = x + dt / 6 * (f + 4 * f_c + f_n) - x_n
    return res


def hs_parab_step(x, u, u_n, F, dt, params, scheme_params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_opti_step, x_0, (x, u, u_n, F, dt, params, scheme_params))
    return x_n.x


def hs_mod_parab_opti_step(x_n, x, u, u_n, F, dt, params, scheme_params):
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    u_c = scheme_params
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt ** 2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 6 * (f + 4 * f_c + f_n) - v_n
    res[:dim] = q + dt * v + dt ** 2 / 6 * (f + 2 * f_c) - q_n
    return res


def hs_mod_parab_step(x, u, u_n, F, dt, params, scheme_params):
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_mod_opti_step, x_0, (x, u, u_n, F, dt, params, scheme_params))
    return x_n.x


# --- Integrations ---
# These functions are expected to work with numpy arrays, and will
# convert other formats for X and U into them


def coherent_dimensions(func):
    """
    Adapts input variables to ensure that they are compatible
    with functions of structure integrate_x(x_0, u, F, dt, params)

    Parameters
    ----------
    func : Function
        Integration function whose structure is F(x_0, u, F, dt, params).
    -------
    Function
        The same function, but with additional comprobations
        that the input variables are coherent.

    """

    @functools.wraps(func)
    def wrapper_decorator(x_0, u, F, dt, params):
        x_0 = array(x_0, dtype=float)
        u = array(u, dtype=float)

        # If u was a number, it will produce errors later
        # while trying to iterate over it. We have to to convert it
        # into a 1D array of lenght 1
        if u.size == 1 and u.shape == ():
            u = expand_dims(u, axis=0)

        # x_0 is the initial state and must be 1D:
        if not (len(x_0.shape) == 1 or (len(x_0.shape) == 2 and x_0.shape[0] == 1)):
            raise ValueError(
                f"x_0 must be a 1D array, but instead its shape is {x_0.shape}"
            )
        # If x_0 is a 2D array of one line, we have to convert it
        # to a normal 1D array so that the integration is coherent.
        if len(x_0.shape) == 2:
            x_0 = x_0[0]
        # If u is 1D but the problem has more than 1 q,
        # If u is 1D but the problem has more than 1 q,
        # it can mean that it corresponds to only one step
        if len(u.shape) == 1 and x_0.shape[0] != 2:
            try:
                F(x_0, u, params)
            except TypeError:
                pass
            else:
                u = expand_dims(u, axis=0)
        value = func(x_0, u, F, dt, params)
        return value

    return wrapper_decorator


@coherent_dimensions
def integrate_euler(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(vec_len(u)):
        x_i = euler_step(x[-1], u[ii], F, dt, params)
        x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_rk4(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(vec_len(u)):
        x_i = rk4_step(x[-1], u[ii], F, dt, params)
        x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_trapz(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = trapz_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = trapz_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_trapz_mod(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = trapz_mod_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = trapz_mod_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = hs_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_mod(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_mod_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = hs_mod_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_parab(x_0, u, F, dt, params, scheme_params):
    x = [
        x_0,
    ]
    u_c = scheme_params[0]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_parab_step(x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii])
        x.append(x_i)
    x_i = hs_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_mod_parab(x_0, u, F, dt, params, scheme_params):
    x = [
        x_0,
    ]
    u_c = scheme_params[0]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_mod_parab_step(x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii])
        x.append(x_i)
    x_i = hs_mod_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return array(x)


# --- Schemes as Restrictions ---


def euler_restr(x, x_n, u, u_n, F, dt, params):
    return x_n - (x + dt * F(x, u, params))


def rk4_restr(x, x_n, u, u_n, F, dt, params):
    k1 = F(x, u, params)
    k2 = F(x + dt / 2 * k1, u, params)
    k3 = F(x + dt / 2 * k2, u, params)
    k4 = F(x + dt * k3, u, params)
    return x_n - (x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def trapz_restr(x, x_n, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    return x_n - (x + dt / 2 * (f + f_n))


def trapz_mod_restr(x, x_n, u, u_n, F, dt, params):
    dim = vec_len(x) // 2
    res = copy(x)
    if is2d(x):
        first_ind = slice(None, None), slice(None, dim)
        last_ind = slice(None, None), slice(dim, None)
    else:
        first_ind = slice(None, dim)
        last_ind = slice(dim, None)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    res[last_ind] = x[last_ind] + dt / 2 * (f + f_n)
    res[first_ind] = x[first_ind] + dt * x[last_ind] + dt ** 2 / 6 * (f_n + 2 * f)
    return x_n - res


def hs_restr(x, x_n, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    u_c = (u + u_n) / 2
    f_c = F(x_c, u_c, params)
    return x + dt / 6 * (f + 4 * f_c + f_n) - x_n


def hs_mod_restr(x, x_n, u, u_n, F, dt, params):
    dim = vec_len(x) // 2
    x_c = copy(x)
    res = copy(x)
    if is2d(x):
        first_ind = slice(None, None), slice(None, dim)
        last_ind = slice(None, None), slice(dim, None)
    else:
        first_ind = slice(None, dim)
        last_ind = slice(dim, None)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = (u + u_n) / 2
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt ** 2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 6 * (f + 4 * f_c + f_n)
    res[first_ind] = q + dt * v + dt ** 2 / 6 * (f + 2 * f_c)
    return x_n - res


def hs_parab_restr(x, x_n, u, u_n, F, dt, params, scheme_params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    u_c = scheme_params[0]
    f_c = F(x_c, u_c, params)
    return x + dt / 6 * (f + 4 * f_c + f_n) - x_n


def hs_mod_parab_restr(x, x_n, u, u_n, F, dt, params, scheme_params):
    from optibot.schemes import vec_len, is2d
    from copy import copy

    dim = vec_len(x) // 2
    x_c = copy(x)
    res = copy(x)
    if is2d(x):
        first_ind = slice(None, None), slice(None, dim)
        last_ind = slice(None, None), slice(dim, None)
    else:
        first_ind = slice(None, dim)
        last_ind = slice(dim, None)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = scheme_params[0]
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt ** 2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 6 * (f + 4 * f_c + f_n)
    res[first_ind] = q + dt * v + dt ** 2 / 6 * (f + 2 * f_c)
    return x_n - res


# --- Interpolations ---


def interp_parab(tau, h, y_0, y_c, y_n):
    xi = tau / h
    return y_0 + xi * (-3 * y_0 + 4 * y_c - y_n) + 2 * xi ** 2 * (y_0 - 2 * y_c + y_n)


def trap_mod_interp(x, x_n, u, u_n, tau, F, h, params):
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q_interp = q + v * tau + 1 / 2 * f * tau ** 2 + 1 / (6 * h) * tau ** 3 * (f_n - f)
    v_interp = v + tau * f + tau ** 2 / (2 * h) * (f_n - f)
    return concatenate([q_interp, v_interp])


def trap_interp(x, x_n, u, u_n, tau, F, h, params):
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q_interp = q + v * tau + 1 / (2 * h) * tau ** 2 * (v_n - v)
    v_interp = v + f * tau + 1 / (2 * h) * tau ** 2 * (f_n - f)
    return concatenate([q_interp, v_interp])


def hs_midpoint(x, x_n, u, u_n, tau, F, h, params):
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    v_c = (v + v_n) / 2 + h / 8 * (f - f_n)
    q_c = (q + q_n) / 2 + h / 8 * (v - v_n)
    return concatenate([q_c, v_c])


def hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params):
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    v_c = (v + v_n) / 2 + h / 8 * (f - f_n)
    q_c = (13 * q + 3 * q_n + 5 * v * h) / 16 + h ** 2 / 96 * (4 * f - f_n)
    return concatenate([q_c, v_c])


def hs_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        q
        + v * tau
        + 1 / 2 * (-3 * v + 4 * v_c - v_n) * tau ** 2 / h
        + 1 / 3 * (2 * v - 4 * v_c + 2 * v_n) * tau ** 3 / (h ** 2)
    )
    v_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    return concatenate([q_interp, v_interp])


def hs_mod_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        q
        + v * tau
        + 1 / 2 * f * tau ** 2
        + 1 / 6 * (-3 * f + 4 * f_c - f_n) * tau ** 3 / h
        + 1 / 12 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 4 / (h ** 2)
    )
    v_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    return concatenate([q_interp, v_interp])


def hs_parab_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        q
        + v * tau
        + 1 / 2 * (-3 * v + 4 * v_c - v_n) * tau ** 2 / h
        + 1 / 3 * (2 * v - 4 * v_c + 2 * v_n) * tau ** 3 / (h ** 2)
    )
    v_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    return concatenate([q_interp, v_interp])


def hs_mod_parab_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        q
        + v * tau
        + 1 / 2 * f * tau ** 2
        + 1 / 6 * (-3 * f + 4 * f_c - f_n) * tau ** 3 / h
        + 1 / 12 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 4 / (h ** 2)
    )
    v_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    return concatenate([q_interp, v_interp])


def _newpoint_u(U, h, t, u_scheme, scheme_params=0):
    n = int(t // h)
    tau = t % h
    if (n + 1) > U.shape[0]:
        raise ValueError(f"Value of time {t} detected outside interpolation limits")
    # print(f't = {t} , tau = {tau} , n = {n} , h = {h}')
    if abs(tau) < h * 1e-8:
        u_interp = U[n]
    elif abs(tau - h) < h * 1e-8:
        u_interp = U[n + 1]
    else:
        u, u_n = U[n], U[n + 1]
        if u_scheme == "parab":
            U_c = scheme_params[0]
            u_c = U_c[n]
            u_interp = interp_parab(tau, h, u, u_c, u_n)
        else:
            raise NameError(f"scheme {u_scheme} not recognized")
    return u_interp


def _newpoint(X, U, F, h, t, params, scheme, scheme_params=0):
    n = int(t // h)
    tau = t % h
    if (n + 1) > X.shape[0]:
        raise ValueError(f"Value of time {t} detected outside interpolation limits")
    # print(f't = {t} , tau = {tau} , n = {n} , h = {h}')
    if abs(tau) < h * 1e-8:
        x_interp = X[n]
    elif abs(tau - h) < h * 1e-8:
        x_interp = X[n + 1]
    else:
        x, x_n, u, u_n = X[n], X[n + 1], U[n], U[n + 1]
        if scheme == "trapz_mod":
            x_interp = trap_mod_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "trapz":
            x_interp = trap_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "hs":
            x_interp = hs_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "hs_mod":
            x_interp = hs_mod_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "hs_parab":
            U_c = scheme_params[0]
            u_c = U_c[n]
            x_interp = hs_parab_interp(x, x_n, u, u_n, tau, F, h, params, u_c)
        elif scheme == "hs_mod_parab":
            U_c = scheme_params[0]
            u_c = U_c[n]
            x_interp = hs_mod_parab_interp(x, x_n, u, u_n, tau, F, h, params, u_c)
        else:
            raise NameError(f"scheme {scheme} not recognized")
    return x_interp


def _prepare_interp(X, U, F, h, t_array):
    N = t_array.size
    arr_width = X.shape[-1]
    new_X = zeros([N, arr_width])
    if X.shape[0] == U.shape[0] + 1:
        U = extend_array(U)
    if X.shape[0] != U.shape[0]:
        raise ValueError("X and U have incompatible sizes")
    old_t_array = linspace(0, (X.shape[0] - 1) * h, X.shape[0])
    if t_array[-1] - old_t_array[-1] > h * 1e-9:
        raise ValueError(
            f"Proposed time array{t_array[-1]} extends outside interpolation{old_t_array[-1]}"
        )

    return N, new_X, U, old_t_array


def interpolate_u(U, old_t_array, t_array, u_scheme="lin", scheme_params=0):
    if u_scheme == "lin":
        if len(U.shape) == 1:
            new_U = interp(t_array, old_t_array, U)
        elif len(U.shape) == 2:
            new_U = interp_2d(t_array, old_t_array, U)
        else:
            raise ValueError(
                f"U has {len(U.shape)} dimensions, values accepted are 1 and 2"
            )
    else:
        h = (old_t_array[-1] - old_t_array[0]) / (old_t_array.size - 1)
        N = t_array.size
        new_shape = list(U.shape)
        new_shape[0] = N
        new_U = zeros(new_shape)
        for ii in range(N):
            new_U[ii] = array(
                _newpoint_u(U, h, t_array[ii], u_scheme, scheme_params)
            ).flatten()
    return new_U


def interpolated_array(
    X, U, F, h, t_array, params, scheme="hs_scipy", u_scheme="lin", scheme_params=0
):
    supported_schemes = [
        "trapz",
        "trapz_mod",
        "hs",
        "hs_scipy",
        "hs_mod",
        "hs_parab",
        "hs_mod_parab",
    ]
    if scheme not in supported_schemes:
        raise ValueError(
            f"Unsupported scheme {scheme}, supported schemes are{supported_schemes}"
        )
    supported_u_schemes = [
        "lin",
        "parab",
    ]
    if u_scheme not in supported_u_schemes:
        raise ValueError(
            f"Unsupported u_scheme {u_scheme}, supported schemes are{supported_u_schemes}"
        )

    N, new_X, U, old_t_array = _prepare_interp(X, U, F, h, t_array)
    new_U = interpolate_u(U, old_t_array, t_array, u_scheme, scheme_params)
    if scheme == "hs_scipy":
        X_interp = hermite(old_t_array, X, F(X, U, params))
        new_X = X_interp(t_array)
    else:
        for ii in range(N):
            new_X[ii] = array(
                _newpoint(X, U, F, h, t_array[ii], params, scheme, scheme_params)
            ).flatten()
    return new_X, new_U


# --- Derivatives ---


def hs_dot_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        v
        + tau * (-3 * v + 4 * v_c - v_n) / h
        + tau ** 2 * (2 * v - 4 * v_c + 2 * v_n) / h ** 2
    )
    v_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    return concatenate([q_interp, v_interp])


def hs_parab_dot_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        v
        + tau * (-3 * v + 4 * v_c - v_n) / h
        + tau ** 2 * (2 * v - 4 * v_c + 2 * v_n) / h ** 2
    )
    v_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    return concatenate([q_interp, v_interp])


def hs_mod_dot_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    v_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    return concatenate([q_interp, v_interp])


def hs_mod_parab_dot_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        v
        + f * tau
        + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau ** 2 / h
        + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau ** 3 / (h ** 2)
    )
    v_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    return concatenate([q_interp, v_interp])


def hs_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (-3 * v + 4 * v_c - v_n) / h + 2 * tau * (
        2 * v - 4 * v_c + 2 * v_n
    ) / h ** 2
    v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
        2 * f - 4 * f_c + 2 * f_n
    ) / h ** 2
    return concatenate([q_interp, v_interp])


def hs_mod_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
        2 * f - 4 * f_c + 2 * f_n
    ) / h ** 2
    return concatenate([q_interp, v_interp])


def hs_parab_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (-3 * v + 4 * v_c - v_n) / h + 2 * tau * (
        2 * v - 4 * v_c + 2 * v_n
    ) / h ** 2
    v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
        2 * f - 4 * f_c + 2 * f_n
    ) / h ** 2
    return concatenate([q_interp, v_interp])


def hs_mod_parab_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params, scheme_params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = scheme_params
    dim = vec_len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    f_c = F(x_c, u_c, params)[dim:]
    q_interp = (
        f
        + tau * (-3 * f + 4 * f_c - f_n) / h
        + tau ** 2 * (2 * f - 4 * f_c + 2 * f_n) / h ** 2
    )
    v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
        2 * f - 4 * f_c + 2 * f_n
    ) / h ** 2
    return concatenate([q_interp, v_interp])


def _newpoint_der(X, U, F, h, t, params, scheme, order, scheme_params=0):
    n = int(t // h)
    tau = t % h
    if (n + 1) > X.shape[0]:
        raise ValueError(f"Value of time {t} detected outside interpolation limits")
    # print(f't = {t} , tau = {tau} , n = {n} , h = {h}')
    # if abs(tau) < h * 1e-8:
    #    x_interp = X[n]
    # elif abs(tau - h) < h * 1e-8:
    #    x_interp = X[n + 1]
    else:
        x, x_n, u, u_n = X[n], X[n + 1], U[n], U[n + 1]
        if scheme == "hs":
            if order == 1:
                x_interp = hs_dot_interp(x, x_n, u, u_n, tau, F, h, params)
            elif order == 2:
                x_interp = hs_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "hs_mod":
            if order == 1:
                x_interp = hs_mod_dot_interp(x, x_n, u, u_n, tau, F, h, params)
            elif order == 2:
                x_interp = hs_mod_dot_dot_interp(x, x_n, u, u_n, tau, F, h, params)
        elif scheme == "hs_parab":
            U_c = scheme_params[0]
            u_c = U_c[n]
            if order == 1:
                x_interp = hs_parab_dot_interp(x, x_n, u, u_n, tau, F, h, params, u_c)
            elif order == 2:
                x_interp = hs_parab_dot_dot_interp(
                    x, x_n, u, u_n, tau, F, h, params, u_c
                )
        elif scheme == "hs_mod_parab":
            U_c = scheme_params[0]
            u_c = U_c[n]
            if order == 1:
                x_interp = hs_mod_parab_dot_interp(
                    x, x_n, u, u_n, tau, F, h, params, u_c
                )
            elif order == 2:
                x_interp = hs_mod_parab_dot_dot_interp(
                    x, x_n, u, u_n, tau, F, h, params, u_c
                )
        else:
            raise NameError(f"scheme {scheme} not recognized")
    return x_interp


def interpolated_array_derivative(
    X, U, F, h, t_array, params, scheme="hs_scipy", order=1, scheme_params=0
):
    supported_order = [1, 2]
    if order not in supported_order:
        raise ValueError(
            f"Unsupported derivation order, supported order are{supported_order}"
        )
    supported_schemes = ["hs", "hs_scipy", "hs_mod", "hs_parab", "hs_mod_parab"]
    if scheme not in supported_schemes:
        raise ValueError(
            f"Unsupported scheme, supported schemes are{supported_schemes}"
        )

    N, new_X, U, old_t_array = _prepare_interp(X, U, F, h, t_array)
    if scheme == "hs_scipy":
        X_interp = hermite(old_t_array, X, F(X, U, params))
        X_dot = X_interp.derivative()
        if order == 1:
            new_X = X_dot(t_array)
        elif order == 2:
            new_X = X_dot.derivative()(t_array)
    else:
        for ii in range(N):
            new_X[ii] = array(
                _newpoint_der(
                    X, U, F, h, t_array[ii], params, scheme, order, scheme_params
                )
            ).flatten()
    return new_X


# --- Derivatives ---


def dynamic_error(
    x_arr,
    u_arr,
    t_end,
    F,
    params,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params=0,
    n_interp=2000,
    t_start=0,
):
    N = x_arr.shape[0] - 1
    dim = x_arr.shape[1] // 2
    h = t_end / N
    t_interp = linspace(t_start, t_end, n_interp)
    x_interp, u_interp = interpolated_array(
        x_arr, u_arr, F, h, t_interp, params, scheme, u_scheme, scheme_params
    )
    x_dot_interp = interpolated_array_derivative(
        x_arr, u_arr, F, h, t_interp, params, scheme, 1, scheme_params
    )
    x_dot_dot_interp = interpolated_array_derivative(
        x_arr, u_arr, F, h, t_interp, params, scheme, 2, scheme_params
    )
    f_arr = zeros([n_interp, dim])
    for ii in range(n_interp):
        f_arr[ii, :] = F(x_interp[ii], u_interp[ii], params)[dim:]
    dyn_err_q = x_dot_interp[:, :dim] - x_interp[:, dim:]
    dyn_err_v = x_dot_interp[:, dim:] - f_arr
    dyn_err_2 = x_dot_dot_interp[:, :dim] - f_arr
    return dyn_err_q, dyn_err_v, dyn_err_2
