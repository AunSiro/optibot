#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:52:34 2021

@author: Siro Moreno
"""

from scipy.optimize import root
from numpy import zeros, append, linspace, expand_dims, interp
from scipy.interpolate import CubicHermiteSpline as hermite


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


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
    if not mode in ["numpy", "casadi"]:
        raise NameError(f"Unrecognized mode: {mode}")
    if mode == "numpy":
        from numpy import concatenate, array

        def new_F(x, u, params):
            x = array(x)
            a = F(x, u, params)
            dim = x.shape[-1] // 2
            axnum = len(x.shape) - 1
            if axnum == 1:
                v = x[:, dim:]
            else:
                v = x[dim:]
            res = concatenate((v, array(a)), axnum)
            return res

    elif mode == "casadi":
        from casadi import horzcat

        def new_F(x, u, params):
            a = F(x, u, params)
            x = horzcat(x)
            if x.shape[-1] == 1:
                x = x.T
            dim = x.shape[-1] // 2
            v = array(x)[:, dim:]
            res = horzcat(v, a)
            return res

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
    x_n = root(trapz_opti_step, x, (x, u, u_n, F, dt, params))
    return x_n.x


def trapz_mod_opti_step(x_n, x, u, u_n, F, dt, params):
    dim = len(x) // 2
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    res = x.copy()
    res[dim:] = x[dim:] + dt / 2 * (f + f_n) - x_n[dim:]
    res[:dim] = x[:dim] + dt * x[dim:] + dt ** 2 / 6 * (f_n + 2 * f) - x_n[:dim]
    return res


def trapz_mod_step(x, u, u_n, F, dt, params):
    x_n = root(trapz_mod_opti_step, x, (x, u, u_n, F, dt, params))
    return x_n.x


# --- Integrations ---


def integrate_euler(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(len(u)):
        x_i = euler_step(x[-1], u[ii], F, dt, params)
        x.append(x_i)
    return x


def integrate_rk4(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(len(u)):
        x_i = rk4_step(x[-1], u[ii], F, dt, params)
        x.append(x_i)
    return x


def integrate_trapz(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, len(u) - 1):
        x_i = trapz_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = trapz_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return x


def integrate_trapz_mod(x_0, u, F, dt, params):
    x = [
        x_0,
    ]
    for ii in range(0, len(u) - 1):
        x_i = trapz_step(x[-1], u[ii], u[ii + 1], F, dt, params)
        x.append(x_i)
    x_i = trapz_mod_step(x[-1], u[-1], u[-1], F, dt, params)
    x.append(x_i)
    return x


# --- Interpolations ---


def trap_mod_interp(x, x_n, u, u_n, tau, F, h, params):
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    q_interp = q + v * tau + 1 / 2 * f * tau ** 2 + 1 / (6 * h) * tau ** 3 * (f_n - f)
    v_interp = v + tau * f + tau ** 2 / (2 * h) * (f_n - f)
    return q_interp, v_interp


def trap_interp(x, x_n, u, u_n, tau, F, h, params):
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    q_interp = q + v * tau + 1 / (2 * h) * tau ** 2 * (v_n - v)
    v_interp = v + tau * f + tau ** 2 / (2 * h) * (f_n - f)
    return q_interp, v_interp


def hs_midpoint(x, x_n, u, u_n, tau, F, h, params):
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    v_c = (v + v_n) / 2 + h / 8 * (f - f_n)
    q_c = (q + q_n) / 2 + h / 8 * (v - v_n)
    return q_c, v_c


def hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params):
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    v_c = (v + v_n) / 2 + h / 8 * (f - f_n)
    q_c = (13 * q + 3 * q_n + 5 * v * h) / 16 + h ** 2 / 96 * (4 * f - f_n)
    return q_c, v_c


def hs_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    f_c = F(x_c, u_c, params)[:dim]
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
    return q_interp, v_interp


def hs_mod_interp(x, x_n, u, u_n, tau, F, h, params):
    x_c = hs_mod_midpoint(x, x_n, u, u_n, tau, F, h, params)
    u_c = (u + u_n) / 2
    dim = len(x) // 2
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = x_c[:dim]
    v_c = x_c[dim:]
    f = F(x, u, params)[:dim]
    f_n = F(x_n, u_n, params)[:dim]
    f_c = F(x_c, u_c, params)[:dim]
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
    return q_interp, v_interp


def newpoint(X, U, F, h, t, params, scheme):
    n = int(t // h)
    tau = t % h
    # print(f't = {t} , tau = {tau} , n = {n} , h = {h}')
    if abs(tau) < h * 1e-8:
        x_interp = X[n]
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
        else:
            raise NameError(f"scheme {scheme} not recognized")
    return x_interp


def extend_array(x):
    apppendix = expand_dims(x[-1], axis=0)
    return append(x, apppendix, 0)


def interpolated_array(X, U, F, h, t_array, params, scheme="hs"):
    N = t_array.size
    new_X = zeros(N)
    if X.shape[-1] == U.shape[-1] + 1:
        U = extend_array(U)
    if X.shape[-1] != U.shape[-1]:
        raise ValueError("X and U have incompatible sizes")
    old_t_array = linspace(0, (X.shape[0] - 1) * h, X.shape[0])
    if t_array[-1] - old_t_array[-1] > h * 1e-9:
        raise ValueError("Proposed time array extends outside interpolation")
    # print(Q.size, U.size, old_t_array.size, Q, U, old_t_array)
    new_U = interp(t_array, old_t_array, U)
    if scheme == "hs_scipy":
        X_interp = hermite(old_t_array, X, F(X, U, params))
        new_X = X_interp(t_array)
    else:
        for ii in range(N):
            new_X[ii] = newpoint(X, U, F, h, t_array[ii], params, scheme)
    return new_X, new_U
