#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:52:34 2021

@author: Siro Moreno

Here we present a collection of functions that describe different numerical
schemes, expressed explicitly and implicitly. Related interpolation and 
auxiliar functions are also defined here.
"""

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
from math import factorial as fac
from functools import lru_cache, wraps


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
    # try:
    #     return len(x)
    # except TypeError:
    if x.size == 1:
        return 1
    else:
        if x.shape[0] == 1 and len(x.shape) == 2:
            return x.shape[1]
        else:
            return x.shape[0]


def interp_2d(t_array, old_t_array, Y):
    """
    Interpolates a 2D array 'Y' in dimension 0th

    Parameters
    ----------
    t_array : Numpy Array or List,
        shape = (X)
    old_t_array : Numpy Array or List,
        shape = (Y)
    Y : Numpy Array,
        shape = (Y, Z)

    Returns
    -------
    new_Y : Numpy Array,
        shape = (X, Z)

    """
    new_Y_len = t_array.shape[0]
    new_Y_width = Y.shape[-1]
    new_Y = zeros([new_Y_len, new_Y_width])
    for ii in range(new_Y_width):
        new_Y[:, ii] = interp(t_array, old_t_array, Y[:, ii])
    return new_Y


def extend_array(x):
    """
    Extends an array in dimension 0th by duplicating the last value/s

    Parameters
    ----------
    x : Numpy Array
        Shape = (Y, [...])

    Returns
    -------
    new_x : Numpy Array
        Shape = (Y+1, [...])

    """
    apppendix = expand_dims(x[-1], axis=0)
    return append(x, apppendix, 0)


def expand_F(F, mode="numpy", order=2):
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
    order: int, default 2
        differential order of the problem. For example, if your
        system is like:
            q''' = F(x, u, params),
            q'' = a,
            q' = v
        the order must be set to 3

    Returns
    -------
    Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)

    """
    try:
        old_docstring = str(F.__doc__)
    except:
        old_docstring = """function of (x, u, params)
        A function of a dynamic sistem, so that
            v' = F(x, u, params),
            q' = v"""
    try:
        old_f_name = str(F.__name__)
    except:
        old_f_name = "Function"
        
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
            dim = x.shape[-1] // order
            if axnum == 1:
                v = x[:, dim:]
            else:
                v = x[dim:]
            if is_iterable(a):
                new_a = array(a)
            else:
                new_a = array(
                    [
                        a,
                    ]
                )
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
            dim = x.shape[-1] // order
            v = x[:, dim:]
            res = horzcat(v, a)
            if x_transposed and res.shape[0] == 1:
                res = res.T
                # If the input was x as a vertical array of
                # width 1, we must return a result
                # dimensionally consistend.
            return res

    else:
        raise NameError(f"Unrecognized mode: {mode}")

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


def expand_G(G, mode="numpy", order=2):
    """
    Expands a function G(q, q', u, params) that returns accelerations,
    so that the new function return accelerations and velocities.

    Parameters
    ----------
    G : function of (q, v, u, params)
        A function of a dynamic sistem, so that
            v' = G(q, v, u, params),
            q' = v
    mode : str: 'numpy' o 'casadi', optional
        Wether the function is a numpy or a casadi function.
        The default is "numpy".
    order: int, default 2
        differential order of the problem. For example, if your
        system is like:
            q''' = G(q, q', q'', u, params)
        the order must be set to 3

    Returns
    -------
    Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)

    """
    try:
        old_docstring = str(G.__doc__)
    except:
        old_docstring = """function of (q, v, u, params)
        G : function of (q, v, u, params)
        A function of a dynamic sistem, so that
            v' = G(q, v, u, params),
            q' = v"""
    try:
        old_f_name = str(G.__name__)
    except:
        old_f_name = "Function"

    def F(x, u, params):
        dim = x.shape[-1] // order
        q_and_der = []
        if len(x.shape) == 1:
            for _ii in range(order):
                q_and_der.append(x[dim * _ii : dim * (_ii + 1)])
        elif len(x.shape) == 2:
            for _ii in range(order):
                q_and_der.append(x[:, dim * _ii : dim * (_ii + 1)])
        else:
            raise ValueError("Unsupported array shape")
        return G(*q_and_der, u, params)

    F.__name__ = old_f_name
    F.__doc__ = old_docstring
    new_F = expand_F(F, mode)
    return new_F


def reduce_F(F, mode="numpy", order=2):
    """
    Extract a function G(q, q', u, params) such that q'' = G(q, q', u, params)
    from a function F(x, u, params) such that x' = F(x, u, params)

    Parameters
    ----------
    F : Function
        function F(x, u, params) such that x' = F(x, u, params)
    mode : str: 'numpy' o 'casadi', optional
        Wether the function is a numpy or a casadi function.
        The default is "numpy".
    order: int, default 2
        differential order of the problem. For example, if your
        system is like:
            q''' = G(q, q', q'', u, params),
            q'' = a,
            q' = v
        the order must be set to 3

    Returns
    -------
    G : Function
        function G(q, q', u, params) such that  q'' = G(q, q', u, params)
        (or the equivalent for a higher order equation)

    """
    try:
        old_docstring = str(F.__doc__)
    except:
        old_docstring = """function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)"""
    try:
        old_f_name = str(F.__name__)
    except:
        old_f_name = "Function"
        
    if mode == "numpy":

        def G(*args):
            q = args[0]
            params = args[-1]
            u = args[-2]
            q_and_der = args[:-2]
            dim = q.shape[-1]
            axnum = len(q.shape) - 1
            x = concatenate(q_and_der, axnum)
            res = F(x, u, params)
            if axnum == 0:
                aa = res[-dim:]
            elif axnum == 1:
                aa = res[:, -dim:]
            else:
                raise ValueError(f"unsupported shape por q: {q.shape}")
            return aa

    elif mode == "casadi":
        from casadi import horzcat

        def G(*args):
            q = args[0]
            params = args[-1]
            u = args[-2]
            q_and_der = args[:-2]
            dim = q.shape[-1]
            x = horzcat(*q_and_der)
            res = F(x, u, params)
            aa = res[:, -dim:]
            return aa

    else:
        raise NameError(f"Unrecognized mode: {mode}")

    new_docstring = f"""
    This is an reduced version of function {old_f_name}.
    This reduced function is designed to describe a dinamic sistem so that:
        q'' = G(q, q', u, params)
    While the old function was:
        x' = F(x, u, params)
    Old function documentation:
    """
    new_docstring += old_docstring
    G.__doc__ = new_docstring
    return G


# --- Integration Steps ---


def euler_step(x, u, F, dt, params):
    return x + dt * F(x, u, params)


def rk4_step(x, u, u_n, F, dt, params, u_c):
    k1 = F(x, u, params)
    k2 = F(x + dt / 2 * k1, u_c, params)
    k3 = F(x + dt / 2 * k2, u_c, params)
    k4 = F(x + dt * k3, u_n, params)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def trapz_opti_step(x_n, x, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    res = x + dt / 2 * (f + f_n) - x_n
    return res


def trapz_step(x, u, u_n, F, dt, params, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(trapz_opti_step, x_0, (x, u, u_n, F, dt, params), **root_kwargs)
    return x_n.x


def trapz_mod_opti_step(x_n, x, u, u_n, F, dt, params):
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    res = copy(x)
    res[dim:] = x[dim:] + dt / 2 * (f + f_n) - x_n[dim:]
    res[:dim] = x[:dim] + dt * x[dim:] + dt**2 / 6 * (f_n + 2 * f) - x_n[:dim]
    return res


def trapz_mod_step(x, u, u_n, F, dt, params, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(trapz_mod_opti_step, x_0, (x, u, u_n, F, dt, params), **root_kwargs)
    return x_n.x


def hs_opti_step(x_n, x, u, u_n, F, dt, params):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    u_c = (u + u_n) / 2
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    f_c = F(x_c, u_c, params)
    res = x + dt / 6 * (f + 4 * f_c + f_n) - x_n
    return res


def hs_step(x, u, u_n, F, dt, params, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_opti_step, x_0, (x, u, u_n, F, dt, params), **root_kwargs)
    return x_n.x


def hs_mod_opti_step(x_n, x, u, u_n, F, dt, params):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    u_c = (u + u_n) / 2
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt**2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 6 * (f + 4 * f_c + f_n) - v_n
    res[:dim] = q + dt * v + dt**2 / 6 * (f + 2 * f_c) - q_n
    return res


def hs_mod_step(x, u, u_n, F, dt, params, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_mod_opti_step, x_0, (x, u, u_n, F, dt, params), **root_kwargs)
    return x_n.x


def hs_parab_opti_step(x_n, x, u, u_n, F, dt, params, u_c):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    f_c = F(x_c, u_c, params)
    res = x + dt / 6 * (f + 4 * f_c + f_n) - x_n
    return res


def hs_parab_step(x, u, u_n, F, dt, params, u_c, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hs_parab_opti_step, x_0, (x, u, u_n, F, dt, params, u_c), **root_kwargs)
    return x_n.x


def hs_mod_parab_opti_step(x_n, x, u, u_n, F, dt, params, u_c):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt**2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 6 * (f + 4 * f_c + f_n) - v_n
    res[:dim] = q + dt * v + dt**2 / 6 * (f + 2 * f_c) - q_n
    return res


def hs_mod_parab_step(x, u, u_n, F, dt, params, u_c, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(
        hs_mod_parab_opti_step, x_0, (x, u, u_n, F, dt, params, u_c), **root_kwargs
    )
    return x_n.x


def hsj_opti_step(x_n, x, u, u_n, F, dt, params):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    u_c = 0.6 * u + 0.4 * u_n
    q_c = q + (218 * v + 32 * v_n) * dt / 625 + (82 * f - 28 * f_n) * dt**2 / 1875
    v_c = (81 * v + 44 * v_n) / 125 + (18 * f - 12 * f_n) * dt / 125
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 36 * (3 * f + 25 * f_c + 8 * f_n) - v_n
    res[:dim] = q + dt * v + dt**2 / 72 * (9 * f + 25 * f_c + 2 * f_n) - q_n
    return res


def hsj_step(x, u, u_n, F, dt, params, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hsj_opti_step, x_0, (x, u, u_n, F, dt, params), **root_kwargs)
    return x_n.x


def hsj_parab_opti_step(x_n, x, u, u_n, F, dt, params, u_c):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """
    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    f_n = F(x_n, u_n, params)[dim:]
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    q_c = q + (218 * v + 32 * v_n) * dt / 625 + (82 * f - 28 * f_n) * dt**2 / 1875
    v_c = (81 * v + 44 * v_n) / 125 + (18 * f - 12 * f_n) * dt / 125
    x_c = copy(x)
    x_c[:dim] = q_c
    x_c[dim:] = v_c
    f_c = F(x_c, u_c, params)[dim:]
    res = copy(x)
    res[dim:] = v + dt / 36 * (3 * f + 25 * f_c + 8 * f_n) - v_n
    res[:dim] = q + dt * v + dt**2 / 72 * (9 * f + 25 * f_c + 2 * f_n) - q_n
    return res


def hsj_parab_step(x, u, u_n, F, dt, params, u_c, root_kwargs=None):
    if root_kwargs is None:
        root_kwargs = {}
    x_0 = euler_step(x, u, F, dt, params)
    x_n = root(hsj_parab_opti_step, x_0, (x, u, u_n, F, dt, params, u_c), **root_kwargs)
    return x_n.x


# --- Integrations ---
# These functions are expected to work with numpy arrays, and will
# convert other formats for X and U into them


def coherent_dimensions(func):
    """
    Adapts input variables to ensure that they are compatible
    with functions of structure integrate_x(x_0, u, F, dt, *add_params)
    where the first element of *add_params is params, and may contain
    additional elements such as scheme_params

    Parameters
    ----------
    func : Function
        Integration function whose structure is F(x_0, u, F, dt, params).
    -------
    Function
        The same function, but with additional comprobations
        that the input variables are coherent.

    """

    @wraps(func)
    def wrapper_decorator(x_0, u, F, dt, *add_params, **kwargs):
        if "params" in kwargs:
            params = kwargs["params"]
        else:
            params = add_params[0]
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
        # it can mean that it corresponds to only one step
        # but it also can be an underactuated problem with just one u
        if len(u.shape) == 1 and x_0.shape[0] != 2:
            try:
                F(x_0, u, params)
            except TypeError:
                pass  # u is a 1D a control tape
            else:
                u = expand_dims(u, axis=0)  # u is 1 step
        value = func(x_0, u, F, dt, *add_params, **kwargs)
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
def integrate_rk4(x_0, u, F, dt, params, scheme_params={}):
    x = [
        x_0,
    ]
    if "u_c" in scheme_params:
        u_c = scheme_params["u_c"]
    else:
        u_c = (u[:-1] + u[1:]) / 2
    for ii in range(vec_len(u) - 1):
        x_i = rk4_step(x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii])
        x.append(x_i)
    x_i = rk4_step(x[-1], u[-1], u[-1], F, dt, params, u[-1])
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_trapz(x_0, u, F, dt, params, root_kwargs={}):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = trapz_step(x[-1], u[ii], u[ii + 1], F, dt, params, root_kwargs)
        x.append(x_i)
    x_i = trapz_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_trapz_mod(x_0, u, F, dt, params, root_kwargs={}):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = trapz_mod_step(x[-1], u[ii], u[ii + 1], F, dt, params, root_kwargs)
        x.append(x_i)
    x_i = trapz_mod_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs(x_0, u, F, dt, params, root_kwargs={}):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_step(x[-1], u[ii], u[ii + 1], F, dt, params, root_kwargs)
        x.append(x_i)
    x_i = hs_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_mod(x_0, u, F, dt, params, root_kwargs={}):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_mod_step(x[-1], u[ii], u[ii + 1], F, dt, params, root_kwargs)
        x.append(x_i)
    x_i = hs_mod_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_parab(x_0, u, F, dt, params, scheme_params, root_kwargs={}):
    x = [
        x_0,
    ]
    u_c = scheme_params["u_c"]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_parab_step(
            x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii], root_kwargs
        )
        x.append(x_i)
    x_i = hs_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hs_mod_parab(x_0, u, F, dt, params, scheme_params, root_kwargs={}):
    x = [
        x_0,
    ]
    u_c = scheme_params["u_c"]
    for ii in range(0, vec_len(u) - 1):
        x_i = hs_mod_parab_step(
            x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii], root_kwargs
        )
        x.append(x_i)
    x_i = hs_mod_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hsj(x_0, u, F, dt, params, root_kwargs={}):
    x = [
        x_0,
    ]
    for ii in range(0, vec_len(u) - 1):
        x_i = hsj_step(x[-1], u[ii], u[ii + 1], F, dt, params, root_kwargs)
        x.append(x_i)
    x_i = hsj_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


@coherent_dimensions
def integrate_hsj_parab(x_0, u, F, dt, params, scheme_params, root_kwargs={}):
    x = [
        x_0,
    ]
    u_c = scheme_params["u_c"]
    for ii in range(0, vec_len(u) - 1):
        x_i = hsj_parab_step(
            x[-1], u[ii], u[ii + 1], F, dt, params, u_c[ii], root_kwargs
        )
        x.append(x_i)
    x_i = hsj_step(x[-1], u[-1], u[-1], F, dt, params, root_kwargs)
    x.append(x_i)
    return array(x)


# --- Schemes as Restrictions ---


def index_div(x, order=2):
    dim = vec_len(x) // order
    if is2d(x):
        first_ind = slice(None, None), slice(None, dim)
        last_ind = slice(None, None), slice(-dim, None)
        all_but_first_ind = slice(None, None), slice(dim, None)
        all_but_last_ind = slice(None, None), slice(None, -dim)
    else:
        first_ind = slice(None, dim)
        last_ind = slice(-dim, None)
        all_but_first_ind = slice(dim, None)
        all_but_last_ind = slice(None, -dim)
    return first_ind, last_ind, all_but_first_ind, all_but_last_ind


def order_from_x_a(x, a):
    dim = vec_len(a)
    dim_x = vec_len(x)
    order = dim_x // dim
    return order


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
    order = 2  # This scheme is specifically designed for 2nd order
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    res[last_ind] = x[last_ind] + dt / 2 * (f + f_n)
    res[first_ind] = x[first_ind] + dt * x[last_ind] + dt**2 / 6 * (f_n + 2 * f)
    return x_n - res


def hs_restr(x, x_n, u, u_n, F, dt, params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    u_c = (u + u_n) / 2
    f_c = F(x_c, u_c, params)
    return x + dt / 6 * (f + 4 * f_c + f_n) - x_n


def hs_mod_restr(x, x_n, u, u_n, F, dt, params):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = (u + u_n) / 2
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt**2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 6 * (f + 4 * f_c + f_n)
    res[first_ind] = q + dt * v + dt**2 / 6 * (f + 2 * f_c)
    return x_n - res


def hs_parab_restr(x, x_n, u, u_n, F, dt, params, scheme_params):
    f = F(x, u, params)
    f_n = F(x_n, u_n, params)
    x_c = (x + x_n) / 2 + dt / 8 * (f - f_n)
    u_c = scheme_params
    f_c = F(x_c, u_c, params)
    return x + dt / 6 * (f + 4 * f_c + f_n) - x_n


def hs_mod_parab_restr(x, x_n, u, u_n, F, dt, params, scheme_params):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = scheme_params
    q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt**2 / 96 * (4 * f - f_n)
    v_c = (v + v_n) / 2 + dt / 8 * (f - f_n)
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 6 * (f + 4 * f_c + f_n)
    res[first_ind] = q + dt * v + dt**2 / 6 * (f + 2 * f_c)
    return x_n - res


def hsj_restr(x, x_n, u, u_n, F, dt, params):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    # q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = 0.6 * u + 0.4 * u_n
    q_c = q + (218 * v + 32 * v_n) * dt / 625 + (82 * f - 28 * f_n) * dt**2 / 1875
    v_c = (81 * v + 44 * v_n) / 125 + (18 * f - 12 * f_n) * dt / 125
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 36 * (3 * f + 25 * f_c + 8 * f_n)
    res[first_ind] = q + dt * v + dt**2 / 72 * (9 * f + 25 * f_c + 2 * f_n)
    return x_n - res


def hsj_parab_restr(x, x_n, u, u_n, F, dt, params, scheme_params):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    f = F(x, u, params)[last_ind]
    f_n = F(x_n, u_n, params)[last_ind]
    q = x[first_ind]
    v = x[last_ind]
    # q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    u_c = scheme_params
    q_c = q + (218 * v + 32 * v_n) * dt / 625 + (82 * f - 28 * f_n) * dt**2 / 1875
    v_c = (81 * v + 44 * v_n) / 125 + (18 * f - 12 * f_n) * dt / 125
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    f_c = F(x_c, u_c, params)[last_ind]
    res[last_ind] = v + dt / 36 * (3 * f + 25 * f_c + 8 * f_n)
    res[first_ind] = q + dt * v + dt**2 / 72 * (9 * f + 25 * f_c + 2 * f_n)
    return x_n - res


# --- Schemes as Acceleration Restrictions ---


def euler_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    order = order_from_x_a(x, a)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    x_d = copy(x)
    x_d[all_but_last_ind] = x[all_but_first_ind]
    x_d[last_ind] = a
    return x_n - (x + dt * x_d)


# def rk4_accel_restr(x, x_n, a, a_n, dt, params, scheme_params):
#     k1 = F(x, u, params)
#     k2 = F(x + dt / 2 * k1, u, params)
#     k3 = F(x + dt / 2 * k2, u, params)
#     k4 = F(x + dt * k3, u, params)
#     return x_n - (x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def trapz_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    order = order_from_x_a(x, a)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    x_d = copy(x)
    x_d[all_but_last_ind] = x[all_but_first_ind]
    x_d[last_ind] = a
    x_d_n = copy(x)
    x_d_n[all_but_last_ind] = x_n[all_but_first_ind]
    x_d_n[last_ind] = a_n
    return x_n - (x + dt / 2 * (x_d + x_d_n))


def trapz_mod_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    order = 2  # This scheme is specifically designed for 2nd order
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    res[last_ind] = x[last_ind] + dt / 2 * (a + a_n)
    res[first_ind] = x[first_ind] + dt * x[last_ind] + dt**2 / 6 * (a_n + 2 * a)
    return x_n - res


def hs_half_x(x, x_n, a, a_n, dt):
    order = order_from_x_a(x, a)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    x_d = copy(x)
    x_d[all_but_last_ind] = x[all_but_first_ind]
    x_d[last_ind] = a

    x_d_n = copy(x)
    x_d_n[all_but_last_ind] = x_n[all_but_first_ind]
    x_d_n[last_ind] = a_n

    x_c = (x + x_n) / 2 + dt / 8 * (x_d - x_d_n)
    return x_c


def hs_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    a_c = scheme_params
    order = order_from_x_a(x, a)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    x_d = copy(x)
    x_d[all_but_last_ind] = x[all_but_first_ind]
    x_d[last_ind] = a

    x_d_n = copy(x)
    x_d_n[all_but_last_ind] = x_n[all_but_first_ind]
    x_d_n[last_ind] = a_n

    x_c = hs_half_x(x, x_n, a, a_n, dt)
    x_d_c = copy(x)
    x_d_c[all_but_last_ind] = x_c[all_but_first_ind]
    x_d_c[last_ind] = a_c

    res = x + dt / 6 * (x_d + 4 * x_d_c + x_d_n)
    return x_n - res


def hs_mod_half_x(x, x_n, a, a_n, dt):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    q = x[first_ind]
    v = x[last_ind]
    # q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    # q_c = (13 * q + 3 * q_n) / 16 + 5 * dt / 16 * v + dt**2 / 96 * (4 * a - a_n)
    q_c = q + dt / 32 * (13 * v + 3 * v_n) + dt**2 / 192 * (11 * a - 5 * a_n)
    v_c = (v + v_n) / 2 + dt / 8 * (a - a_n)
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    return x_c


def hs_mod_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    order = 2  # This scheme is specifically designed for 2nd order
    a_c = scheme_params
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    q = x[first_ind]
    v = x[last_ind]
    res[last_ind] = v + dt / 6 * (a + 4 * a_c + a_n)
    res[first_ind] = q + dt * v + dt**2 / 6 * (a + 2 * a_c)
    return x_n - res


def hsj_half_x(x, x_n, a, a_n, dt):
    order = 2  # This scheme is specifically designed for 2nd order
    x_c = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    q = x[first_ind]
    v = x[last_ind]
    q_n = x_n[first_ind]
    v_n = x_n[last_ind]
    # q_c = q + (218 * v + 32 * v_n)*dt/625 +(82*a -28*a_n)*dt**2/1875
    q_c = (561 * q + 64 * q_n + 186 * dt * v + dt**2 * (22 * a - 4 * a_n)) / 625
    v_c = (81 * v + 44 * v_n) / 125 + (18 * a - 12 * a_n) * dt / 125
    x_c[first_ind] = q_c
    x_c[last_ind] = v_c
    return x_c


def hsj_accel_restr(x, x_n, a, a_n, dt, scheme_params):
    order = 2  # This scheme is specifically designed for 2nd order
    a_c = scheme_params
    res = copy(x)
    first_ind, last_ind, all_but_first_ind, all_but_last_ind = index_div(x, order)
    q = x[first_ind]
    v = x[last_ind]
    res[last_ind] = v + dt / 36 * (3 * a + 25 * a_c + 8 * a_n)
    res[first_ind] = q + dt * v + dt**2 / 72 * (9 * a + 25 * a_c + 2 * a_n)
    return x_n - res


# Oder M:


def get_x_divisions(x, order=2, return_indices=False):
    dim = x.shape[-1] // order
    x_list = []
    if is2d(x):
        for ii in range(order):
            index = (slice(None, None), slice(dim * ii, dim * (ii + 1)))
            if return_indices:
                x_list.append(index)
            else:
                x_list.append(x[index])
    else:
        for ii in range(order):
            index = slice(dim * ii, dim * (ii + 1))
            if return_indices:
                x_list.append(index)
            else:
                x_list.append(x[index])
    return x_list


def _itemequal(arr_to, arr_from, shape, index=()):
    if len(shape) == 1:
        for ii in range(shape[0]):
            index_i = index + (ii,)
            arr_to[index_i] = arr_from[index_i]
    else:
        for ii in range(shape[0]):
            index_i = index + (ii,)
            shape_i = shape[1:]
            _itemequal(arr_to, arr_from, shape_i, index_i)


def itemequal(arr_to, arr_from):
    assert arr_from.shape == arr_to.shape
    shape = arr_from.shape
    _itemequal(arr_to, arr_from, shape, index=())


def reunite_matrix(arr, arr_list, order):
    indices = get_x_divisions(arr, order, return_indices=True)
    for ii in range(len(indices)):
        arr[indices[ii]] = arr_list[ii]


def generate_trapz_m_func(order):
    M = order  # in order to keep the paper naming
    q_coefs = zeros(order)
    for i in range(order):
        q_coefs[i] = 1 / fac(i)
    g_coefs = zeros(order + 1)
    for l in range(order + 1):
        g_coefs[l] = 1 / fac(l + 1)

    def trapz_m_accel_restr(x, x_n, a, a_n, dt, scheme_params):
        res = copy(x)
        x_list = get_x_divisions(x, order)
        res_list = get_x_divisions(res, order)
        for l in range(1, M + 1):
            _t = dt**l * g_coefs[l] * (l * a + a_n)
            itemequal(res_list[M - l], _t)
            for i in range(l):
                _t = res_list[M - l] + dt**i * q_coefs[i] * x_list[i + M - l]
                itemequal(res_list[M - l], _t)
        reunite_matrix(res, res_list, order)
        return x_n - res

    return trapz_m_accel_restr


def generate_hs_m_funcs(order):
    M = order  # in order to keep the paper naming
    q_coefs = zeros(order)
    for i in range(order):
        q_coefs[i] = 1 / fac(i)
    q_coefs_c = zeros(order - 1)
    for i in range(order - 1):
        q_coefs_c[i] = 1 / (2**i * fac(i))
    g_coefs = zeros(order + 1)
    g_coefs_c = zeros([order + 1, 5])
    for l in range(1, order + 1):
        g_coefs[l] = fac(l + 2)
        g_coefs_c[l, 0] = 2**l * fac(l) * (l + 2)
        g_coefs_c[l, 1] = 2 * l**2 + 4 * l - 3
        g_coefs_c[l, 2] = 2 ** (l + 1) * fac(l + 2)
        g_coefs_c[l, 3] = 2 * l**2 + 2 * l - 1
        g_coefs_c[l, 4] = -2 * l - 1

    def hs_m_half_x(x, x_n, a, a_n, dt):
        x_c = copy(x)
        x_list = get_x_divisions(x, order)
        x_n_list = get_x_divisions(x_n, order)
        x_c_list = get_x_divisions(x_c, order)
        v = x_list[-1]  # V represents here the highest q derivative contained in x
        v_n = x_n_list[-1]
        for l in range(1, M + 1):
            _t = dt ** (l - 1) / g_coefs_c[l, 0] * (
                3 * v_n + g_coefs_c[l, 1] * v
            ) + dt**l / g_coefs_c[l, 2] * (
                g_coefs_c[l, 3] * a + g_coefs_c[l, 4] * a_n
            )
            itemequal(x_c_list[M - l], _t)
            for i in range(l - 1):
                _t = x_c_list[M - l] + dt**i * q_coefs_c[i] * x_list[i + M - l]
                itemequal(x_c_list[M - l], _t)
        reunite_matrix(x_c, x_c_list, order)
        return x_c

    def hs_m_accel_restr(x, x_n, a, a_n, dt, scheme_params):
        a_c = scheme_params
        res = copy(x)
        x_list = get_x_divisions(x, order)
        res_list = get_x_divisions(res, order)
        for l in range(1, M + 1):
            _t = dt**l / g_coefs[l] * (l**2 * a + 4 * l * a_c + (2 - l) * a_n)
            itemequal(res_list[M - l], _t)
            for i in range(l):
                _t = res_list[M - l] + dt**i * q_coefs[i] * x_list[i + M - l]
                itemequal(res_list[M - l], _t)
        reunite_matrix(res, res_list, order)
        return x_n - res

    return hs_m_half_x, hs_m_accel_restr


# --- Interpolations ---


def _gen_basic_values(dim, x, x_n, x_dot, x_dot_n):
    q = x[:dim]
    v = x[dim:]
    q_n = x_n[:dim]
    v_n = x_n[dim:]
    f = x_dot[dim:]
    f_n = x_dot_n[dim:]
    return q, q_n, v, v_n, f, f_n


def interp_parab(tau, h, y_0, y_c, y_n):
    xi = tau / h
    return y_0 + xi * (-3 * y_0 + 4 * y_c - y_n) + 2 * xi**2 * (y_0 - 2 * y_c + y_n)


def interp_parab_j(tau, h, y_0, y_c, y_n):
    xi = tau / h
    return (
        y_0
        + xi / 6 * (-21 * y_0 + 25 * y_c - 4 * y_n)
        + xi**2 / 6 * (15 * y_0 - 25 * y_c + 10 * y_n)
    )


# def trap_mod_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     dim = vec_len(x) // 2
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     q_interp = q + v * tau + 1 / 2 * f * tau**2 + 1 / (6 * h) * tau**3 * (f_n - f)
#     v_interp = v + tau * f + tau**2 / (2 * h) * (f_n - f)
#     return concatenate([q_interp, v_interp])


# def trap_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     x_interp = x + x_dot * tau + 1 / (2 * h) * tau**2 * (x_dot_n - x_dot)
#     return x_interp


def hs_midpoint(x, x_n, x_dot, x_dot_n, h):
    x_c = (x + x_n) / 2 + h / 8 * (x_dot - x_dot_n)
    return x_c


def hs_mod_midpoint(x, x_n, x_dot, x_dot_n, h, params):
    dim = vec_len(x) // 2
    q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
    v_c = (v + v_n) / 2 + h / 8 * (f - f_n)
    q_c = (13 * q + 3 * q_n + 5 * v * h) / 16 + h**2 / 96 * (4 * f - f_n)
    return concatenate([q_c, v_c])


def hsj_midpoint(x, x_n, x_dot, x_dot_n, h, params):
    dim = vec_len(x) // 2
    q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
    q_c = q + (218 * v + 32 * v_n) * h / 625 + (82 * f - 28 * f_n) * h**2 / 1875
    v_c = (81 * v + 44 * v_n) / 125 + (18 * f - 12 * f_n) * h / 125
    return concatenate([q_c, v_c])


# def hs_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     x_dot_c = scheme_params
#     x_interp = (
#         x
#         + x_dot * tau
#         + 1 / 2 * (-3 * x_dot + 4 * x_dot_c - x_dot_n) * tau**2 / h
#         + 1 / 3 * (2 * x_dot - 4 * x_dot_c + 2 * x_dot_n) * tau**3 / (h**2)
#     )
#     return x_interp


# def hs_mod_interp(x, x_n, x_dot, x_dot_n, tau, h,  scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n, h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         q
#         + v * tau
#         + 1 / 2 * f * tau**2
#         + 1 / 6 * (-3 * f + 4 * f_c - f_n) * tau**3 / h
#         + 1 / 12 * (2 * f - 4 * f_c + 2 * f_n) * tau**4 / (h**2)
#     )
#     v_interp = (
#         v
#         + f * tau
#         + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau**2 / h
#         + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau**3 / (h**2)
#     )
#     return concatenate([q_interp, v_interp])


# def hs_mod_parab_interp(x, x_n, x_dot, x_dot_n, tau, h, params, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n, h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         q
#         + v * tau
#         + 1 / 2 * f * tau**2
#         + 1 / 6 * (-3 * f + 4 * f_c - f_n) * tau**3 / h
#         + 1 / 12 * (2 * f - 4 * f_c + 2 * f_n) * tau**4 / (h**2)
#     )
#     v_interp = (
#         v
#         + f * tau
#         + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau**2 / h
#         + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau**3 / (h**2)
#     )
#     return concatenate([q_interp, v_interp])

# Order M expressions


@lru_cache(maxsize=None)
def generate_tz_interp(order, deriv=0):
    M = order
    q_coefs = [fac(i) for i in range(M + 2)]

    def tz_interp(x, x_n, x_dot, x_dot_n, tau, h):
        x_interp = copy(x)
        x_list = get_x_divisions(x, order)
        x_dot_list = get_x_divisions(x_dot, order)
        x_dot_n_list = get_x_divisions(x_dot_n, order)
        x_interp_list = get_x_divisions(x_interp, order)
        a = x_dot_list[-1]
        a_n = x_dot_n_list[-1]
        x_list.append(a)
        x_list.append((-a + a_n) / h)
        for l in range(1 - deriv, M + 1 - deriv):
            pos = M - l - deriv
            x_interp_list[pos][:] = 0
            for i in range(l + 2):
                x_interp_list[pos][:] = (
                    x_interp_list[pos] + tau**i / q_coefs[i] * x_list[i + M - l]
                )
        return x_interp

    return tz_interp


@lru_cache(maxsize=None)
def generate_hs_interp(order, deriv=0):
    M = order
    q_coefs = [fac(i) for i in range(M + 3)]

    def hs_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
        x_dot_c = scheme_params
        x_interp = copy(x)
        x_list = get_x_divisions(x, order)
        x_dot_list = get_x_divisions(x_dot, order)
        x_dot_n_list = get_x_divisions(x_dot_n, order)
        x_dot_c_list = get_x_divisions(x_dot_c, order)
        x_interp_list = get_x_divisions(x_interp, order)
        a = x_dot_list[-1]
        a_n = x_dot_n_list[-1]
        a_c = x_dot_c_list[-1]
        x_list.append(a)
        x_list.append((-3 * a + 4 * a_c - a_n) / h)
        x_list.append((4 * a - 8 * a_c + 4 * a_n) / (h**2))
        for l in range(1 - deriv, M + 1 - deriv):
            pos = M - l - deriv
            x_interp_list[pos][:] = 0
            for i in range(l + 3):
                x_interp_list[pos][:] = (
                    x_interp_list[pos] + tau**i / q_coefs[i] * x_list[i + M - l]
                )
        return x_interp

    return hs_interp


def hsj_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
    dim = vec_len(x) // 2
    x_dot_c = scheme_params
    f_c = x_dot_c[dim:]
    q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
    # q_c = x_c[:dim]
    # v_c = x_c[dim:]
    q_interp = (
        q
        + v * tau
        + 1 / 2 * f * tau**2
        + 1 / 36 * (-21 * f + 25 * f_c - 4 * f_n) * tau**3 / h
        + 1 / 72 * (15 * f - 25 * f_c + 10 * f_n) * tau**4 / (h**2)
    )
    v_interp = (
        v
        + f * tau
        + 1 / 12 * (-21 * f + 25 * f_c - 4 * f_n) * tau**2 / h
        + 1 / 18 * (15 * f - 25 * f_c + 10 * f_n) * tau**3 / (h**2)
    )
    return concatenate([q_interp, v_interp])


# def hsj_parab_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n, h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         q
#         + v * tau
#         + 1 / 2 * f * tau**2
#         + 1 / 36 * (-21 * f + 25 * f_c - 4 * f_n) * tau**3 / h
#         + 1 / 72 * (15 * f - 25 * f_c + 10 * f_n) * tau**4 / (h**2)
#     )
#     v_interp = (
#         v
#         + f * tau
#         + 1 / 12 * (-21 * f + 25 * f_c - 4 * f_n) * tau**2 / h
#         + 1 / 18 * (15 * f - 25 * f_c + 10 * f_n) * tau**3 / (h**2)
#     )
#     return concatenate([q_interp, v_interp])


def _u_min_err_opti_step(u, x, x_dot, F, params):
    """
    Must be equal to zero in order to fulfill the implicit scheme

    Returns
    -------
    res : Numpy array or Casadi array
        Residue to minimize

    """

    dim = vec_len(x) // 2
    f = F(x, u, params)[dim:]
    a = x_dot[dim:]
    res = sum(abs(f - a))
    return res


def _newpoint_u(U, h, t, u_scheme, scheme_params=None):
    if scheme_params is None:
        scheme_params = {}
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
        if u_scheme == "lin":
            u_interp = (u * (h - tau) + u_n * tau) / h
        elif u_scheme == "parab":
            U_c = scheme_params["u_c"]
            u_c = U_c[n]
            u_interp = interp_parab(tau, h, u, u_c, u_n)
        elif u_scheme == "parab_j":
            U_c = scheme_params["u_c"]
            u_c = U_c[n]
            u_interp = interp_parab_j(tau, h, u, u_c, u_n)

        elif u_scheme == "min_err":
            F = scheme_params["F"]
            X = scheme_params["X"]
            X_dot = scheme_params["X_dot"]
            scheme = scheme_params["scheme"]
            params = scheme_params["params"]
            x_interp = _newpoint(X, X_dot, h, t, params, scheme, scheme_params)
            x_dot_interp = _newpoint_der(X, X_dot, h, t, scheme, 1, scheme_params)
            xi = tau / h
            u_0 = u_n * xi + u * (1 - xi)
            u_interp = minimize(
                _u_min_err_opti_step, u_0, (x_interp, x_dot_interp, F, params)
            )
            u_interp = u_interp.x

        elif u_scheme == "pinv_dyn":
            X = scheme_params["X"]
            X_dot = scheme_params["X_dot"]
            scheme = scheme_params["scheme"]
            params = scheme_params["params"]
            x_interp = _newpoint(X, X_dot, h, t, params, scheme, scheme_params)
            x_dot_interp = _newpoint_der(X, X_dot, h, t, scheme, 1, scheme_params)
            pinv_F = scheme_params["pinv_f"]
            dim = vec_len(x_interp) // 2
            q_interp = x_interp[:dim]
            v_interp = x_interp[dim:]
            a_interp = x_dot_interp[dim:]
            u_interp = pinv_F(q_interp, v_interp, a_interp, params)

        else:
            raise NameError(f"scheme {u_scheme} not recognized")
    return u_interp


def _newpoint(X, X_dot, h, t, params, scheme, scheme_params=None):
    if scheme_params is None:
        scheme_params = {}
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
        x, x_n, x_dot, x_dot_n = X[n], X[n + 1], X_dot[n], X_dot[n + 1]
        if scheme == "trapz_mod":
            f_interp = generate_tz_interp(2, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "trapz":
            f_interp = generate_tz_interp(1, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "trapz_n":
            order = scheme_params["order"]
            f_interp = generate_tz_interp(order, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "hs" or scheme == "hs_parab":
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(1, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme == "hs_mod" or scheme == "hs_mod_parab":
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(2, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme == "hsn" or scheme == "hsn_parab":
            order = scheme_params["order"]
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(order, deriv=0)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme == "hsj" or scheme == "hsj_parab":
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            x_interp = hsj_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)

        else:
            raise NameError(f"scheme {scheme} not recognized")
    return x_interp


def _prepare_interp(X, U, h, t_array):
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


def interpolate_u(U, old_t_array, t_array, u_scheme="lin", scheme_params=None):
    """
    Interpolates values of U using the appropiate form function

    Parameters
    ----------
    U : Numpy Array, shape = (X, [Y])
        Values of U at 'old_t_array'
    old_t_array : Numpy Array, shape = (X,)
        Values of time where U is known.
    t_array : Numpy Array, shape = (Z,)
        Values of time where U is to be interpolated.
    u_scheme : string, optional
        Model of the interpolation that must be used. The default is "lin".
        Acceptable values are:
            "lin": lineal interpolation
            "parab": parabolic interpolation, requires central points array
            as scheme params[0]
            "parab_j": parabolic interpolation, with the intermediate point
            at 2h/5, for work with HSJ. Requires central points array
            as scheme params[0]
            "min_err": for every point in the interpolation, the values of u
            will be calculated to minimize dynamical error |q'' - f(q, q', u)|
            "pinv_dyn": for every point in the interpolation, the pseudoinverse
            dynamics are calculated
    scheme_params : dict or None, optional
        Aditional parameters of the scheme. The default is None.

    Raises
    ------
    ValueError
        If the array U has a shape with a number of dimensions different
        of 1 or 2.

    Returns
    -------
    new_U : Numpy Array, shape = (Z, [Y])
        interpolated values of U.

    """
    if scheme_params is None:
        scheme_params = {}
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


def _calculate_missing_arrays(
    X, U, h, params, F, X_dot, scheme, u_scheme, scheme_params
):
    if X_dot is None:
        if F is None:
            raise ValueError("X_dot and F cannot be None at the same time")
        X_dot = array([list(F(X[ii], U[ii], params)) for ii in range(X.shape[0])])

    if "hs" in scheme and scheme != "hs_scipy":
        if "x_dot_c" not in scheme_params:
            if "x_c" not in scheme_params:
                if "mod" in scheme:
                    X_c = array(
                        [
                            list(
                                hs_mod_midpoint(
                                    X[ii],
                                    X[ii + 1],
                                    X_dot[ii],
                                    X_dot[ii + 1],
                                    h,
                                    params,
                                )
                            )
                            for ii in range(X.shape[0] - 1)
                        ]
                    )
                elif "j" in scheme:
                    X_c = array(
                        [
                            list(
                                hsj_midpoint(
                                    X[ii],
                                    X[ii + 1],
                                    X_dot[ii],
                                    X_dot[ii + 1],
                                    h,
                                    params,
                                )
                            )
                            for ii in range(X.shape[0] - 1)
                        ]
                    )
                else:
                    X_c = array(
                        [
                            list(
                                hs_midpoint(
                                    X[ii],
                                    X[ii + 1],
                                    X_dot[ii],
                                    X_dot[ii + 1],
                                    h,
                                )
                            )
                            for ii in range(X.shape[0] - 1)
                        ]
                    )
            else:
                X_c = scheme_params["x_c"]
            if "parab" not in u_scheme and (
                "u_c" not in scheme_params or scheme_params["u_c"] is None
            ):
                if "j" in scheme:
                    U_c = 0.6 * U[:-1] + 0.4 * U[1:]
                else:
                    U_c = (U[:-1] + U[1:]) / 2
                scheme_params["u_c"] = U_c
            U_c = scheme_params["u_c"]
            # print(X_c, U_c, X_c.shape)
            X_dot_c = array(
                [list(F(X_c[ii], U_c[ii], params)) for ii in range(X_c.shape[0])]
            )
            scheme_params["x_dot_c"] = X_dot_c
    return X_dot


def interpolated_array(
    X,
    U,
    h,
    t_array,
    params,
    F=None,
    X_dot=None,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params=None,
):
    """
    Interpolates values of X and U using the appropiate form functions.
    It is assumed that X and U start at t = 0 and are equispaced in time
    with a dt = h.
    Either F or X_dot must be provided


    Parameters
    ----------
    X : Numpy Array, shape = (W, 2N)
        Known values of X
    U : Numpy Array, shape = (W, [Y])
        Known Values of U
    h : float
        Time step between points
    t_array : Numpy array, shape = (Z,)
        Values of time where X and U are to be interpolated.
    params : list
        Physical problem parameters to be passed to F
    F : Function of (x, u, params), default None
        A function of a dynamic sistem, so that
            x' = F(x, u, params)
        if X_dot is None and F is not, F will be used to calculate X'
    X_dot : Numpy Array, shape = (W, 2N), default = None
        Known values of X'
        if X_dot is None and F is not, F will be used to calculate X'
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
            "hsj":Hermite-Simpson-Jacobi scheme compatible interpolation
            "hsj_parab":Hermite-Simpson-Jacobi scheme compatible interpolation with parabolic U
    u_scheme : string, optional
        Model of the interpolation that must be used. The default is "lin".
        Acceptable values are:
            "lin": lineal interpolation
            "parab": parabolic interpolation, requires central points array
            as scheme params[0]
            "parab_j": parabolic interpolation, with the intermediate point
            at 2h/5, for work with HSJ. Requires central points array
            as scheme params[0]
            "min_err": for every point in the interpolation, the values of u
            will be calculated to minimize dynamical error |q'' - f(q, q', u)|
            Using this method requires that F is not None
            "pinv_dyn": for every point in the interpolation, the pseudoinverse
            dynamics are calculated
    scheme_params :dict or None, optional
        Aditional parameters of the scheme. The default is None.

    Raises
    ------
    ValueError
        When either 'scheme' or 'u_scheme' are given unsupported values

    Returns
    -------
    new_X : Numpy Array, shape = (Z, 2N)
        Interpolated X array
    new_U : Numpy Array, shape = (Z, [Y])
        interpolated values of U.

    """
    supported_schemes = [
        "trapz",
        "trapz_mod",
        "trapz_n",
        "hs",
        "hs_scipy",
        "hs_mod",
        "hs_parab",
        "hs_mod_parab",
        "hsj",
        "hsj_parab",
        "hsn",
        "hsn_parab",
    ]
    if scheme not in supported_schemes:
        raise ValueError(
            f"Unsupported scheme {scheme}, supported schemes are{supported_schemes}"
        )
    supported_u_schemes = [
        "lin",
        "parab",
        "parab_j",
        "min_err",
        "pinv_dyn",
    ]
    if u_scheme not in supported_u_schemes:
        raise ValueError(
            f"Unsupported u_scheme {u_scheme}, supported schemes are{supported_u_schemes}"
        )
    if "j" in scheme and u_scheme == "parab":
        raise ValueError(
            f"Scheme {scheme} incompatible with u_scheme{u_scheme}"
            + ", 'parab_j' must be used with 'hsj'."
        )
    if "j" in u_scheme and "j" not in scheme:
        raise ValueError(
            f"Scheme {scheme} incompatible with u_scheme{u_scheme}"
            + ", 'parab_j' must be used with 'hsj'."
        )
    if "hsn" in scheme or "trapz_n" in scheme:
        if "order" not in scheme_params:
            raise ValueError(
                "If you use an N-order scheme, such as HSN, "
                + "the order must be present as 'order' in the scheme_params"
                + " dictionary"
            )
    if scheme_params is None:
        scheme_params = {}

    N, new_X, U, old_t_array = _prepare_interp(X, U, h, t_array)
    X_dot = _calculate_missing_arrays(
        X, U, h, params, F, X_dot, scheme, u_scheme, scheme_params
    )

    if u_scheme in ["min_err", "pinv_dyn"]:
        scheme_params["X"] = X
        scheme_params["scheme"] = scheme
        scheme_params["params"] = params
        scheme_params["X_dot"] = X_dot
        if u_scheme == "min_err":
            if F is None:
                raise ValueError(
                    "F cannot be None when using min_err as u interpolation"
                )
            scheme_params["F"] = F

    new_U = interpolate_u(U, old_t_array, t_array, u_scheme, scheme_params)
    if scheme == "hs_scipy":
        X_interp = hermite(old_t_array, X, X_dot)
        new_X = X_interp(t_array)
    else:
        for ii in range(N):
            new_X[ii] = array(
                _newpoint(X, X_dot, h, t_array[ii], params, scheme, scheme_params)
            ).flatten()
    return new_X, new_U


# --- Derivatives ---


# def trap_dot_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     x_interp = x_dot + 1 / h * tau * (x_dot_n-x_dot)
#     return x_interp


# def trap_mod_dot_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     dim = vec_len(x) // 2
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     q_interp = v + f * tau + 1 / (2 * h) * tau**2 * (f_n - f)
#     v_interp = f + 1 / h * tau * (f_n - f)
#     return concatenate([q_interp, v_interp])


# def trap_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     x_interp =  1 / h * (x_dot_n-x_dot)
#     return x_interp


# def trap_mod_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h):
#     dim = vec_len(x) // 2
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     q_interp = f + 1 / h * tau * (f_n - f)
#     v_interp = 1 / h * (f_n - f)
#     return concatenate([q_interp, v_interp])


# def hs_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     x_dot_c = scheme_params
#     x_interp = (
#         + x_dot
#         + (-3 * x_dot + 4 * x_dot_c - x_dot_n) * tau/ h
#         + (2 * x_dot - 4 * x_dot_c + 2 * x_dot_n) * tau**2 / (h**2)
#     )
#     return x_interp


# def hs_parab_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, params, scheme_params):
#     x_c = hs_midpoint(x, x_n, x_dot, x_dot_n, h)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     v_c = x_c[dim:]
#     q_interp = (
#         v
#         + tau * (-3 * v + 4 * v_c - v_n) / h
#         + tau**2 * (2 * v - 4 * v_c + 2 * v_n) / h**2
#     )
#     v_interp = (
#         f
#         + tau * (-3 * f + 4 * f_c - f_n) / h
#         + tau**2 * (2 * f - 4 * f_c + 2 * f_n) / h**2
#     )
#     return concatenate([q_interp, v_interp])


# def hs_mod_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         v
#         + f * tau
#         + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau**2 / h
#         + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau**3 / (h**2)
#     )
#     v_interp = (
#         f
#         + tau * (-3 * f + 4 * f_c - f_n) / h
#         + tau**2 * (2 * f - 4 * f_c + 2 * f_n) / h**2
#     )
#     return concatenate([q_interp, v_interp])


def hsj_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
    # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
    dim = vec_len(x) // 2
    x_dot_c = scheme_params
    f_c = x_dot_c[dim:]
    q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
    # q_c = x_c[:dim]
    # v_c = x_c[dim:]
    q_interp = (
        v
        + f * tau
        + 1 / 12 * (-21 * f + 25 * f_c - 4 * f_n) * tau**2 / h
        + 1 / 18 * (15 * f - 25 * f_c + 10 * f_n) * tau**3 / (h**2)
    )
    v_interp = (
        f
        + 1 / 6 * (-21 * f + 25 * f_c - 4 * f_n) * tau / h
        + 1 / 6 * (15 * f - 25 * f_c + 10 * f_n) * tau**2 / (h**2)
    )
    return concatenate([q_interp, v_interp])


# def hs_mod_parab_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, params, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         v
#         + f * tau
#         + 1 / 2 * (-3 * f + 4 * f_c - f_n) * tau**2 / h
#         + 1 / 3 * (2 * f - 4 * f_c + 2 * f_n) * tau**3 / (h**2)
#     )
#     v_interp = (
#         f
#         + tau * (-3 * f + 4 * f_c - f_n) / h
#         + tau**2 * (2 * f - 4 * f_c + 2 * f_n) / h**2
#     )
#     return concatenate([q_interp, v_interp])


# def hs_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     x_dot_c = scheme_params
#     x_interp = (
#         + (-3 * x_dot + 4 * x_dot_c - x_dot_n) / h
#         + 2*(2 * x_dot - 4 * x_dot_c + 2 * x_dot_n) * tau / (h**2)
#     )
#     return x_interp


# def hs_mod_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         f
#         + tau * (-3 * f + 4 * f_c - f_n) / h
#         + tau**2 * (2 * f - 4 * f_c + 2 * f_n) / h**2
#     )
#     v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
#         2 * f - 4 * f_c + 2 * f_n
#     ) / h**2
#     return concatenate([q_interp, v_interp])


# def hs_parab_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, params, scheme_params):
#     x_c = hs_midpoint(x, x_n, x_dot, x_dot_n, h)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     v_c = x_c[dim:]
#     q_interp = (-3 * v + 4 * v_c - v_n) / h + 2 * tau * (
#         2 * v - 4 * v_c + 2 * v_n
#     ) / h**2
#     v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
#         2 * f - 4 * f_c + 2 * f_n
#     ) / h**2
#     return concatenate([q_interp, v_interp])


# def hs_mod_parab_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, params, scheme_params):
#     # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
#     dim = vec_len(x) // 2
#     x_dot_c = scheme_params
#     f_c = x_dot_c[dim:]
#     q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
#     # q_c = x_c[:dim]
#     # v_c = x_c[dim:]
#     q_interp = (
#         f
#         + tau * (-3 * f + 4 * f_c - f_n) / h
#         + tau**2 * (2 * f - 4 * f_c + 2 * f_n) / h**2
#     )
#     v_interp = (-3 * f + 4 * f_c - f_n) / h + 2 * tau * (
#         2 * f - 4 * f_c + 2 * f_n
#     ) / h**2
#     return concatenate([q_interp, v_interp])


def hsj_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, scheme_params):
    # x_c = hs_mod_midpoint(x, x_n, x_dot, x_dot_n,h, params)
    dim = vec_len(x) // 2
    x_dot_c = scheme_params
    f_c = x_dot_c[dim:]
    q, q_n, v, v_n, f, f_n = _gen_basic_values(dim, x, x_n, x_dot, x_dot_n)
    # q_c = x_c[:dim]
    # v_c = x_c[dim:]
    q_interp = (
        f
        + 1 / 6 * (-21 * f + 25 * f_c - 4 * f_n) * tau / h
        + 1 / 6 * (15 * f - 25 * f_c + 10 * f_n) * tau**2 / (h**2)
    )
    v_interp = 1 / 6 * (-21 * f + 25 * f_c - 4 * f_n) / h + 1 / 3 * (
        15 * f - 25 * f_c + 10 * f_n
    ) * tau / (h**2)
    return concatenate([q_interp, v_interp])


def _newpoint_der(X, X_dot, h, t, scheme, order, scheme_params=None):
    if scheme_params is None:
        scheme_params = {}
    # Avoid out of interpolation error when t == t_final
    if abs(t - h * (X.shape[0] - 1)) < h * 1e-8:
        n = X.shape[0] - 2
        tau = h
    else:
        n = int(t // h)
        tau = t % h
    # if abs(tau) < h * 1e-8:
    #    x_interp = X[n]
    # elif abs(tau - h) < h * 1e-8:
    #    x_interp = X[n + 1]
    if (n + 1) >= X.shape[0]:
        raise ValueError(f"Value of time {t} detected outside interpolation limits")
    else:
        x, x_n, x_dot, x_dot_n = X[n], X[n + 1], X_dot[n], X_dot[n + 1]
        if scheme == "trapz":
            f_interp = generate_tz_interp(1, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "trapz_mod":
            f_interp = generate_tz_interp(2, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "trapz_n":
            order_sch = scheme_params["order"]
            f_interp = generate_tz_interp(order_sch, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h)
        elif scheme == "hs" or scheme == "hs_parab":
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(1, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme == "hs_mod" or scheme == "hs_mod_parab":
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(2, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme == "hsn" or scheme == "hsn_parab":
            order_sch = scheme_params["order"]
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            f_interp = generate_hs_interp(order_sch, deriv=order)
            x_interp = f_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        elif scheme in ["hsj", "hsj_parab"]:
            X_dot_c = scheme_params["x_dot_c"]
            x_dot_c = X_dot_c[n]
            if order == 1:
                x_interp = hsj_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
            elif order == 2:
                x_interp = hsj_dot_dot_interp(x, x_n, x_dot, x_dot_n, tau, h, x_dot_c)
        else:
            raise NameError(f"scheme {scheme} not recognized")
    return x_interp


def interpolated_array_derivative(
    X,
    U,
    h,
    t_array,
    params,
    F=None,
    X_dot=None,
    scheme="hs_scipy",
    order=1,
    scheme_params=None,
):
    """
    Calculates the n-th order derivatives of an interpolation of X using
    the appropiate form functions and returns its values at 't_array'.
    It is assumed that X and U start at t = 0 and are equispaced in time
    with a dt = h.
    Either F or X_dot must be provided



    Parameters
    ----------
    X : Numpy Array, shape = (W, 2N)
        Known values of X
    U : Numpy Array, shape = (W, [Y])
        Known Values of U
    h : float
        Time step between points
    t_array : Numpy array, shape = (Z,)
        Values of time where X and U are to be interpolated.
    params : list
        Physical problem parameters to be passed to F
    F : Function of (x, u, params), default None
        A function of a dynamic sistem, so that
            x' = F(x, u, params)
        if X_dot is None and F is not, F will be used to calculate X'
    X_dot : Numpy Array, shape = (W, 2N), default = None
        Known values of X'
        if X_dot is None and F is not, F will be used to calculate X'
    scheme : str, optional
        Scheme to be used in the X interpolation. The default is "hs_scipy".
        Acceptable values are:
            "hs_scipy": 3d order polynomial that satisfies continuity in x(t) and x'(t)
            "hs": Hermite-Simpson scheme compatible interpolation
            "hs_mod": modified Hermite-Simpson scheme compatible interpolation
            "hs_parab": Hermite-Simpson scheme compatible interpolation with parabolic U
            "hs_mod_parab": modified Hermite-Simpson scheme compatible interpolation with parabolic U
            "hsj":Hermite-Simpson-Jacobi scheme compatible interpolation
            "hsj_parab":Hermite-Simpson-Jacobi scheme compatible interpolation with parabolic U
            "trapz": Trapezoidal scheme
            "trapz_mod": modified trapezoidal scheme
    order : int, optional
        Derivation order. The default is 1. For HSJ and other prototype schemes,
        acceptable values may be 1 and 2.
    scheme_params :dict or None, optional
        Aditional parameters of the scheme. The default is None.

    Raises
    ------
    ValueError
        When either 'scheme' or 'u_scheme' are given unsupported values

    Returns
    -------
    new_X : Numpy Array, shape = (Z, 2N)
        Interpolated X derivative array

    """
    # supported_order = [1, 2]
    # if order not in supported_order:
    #     raise ValueError(
    #         f"Unsupported derivation order, supported order are{supported_order}"
    #     )
    supported_schemes = [
        "hs",
        "hs_scipy",
        "hs_mod",
        "hs_parab",
        "hs_mod_parab",
        "hsn",
        "hsn_parab",
        "hsj",
        "hsj_parab",
        "trapz",
        "trapz_mod",
        "trapz_n",
    ]
    if scheme not in supported_schemes:
        raise ValueError(
            f"Unsupported scheme, supported schemes are{supported_schemes}"
        )
    if scheme_params is None:
        scheme_params = {}

    N, new_X, U, old_t_array = _prepare_interp(X, U, h, t_array)

    u_scheme = "None"  # It is required in the next function but won't be used.
    X_dot = _calculate_missing_arrays(
        X, U, h, params, F, X_dot, scheme, u_scheme, scheme_params
    )

    if scheme == "hs_scipy":
        X_interp = hermite(old_t_array, X, X_dot)
        X_dot_interp = X_interp.derivative()
        if order == 1:
            new_X = X_dot_interp(t_array)
        elif order == 2:
            new_X = X_dot_interp.derivative()(t_array)
    else:
        for ii in range(N):
            new_X[ii] = array(
                _newpoint_der(X, X_dot, h, t_array[ii], scheme, order, scheme_params)
            ).flatten()
    return new_X
