# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:08:53 2022

@author: Siro Moreno

Here we will define functions dedicated to analysis and post-processing
of obtained solutions.
"""

from .piecewise import (
    interpolated_array,
    interpolated_array_derivative,
    _newpoint,
    interp_2d,
    _newpoint_der,
    _newpoint_u,
    _calculate_missing_arrays,
    get_x_divisions,
    expand_F,
)
from .opti import (
    _implemented_bottom_up_pseudospectral_schemes,
    _implemented_equispaced_schemes,
    _implemented_pseudospectral_schemes,
    _implemented_top_down_pseudospectral_schemes,
    get_q_and_ders_names,
)
from .pseudospectral import interpolations_pseudospectral
from .bu_pseudospectral import (
    interpolations_BU_pseudospectral,
    interpolations_deriv_BU_pseudospectral,
)
from .td_pseudospectral import (
    interpolations_TD_pseudospectral,
    interpolations_deriv_TD_pseudospectral,
)

from scipy.optimize import root, minimize
from scipy.integrate import quad
from functools import lru_cache
from numpy import (
    zeros,
    zeros_like,
    append,
    concatenate,
    linspace,
    expand_dims,
    interp,
    array,
    sqrt,
    trapz,
    mean,
)

from numpy import sum as npsum
from numpy import abs as npabs
from numpy import max as npmax

from numpy.linalg import inv, solve
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
    scheme_params=None,
    n_interp=2000,
    order=2,
    mode="q",
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
    scheme_params :dict or none, optional
        Aditional parameters of the scheme. The default is None.
    n_interp : int, optional
        Number of interpolation points. The default is 2000.
    order : int, default 2
        differential order of the problem
    mode : str, 'q' or 'x', default 'q'.
        if 'q': q and its derivatives will be used in G, such as:
            G(q(t), q'(t), u(t))
        if 'x': components of x will be used in G, such as:
            G(q(t), v(t), u(t))

    Returns
    -------
    dyn_errs: list of lists of arrs that contain the dynamic errors
        A list of [order] items, the n-th item is a list of the n-th dynamic errors.
        First item is the list (first order errors):
            [q' - v,
             v' - a,
             ...
             x'[last] - f,]
        last item in the list (highest order errors):
            [q^(order) - f,]

    """
    if "parab" in scheme and u_scheme == "lin":
        warnings.warn(
            "You are currently using a u-parabolic interpolation for x with a lineal interpolation of u"
        )
    if "parab" in u_scheme and "parab" not in scheme:
        warnings.warn(
            "You are currently using a parabolic interpolation for u with a non u-parabolic interpolation of x"
        )
    if ("j" in u_scheme and "j" not in scheme) or (
        "j" not in u_scheme and "j" in scheme
    ):
        warnings.warn(
            f"Scheme {scheme} incompatible with u_scheme{u_scheme}"
            + ", 'parab_j' must be used with 'hsj'."
        )
    if scheme_params is None:
        scheme_params = {}
    N = x_arr.shape[0] - 1
    dim = x_arr.shape[1] // order
    h = t_end / N
    t_interp = linspace(0, t_end, n_interp)
    x_and_derivs = []
    scheme_params["order"] = order
    x_interp, u_interp = interpolated_array(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        X_dot=X_dot,
        F=F,
        scheme=scheme,
        u_scheme=u_scheme,
        scheme_params=scheme_params,
    )

    x_and_derivs.append(get_x_divisions(x_interp, order))
    for jj in range(1, order + 1):
        x_and_derivs.append(
            get_x_divisions(
                interpolated_array_derivative(
                    x_arr,
                    u_arr,
                    h,
                    t_interp,
                    params,
                    F=F,
                    X_dot=X_dot,
                    scheme=scheme,
                    order=jj,
                    scheme_params=scheme_params,
                ),
                order,
            )
        )

    q_and_d_interp = copy(x_interp)
    for jj in range(order):
        q_and_d_interp[:, dim * jj : dim * (jj + 1)] = x_and_derivs[jj][0]

    if mode == "q":
        x_in_f = q_and_d_interp
    elif mode == "x":
        x_in_f = x_interp
    else:
        raise ValueError(
            f"Value of mode {mode} not valid. Valid values are 'q' and 'x'."
        )

    f_interp = zeros([n_interp, dim])
    for ii in range(n_interp):
        f_interp[ii, :] = F(x_in_f[ii], u_interp[ii], params)[-dim:]
    x_and_derivs[0].append(f_interp)

    dyn_errs = []
    for jj in range(order):
        dyn_errs_order = []
        for ii in range(order - jj):
            dyn_errs_order.append(
                x_and_derivs[jj + 1][ii] - x_and_derivs[0][ii + jj + 1]
            )
        dyn_errs.append(dyn_errs_order)
    return dyn_errs


def generate_G(M, F_impl, order=2):
    """
    Generate a function G from M and F, so that from

            | q''  |   |                | -1   |                 |
            |      | = |  M(x, params)  |    @ | F(x, u, params) |
            |lambda|   |                |      |                 |,

    we can get a function G so that:

         q'' = G(x, u) = (M(x)^-1 @  F(x, u)) [upperside]

    Parameters
    ----------
    M : Function of (x, params)
        Returns a Numerical Matrix.
    F : Function of (x, u, params)
        Returns a Numerical Vector.
    order: int, default 2
        differential order of the problem

    Returns
    -------
    G : Function of (x, u, params)
        Equal to q'' where the collocation constraint is enforced.

    """

    def G(x, u, params):
        dim = x.shape[-1] // order
        new_shape = list(x.shape)
        new_shape[-1] = dim
        mm = M(x, params)
        ff = F_impl(x, u, params)
        res = solve(mm, ff).reshape(new_shape)
        return res[:dim]

    return G


def interpolation(
    res,
    problem_order=2,
    scheme_order=2,
    x_interp=None,
    u_interp=None,
    n_interp=1000,
):
    scheme = res["scheme"]
    mode = res["scheme_mode"]
    interp_order = res["solve_order"]
    params = res["params"]

    xx = res["x"]
    xx_d = res["x_d"]
    uu = res["u"]
    tt = res["t"]
    t0 = tt[0]
    tf = tt[-1]

    q_and_d_names = get_q_and_ders_names(problem_order)
    n_x = xx.shape[-1]
    # n_q = n_x // problem_order
    assert n_x % problem_order == 0

    if mode == "equispaced":
        t_arr = linspace(t0, tf, n_interp)
        h = (tf - t0) / (tt.shape[0] - 1)
        if u_interp is None:
            if "parab" in scheme:
                if "j" in scheme:
                    u_interp = "parab_j"
                else:
                    u_interp = "parab"
            else:
                u_interp = "lin"

        if "hs" in scheme:
            scheme_params = {
                "u_c": res["u_c"],
                "x_dot_c": res["x_d_c"],
                "x_c": res["x_c"],
            }
        else:
            scheme_params = {}
        if "n" in scheme:
            scheme_params["order"] = interp_order

        x_arr, u_arr = interpolated_array(
            X=xx,
            U=uu,
            h=h,
            t_array=t_arr,
            params=params,
            F=None,
            X_dot=xx_d,
            scheme=scheme,
            u_scheme=u_interp,
            scheme_params=scheme_params,
        )
        interpolations = {"x": x_arr, "u": u_arr}
        for ii in range(1, problem_order + 1):
            _arr = interpolated_array_derivative(
                X=xx,
                U=uu,
                h=h,
                t_array=t_arr,
                params=params,
                F=None,
                X_dot=xx_d,
                scheme=scheme,
                order=ii,
                scheme_params=scheme_params,
            )
            interpolations["x" + "_d" * ii] = _arr

    elif mode == "pseudospectral":
        if u_interp is None:
            u_interp = "pol"
        if x_interp is None:
            x_interp = "pol"
        qq = res["q_node"]
        vv = res["v_node"]
        (
            q_arr,
            q_arr_d,
            v_arr,
            v_arr_d,
            q_arr_d_d,
            u_arr,
        ) = interpolations_pseudospectral(
            qq,
            vv,
            uu,
            scheme,
            t0,
            tf,
            u_interp=u_interp,
            x_interp=x_interp,
            params=params,
            n_interp=n_interp,
        )

        if interp_order == 2:
            x_arr = concatenate((q_arr, v_arr), axis=1)
            x_dot_arr = concatenate((q_arr_d, v_arr_d), axis=1)
        elif interp_order == 1:
            x_arr = q_arr
            x_dot_arr = v_arr
            _arr_d_d = get_x_divisions(q_arr_d_d, problem_order)
            q_arr_d_d = _arr_d_d[0]
        else:
            raise NotImplementedError(
                "Pseudospectral schemes not yet prepared for higher than 2 order"
            )

        interpolations = {"x": x_arr, "x_d": x_dot_arr, "u": u_arr}

    elif mode == "bottom-up pseudospectral":
        if u_interp is None:
            u_interp = "pol"
        if x_interp is None:
            x_interp = "pol"

        x_arr, x_dot_arr, u_arr = interpolations_BU_pseudospectral(
            xx,
            xx_d,
            uu,
            scheme,
            interp_order,
            t0,
            tf,
            scheme_order=scheme_order,
            u_interp=u_interp,
            x_interp=x_interp,
            n_interp=n_interp,
        )
        interpolations = {"x": x_arr, "x_d": x_dot_arr, "u": u_arr}
        for ii in range(2, problem_order + 1):
            _arr = interpolations_deriv_BU_pseudospectral(
                xx,
                xx_d,
                scheme,
                interp_order,
                ii,
                t0,
                tf,
                scheme_order=scheme_order,
                x_interp=x_interp,
                n_interp=n_interp,
            )
            interpolations["x" + "_d" * ii] = _arr

    elif mode == "top-down pseudospectral":
        if u_interp is None:
            u_interp = "pol"
        if x_interp is None:
            x_interp = "pol"

        n_coll = uu.shape[0]

        q_constr = res["q_constr"]
        x_arr, x_dot_arr, u_arr = interpolations_TD_pseudospectral(
            q_constr,
            xx,
            xx_d,
            uu,
            scheme,
            t0,
            tf,
            scheme_order=scheme_order,
            u_interp=u_interp,
            x_interp=x_interp,
            n_interp=n_interp,
        )
        interpolations = {"x": x_arr, "x_d": x_dot_arr, "u": u_arr}
        for ii in range(2, problem_order + 1):
            _arr = interpolations_deriv_TD_pseudospectral(
                q_constr,
                xx,
                xx_d,
                scheme,
                ii,
                t0,
                tf,
                n_coll,
                scheme_order=scheme_order,
                x_interp=x_interp,
                n_interp=n_interp,
            )
            interpolations["x" + "_d" * ii] = _arr

    else:
        raise ValueError(f"Unrecognized mode {mode}")

    # distributing q and derivatives arrays
    if mode == "pseudospectral":
        if problem_order == 1:
            interpolations["q"] = x_arr
            interpolations["q_d"] = x_dot_arr
            interpolations["v"] = x_dot_arr
        elif problem_order == 2:
            q_arr, v_arr = get_x_divisions(x_arr, problem_order)
            q_arr_d, v_arr_d = get_x_divisions(x_dot_arr, problem_order)
            interpolations["q"] = q_arr
            interpolations["q_d"] = q_arr_d
            interpolations["q_d_d"] = q_arr_d_d
            interpolations["v"] = v_arr
            interpolations["v_d"] = v_arr_d
            interpolations["a"] = v_arr_d
        else:
            raise NotImplementedError(
                "pseudospectral schemes not prepared for higher order interpolations"
            )
    else:
        for ii in range(problem_order + 1):
            _arr = interpolations["x" + "_d" * ii]
            _arr_divs = get_x_divisions(_arr, problem_order)
            for jj in range(problem_order - max([1, ii]) + 1):
                arr_name = q_and_d_names[jj] + "_d" * ii
                interpolations[arr_name] = _arr_divs[jj]
        interpolations[q_and_d_names[-1]] = interpolations[q_and_d_names[-2] + "_d"]

    res["interpolations"] = interpolations

    return interpolations


def dynamic_errors(
    res,
    F,
    dynamics_error_mode="q",
    problem_order=2,
    scheme_order=2,
    x_interp=None,
    u_interp=None,
    n_interp=1000,
):

    q_and_d_names = get_q_and_ders_names(problem_order)

    interpolations = interpolation(
        res,
        problem_order,
        scheme_order,
        x_interp,
        u_interp,
        n_interp,
    )

    q_and_d_interp = [interpolations["q" + ii * "_d"] for ii in range(problem_order)]
    q_and_d_interp = concatenate(q_and_d_interp, 1)
    if dynamics_error_mode == "q":
        x_in_f = q_and_d_interp
    elif dynamics_error_mode == "x":
        x_in_f = interpolations["x"]
    else:
        raise ValueError(
            f"Value of dynamics_error_mode {dynamics_error_mode} not valid. Valid values are 'q' and 'x'."
        )

    params = res["params"]
    u_arr = interpolations["u"]
    n_x = interpolations["x"].shape[-1]
    n_q = n_x // problem_order
    assert n_x % problem_order == 0

    f_interp = zeros([n_interp, n_q])
    for ii in range(n_interp):
        f_interp[ii, :] = F(x_in_f[ii], u_arr[ii], params)[-n_q:]
    interpolations["f"] = f_interp
    q_highest_d = interpolations["q" + "_d" * (problem_order)]

    # Errors as defined on section 6.3 of Collocation methods for second
    # and higher order systems
    # https://link.springer.com/article/10.1007/s10514-023-10155-z

    errors = {"dyn_err_interp": q_highest_d - f_interp}

    for jj in range(problem_order - 1):
        err_name = f"compat_err_{jj+1}_interp"
        arr1 = interpolations["q" + "_d" * (jj + 1)]
        arr2 = interpolations[q_and_d_names[jj + 1]]
        errors[err_name] = arr1 - arr2

    res["error"] = errors
    return errors


def dynamic_error_implicit(
    x_arr,
    u_arr,
    t_end,
    params,
    F_impl,
    M,
    X_dot=None,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params=None,
    n_interp=2000,
    order=2,
    mode="q",
):
    """
    Generate arrays of equispaced points with values of dynamic error.

    If x(t) = [q(t), v(t)], and the dynamics can be written as:
            | M    A_c|   | q''  |   |f_d (q, v, u)|
            |         | @ |      | = |             |
            |m_cd   0 |   |lambda|   |f_dc(q, v)   |,

    and therefore:
            | q''  |   | M    A_c| -1   |f_d (q, v, u)|
            |      | = |         |    @ |             |
            |lambda|   |m_cd   0 |      |f_dc(q, v)   |,

    Calling M_comp to the whole inversed matrix, the first equation is:

        q''= (M_comp ^-1 @ [f_d  f_dc]^T) [upperside]

    We define F(q, v, u) = [f_d(q, v, u)  f_dc(q, v)]^T

    we can define G(q, v, u) = (M_comp(q, v) ^-1 @  F(q, v, u)) [upperside]
    For notation simplicity, we will from now on extend the notation of
    M to encompass the whole M_comp, resulting in:

        q'' = G(q, v, u) = (M(q, v)^-1 @  F(q, v, u)) [upperside]

    so that the equation plus the definition of v(t) as q'(t)
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
    F : Function of (x, u,  params)
        A function of a dynamic sistem, so that
            G(q, v, u) = M^-1 @ F(x, u)
    M : Function of (x, params)
        Calculates the numerical value of the complete mass matrix at a given configuration
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
    scheme_params :dict or None, optional
        Aditional parameters of the scheme. The default is None.
    n_interp : int, optional
        Number of interpolation points. The default is 2000.
    order : int, default 2
        differential order of the problem
    mode : str, 'q' or 'x', default 'q'.
        if 'q': q and its derivatives will be used in G, such as:
            G(q(t), q'(t), u(t))
        if 'x': components of x will be used in G, such as:
            G(q(t), v(t), u(t))

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
    if "parab" in scheme and "parab" not in u_scheme:
        warnings.warn(
            "You are currently using a u-parabolic interpolation for x with a lineal interpolation of u"
        )
    if "parab" in u_scheme and "parab" not in scheme:
        warnings.warn(
            "You are currently using a parabolic interpolation for u with a non u-parabolic interpolation of x"
        )
    if ("j" in u_scheme and "j" not in scheme) or (
        "j" not in u_scheme and "j" in scheme
    ):
        warnings.warn(
            f"Scheme {scheme} incompatible with u_scheme{u_scheme}"
            + ", 'parab_j' must be used with 'hsj'."
        )
    if scheme_params is None:
        scheme_params = {}

    N = x_arr.shape[0] - 1
    dim = x_arr.shape[1] // order
    h = t_end / N
    t_interp = linspace(0, t_end, n_interp)
    G = generate_G(M, F_impl)
    F = expand_F(G, mode="numpy", order=order)

    x_and_derivs = []
    scheme_params["order"] = order
    x_interp, u_interp = interpolated_array(
        x_arr,
        u_arr,
        h,
        t_interp,
        params,
        X_dot=X_dot,
        F=F,
        scheme=scheme,
        u_scheme=u_scheme,
        scheme_params=scheme_params,
    )

    x_and_derivs.append(get_x_divisions(x_interp, order))
    for jj in range(1, order + 1):
        x_and_derivs.append(
            get_x_divisions(
                interpolated_array_derivative(
                    x_arr,
                    u_arr,
                    h,
                    t_interp,
                    params,
                    F=F,
                    X_dot=X_dot,
                    scheme=scheme,
                    order=jj,
                    scheme_params=scheme_params,
                ),
                order,
            )
        )

    q_and_d_interp = copy(x_interp)
    for jj in range(order):
        q_and_d_interp[:, dim * jj : dim * (jj + 1)] = x_and_derivs[jj][0]

    if mode == "q":
        x_in_f = q_and_d_interp
    elif mode == "x":
        x_in_f = x_interp
    else:
        raise ValueError(
            f"Value of mode {mode} not valid. Valid values are 'q' and 'x'."
        )

    f_interp = zeros([n_interp, dim])
    for ii in range(n_interp):
        f_interp[ii, :] = F(x_in_f[ii], u_interp[ii], params)[-dim:]
    x_and_derivs[0].append(f_interp)

    dyn_errs = []
    for jj in range(order):
        dyn_errs_order = []
        for ii in range(order - jj):
            dyn_errs_order.append(
                x_and_derivs[jj + 1][ii] - x_and_derivs[0][ii + jj + 1]
            )
        dyn_errs.append(dyn_errs_order)
    return dyn_errs


def arr_mod(x):
    x_1 = npsum(x * x, axis=1)
    return sqrt(x_1)


def arr_sum(x):
    return npsum(npabs(x), axis=1)


def arr_max(x):
    return npmax(npabs(x), axis=1)


def arr_abs_integr_vert(t_arr, x):
    errors = trapz(npabs(x), t_arr, axis=0)
    return errors


def F_point(
    x_arr,
    u_arr,
    t_arr,
    t,
    params,
    F,
    x_dot_arr=None,
    scheme="hs_scipy",
    u_scheme="lin",
    scheme_params=None,
):
    """
    Calculate the value of F(X_q(t), u(t), params), interpolating X_q(t) and u(t)
    first.

    If X(t) = [q(t), v(t)], we define X_q(t) as [q(t), q'(t)]

    'scheme' and 'u_scheme' define the way in which we interpolate the values
    of q, v and u between the given points.




    Parameters
    ----------
    x_arr : Numpy Array, shape = (W, 2N)
        Values known of x(t)
    u_arr : Numpy Array, shape = (W, [Y])
        Values known of u(t)
    t_arr : Numpy Array, shape = (W)
        Values known of t
    t : float
        Time where we want to calculate F(X_q(t), u(t))
    params : list
        Physical problem parameters to be passed to F
    F : Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)
        if x_dot_arr is None, F will be used to calculate X'
    x_dot_arr : Numpy Array, optional, shape = (W, 2N), default = None
        Known values of X'
        if x_dot_arr is None, F will be used to calculate X'
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
    scheme_params :dict or None, optional
        Aditional parameters of the scheme. The default is None.

    Returns
    -------
    F(X_q(t), u(t), params)
        Format and structure will depend on the results of given F
        Usually, it will be a NumPy Array of shape (1, N)

    """
    if "parab" in scheme and u_scheme == "lin":
        warnings.warn(
            "You are currently using a u-parabolic interpolation for x with a lineal interpolation of u"
        )
    if "parab" in u_scheme and "parab" not in scheme:
        warnings.warn(
            "You are currently using a parabolic interpolation for u with a non u-parabolic interpolation of x"
        )
    if ("j" in u_scheme and "j" not in scheme) or (
        "j" not in u_scheme and "j" in scheme
    ):
        warnings.warn(
            f"Scheme {scheme} incompatible with u_scheme{u_scheme}"
            + ", 'parab_j' must be used with 'hsj'."
        )
    if scheme_params is None:
        scheme_params = {}
    N = x_arr.shape[0] - 1
    dim = x_arr.shape[1] // 2
    h = (t_arr[-1] - t_arr[0]) / N

    x_dot_arr = _calculate_missing_arrays(
        x_arr, u_arr, h, params, F, x_dot_arr, scheme, u_scheme, scheme_params
    )
    if u_scheme in ["min_err", "pinv_dyn"]:
        scheme_params["X"] = x_arr
        scheme_params["scheme"] = scheme
        scheme_params["params"] = params
        scheme_params["x_dot_arr"] = x_dot_arr
        if u_scheme == "min_err":
            if F is None:
                raise ValueError(
                    "F cannot be None when using min_err as u interpolation"
                )
            scheme_params["F"] = F

    if scheme == "hs_scipy":
        X_interp = hermite(t_arr, x_arr, x_dot_arr)
        x = X_interp(t)
    else:
        x = array(
            _newpoint(x_arr, x_dot_arr, h, t, params, scheme, scheme_params)
        ).flatten()

    # if u_scheme == "lin":
    #     if len(u_arr.shape) == 1:
    #         u = interp(t, t_arr, u_arr)
    #     elif len(u_arr.shape) == 2:
    #         u = interp_2d(t, t_arr, u_arr)
    #     else:
    #         raise ValueError(
    #             f"U has {len(u_arr.shape)} dimensions, values accepted are 1 and 2"
    #         )
    # else:
    u = array(_newpoint_u(u_arr, h, t, u_scheme, scheme_params)).flatten()

    if scheme == "hs_scipy":
        X_interp = hermite(t_arr, x_arr, x_dot_arr)
        X_dot_interp = X_interp.derivative()
        x_d = X_dot_interp(t)
    else:
        x_d = array(
            _newpoint_der(x_arr, x_dot_arr, h, t, params, scheme, 1, scheme_params)
        ).flatten()

    x_q = x.copy()
    x_q[dim:] = x_d[:dim]
    f_b = F(x_q, u, params)[dim:]
    return f_b


def quad_problem(
    x_arr,
    u_arr,
    t_arr,
    params,
    F,
    scheme,
    u_scheme,
    scheme_params,
    x_dot_arr=None,
    discont_at_t_arr=True,
    sub_div_limit=250,
):
    dim = x_arr.shape[-1] // 2
    h = (t_arr[-1] - t_arr[0]) / (t_arr.shape[0] - 1)
    x_dot_arr = _calculate_missing_arrays(
        x_arr, u_arr, h, params, F, x_dot_arr, scheme, u_scheme, scheme_params
    )

    @lru_cache(maxsize=None)
    def dummy_F(t):
        E = F_point(
            x_arr,
            u_arr,
            t_arr,
            t,
            params,
            F=F,
            scheme=scheme,
            u_scheme=u_scheme,
            scheme_params=scheme_params,
        )
        x_d = array(
            _newpoint_der(x_arr, x_dot_arr, h, t, params, scheme, 1, scheme_params)
        ).flatten()
        return npabs(E - x_d[dim:])

    errors = []

    for ii in range(dim):

        def dummy_prob(t):
            return dummy_F(t)[ii]

        if discont_at_t_arr:
            E, E_e = quad(
                dummy_prob, t_arr[0], t_arr[-1], limit=sub_div_limit, points=t_arr[1:-1]
            )
        else:
            E, E_e = quad(dummy_prob, t_arr[0], t_arr[-1], limit=sub_div_limit)

        # E = x_arr[-1, dim + ii] - x_arr[0, dim + ii] - E
        # Integral of (a-F) == delta v -integral(F)

        errors.append(E)

    return array(errors)


def doub_quad_problem(
    x_arr,
    u_arr,
    t_arr,
    params,
    F,
    scheme,
    u_scheme,
    scheme_params,
    discont_at_t_arr=True,
    sub_div_limit=250,
):
    dim = x_arr.shape[-1] // 2

    @lru_cache(maxsize=None)
    def dummy_F(t):
        E = F_point(
            x_arr,
            u_arr,
            t_arr,
            t,
            params,
            F=F,
            scheme=scheme,
            u_scheme=u_scheme,
            scheme_params=scheme_params,
        )
        return E

    errors = []

    for ii in range(dim):

        def dummy_prob(t):
            return dummy_F(t)[ii]

        if discont_at_t_arr:

            def dummy_integ(t):
                E, E_e = quad(
                    dummy_prob, t_arr[0], t, limit=sub_div_limit, points=t_arr[1:-1]
                )
                return E

        else:

            def dummy_integ(t):
                E, E_e = quad(dummy_prob, t_arr[0], t, limit=sub_div_limit)
                return E

        if discont_at_t_arr:
            E, E_e = quad(
                dummy_integ,
                t_arr[0],
                t_arr[-1],
                limit=sub_div_limit,
                points=t_arr[1:-1],
            )
        else:
            E, E_e = quad(dummy_integ, t_arr[0], t_arr[-1], limit=sub_div_limit)

        E = x_arr[-1, ii] - x_arr[0, ii] - x_arr[0, dim + ii] * t_arr[-1] - E

        errors.append(E)

    return array(errors)
