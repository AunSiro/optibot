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
from numpy import (
    array,
    linspace,
    interp,
    gradient,
    concatenate,
    zeros_like,
    zeros,
    expand_dims,
)
from numpy.linalg import matrix_power
from .bu_pseudospectral import (
    BU_coll_points,
    _gauss_like_schemes,
    _radau_like_schemes,
    _radau_inv_schemes,
    _lobato_like_schemes,
    _other_schemes,
    _implemented_schemes,
    tau_to_t_points,
)
from functools import lru_cache
from .pseudospectral import (
    matrix_D_bary,
    bary_poly_2d,
    extend_u_array,
    LGL,
    find_der_polyline,
)
from .piecewise import interp_2d, get_x_divisions, is2d, force2d
from copy import copy


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
        return [-1.0] + coll_p + [1.0]
    elif scheme in _radau_like_schemes:
        return coll_p + [1.0]
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
    return matrix_D_bary(size, "LGL")


@lru_cache(maxsize=2000)
@store_results
def matrix_L(N, scheme, order=2, precission=16):
    constr_points = TD_construction_points(N, scheme, order, precission)
    constr_points = array(constr_points, dtype="float64")
    l = N + order
    polynomial = Lag_pol_2d(l, "LGL")
    return polynomial(constr_points)


def Polynomial_interpolations_TD(
    q_constr,
    uu,
    t0,
    tf,
    n_coll,
    scheme,
    scheme_order=2,
):
    if scheme[:3] == "TD_":
        scheme = scheme[3:]
    N = q_constr.shape[0]
    interp_order = N - n_coll
    h = tf - t0
    # assert N == order + n_coll

    coll_points = tau_to_t_points(BU_coll_points(n_coll, scheme, scheme_order), t0, tf)
    LGL_points = tau_to_t_points(LGL(N), t0, tf)

    q_and_der_polys = []
    D_nu = matrix_D_nu(N)
    for ii in range(interp_order + 1):
        coefs = (2 / h) ** ii * matrix_power(D_nu, ii) @ q_constr
        q_and_der_polys.append(bary_poly_2d(LGL_points, coefs))
    if uu is None:
        return q_and_der_polys
    else:
        u_poly = bary_poly_2d(coll_points, uu)
        return u_poly, q_and_der_polys


def interpolations_TD_pseudospectral(
    q_constr,
    xx,
    xx_dot,
    uu,
    scheme,
    t0,
    tf,
    scheme_order=2,
    u_interp="pol",
    x_interp="pol",
    n_interp=5000,
    given_t_arr=None,
):
    """
    Generates arrays of equispaced points with values of interpolations.

    'x_interp' and 'u_interp' define the way in which we interpolate the values
    of q, v and u between the given points.

    Parameters
    ----------
    q_constr: Numpy Array
        Values of q(t) caculated at certain LGL points in the top-down
        formulation
    xx : Numpy Array
        Values known of x(t)
    xx_dot : Numpy Array
        Values known of x_dot(t)
    uu : Numpy Array
        Values known of u(t)
    scheme : str
        Pseudospectral scheme used in the optimization.
        Acceptable values are:
            'LG'
            'LGR'
            'LGR_inv'
            'LGL'
            'JG'
            'JGR'
            'JGR_inv'
            'JGL'
            'CG'
            'CGR'
            'CGR_inv'
            'CGL'
    t0 : float
        starting time of interval of analysis
    t1 : float
        ending time of interval of analysis
    scheme_order : int, optional, default = 2
        For jacobi schemes, order of the scheme
    u_interp :  string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "smooth": 3d order spline interpolation
    x_interp : string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "Hermite": Hermite's 3d order spline interpolation
    n_interp : int, default 5000
        number of interpolation points

    Raises
    ------
    NameError
        When an unsupported value for scheme, x_interp or u_interp is used.

    Returns
    -------
    x_arr, x_dot_arr, u_arr : Numpy array
        equispaced values of interpolations.
    """
    if scheme[:3] == "TD_":
        scheme = scheme[3:]
    n_coll = uu.shape[0]
    N = q_constr.shape[0]

    if is2d(q_constr):
        n_q = q_constr.shape[-1]
    else:
        n_q = 1

    if is2d(xx):
        n_x = xx.shape[-1]
    else:
        n_x = 1

    problem_order = n_x // n_q
    assert N == problem_order + n_coll
    t_x = tau_to_t_points(
        TD_construction_points(n_coll, scheme, order=scheme_order), t0, tf
    )

    # coll_index = get_coll_indices(scheme)

    if x_interp == "Hermite":
        from scipy.interpolate import CubicHermiteSpline as hermite

    if scheme not in _implemented_schemes:
        NameError(f"Invalid scheme.\n valid options are {_implemented_schemes}")

    if given_t_arr is None:
        t_arr = linspace(t0, tf, n_interp)
    else:
        t_arr = given_t_arr

    if "pol" in [x_interp, u_interp]:
        u_pol, q_and_der_polys = Polynomial_interpolations_TD(
            q_constr,
            uu,
            t0,
            tf,
            n_coll,
            scheme,
            scheme_order,
        )
    if u_interp == "pol":
        u_arr = u_pol(t_arr)

    elif u_interp == "lin":
        tau_u, uu = extend_u_array(uu, scheme, n_coll, scheme_order)
        t_u = tau_to_t_points(tau_u, t0, tf)

        if len(uu.shape) == 1:
            u_arr = interp(t_arr, t_u, uu)
        elif len(uu.shape) == 2:
            u_arr = interp_2d(t_arr, t_u, uu)
        else:
            raise ValueError(
                f"U has {len(uu.shape)} dimensions, values accepted are 1 and 2"
            )
    elif u_interp == "smooth":
        tau_u, uu = extend_u_array(uu, scheme, n_coll, scheme_order)
        t_u = tau_to_t_points(tau_u, t0, tf)
        uu_dot = gradient(uu, t_u)
        u_arr = hermite(t_u, uu, uu_dot)(t_arr)
    else:
        raise NameError(
            f'Invalid interpolation method for u:{u_interp}.\n valid options are "pol", "lin", "smooth"'
        )
    if x_interp == "pol":
        q_and_der_arrs = []
        for ii in range(problem_order + 1):
            _newarr = force2d(q_and_der_polys[ii](t_arr))
            q_and_der_arrs.append(_newarr)
        q_and_der_arrs = concatenate(tuple(q_and_der_arrs), axis=1)
        x_arr = q_and_der_arrs[:, :-n_q]
        x_dot_arr = q_and_der_arrs[:, n_q:]
    elif x_interp == "lin":
        x_arr = interp_2d(t_arr, t_x, xx)
        x_dot_arr = interp_2d(t_arr, t_x, xx_dot)

    elif x_interp == "Hermite":
        herm = hermite(t_x, xx, xx_dot)
        x_arr = herm(t_arr)
        x_dot_arr = herm.derivative()(t_arr)
    else:
        raise NameError(
            f'Invalid interpolation method for x:{x_interp}.\n valid options are "pol", "lin", "Hermite"'
        )

    return x_arr, x_dot_arr, u_arr


def interpolations_deriv_TD_pseudospectral(
    q_constr,
    xx,
    xx_dot,
    scheme,
    deriv_order,
    t0,
    tf,
    n_coll,
    scheme_order,
    x_interp="pol",
    n_interp=5000,
    given_t_arr=None,
):
    """
    Generates arrays of equispaced points with values of interpolations of the
    derivatives of x.

    'x_interp' define the way in which we interpolate the values
    of q, v and u between the given points.

    Parameters
    ----------
    xx : Numpy Array
        Values known of x(t)
    xx_dot : Numpy Array
        Values known of x_dot(t)
    scheme : str
        Pseudospectral scheme used in the optimization.
        Acceptable values are:
            'LG'
            'LGR'
            'LGR_inv'
            'LGL'
            'JG'
            'JGR'
            'JGR_inv'
            'JGL'
            'CG'
            'CGR'
            'CGR_inv'
            'CGL'
    order : int
        differential order of the problem
    deriv_order : int
        differential order of the interpolations
    t0 : float
        starting time of interval of analysis
    t1 : float
        ending time of interval of analysis
    u_interp :  string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "smooth": 3d order spline interpolation
    x_interp : string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "Hermite": Hermite's 3d order spline interpolation
    n_interp : int, default 5000
        number of interpolation points

    Raises
    ------
    NameError
        When an unsupported value for scheme, x_interp or u_interp is used.

    Returns
    -------
    x_arr, x_dot_arr, u_arr : Numpy array
        equispaced values of interpolations.
    """
    if scheme[:3] == "TD_":
        scheme = scheme[3:]

    N = q_constr.shape[0]

    if is2d(q_constr):
        n_q = q_constr.shape[-1]
    else:
        n_q = 1

    if is2d(xx):
        n_x = xx.shape[-1]
    else:
        n_x = 1

    problem_order = n_x // n_q
    assert N == problem_order + n_coll
    # coll_points = tau_to_t_points(BU_coll_points(n_coll, scheme, order), t0, tf)
    t_x = tau_to_t_points(
        TD_construction_points(n_coll, scheme, order=scheme_order), t0, tf
    )
    LGL_points = tau_to_t_points(LGL(N), t0, tf)
    h = tf - t0

    if x_interp == "Hermite":
        from scipy.interpolate import CubicHermiteSpline as hermite

    if scheme not in _implemented_schemes:
        NameError(f"Invalid scheme.\n valid options are {_implemented_schemes}")

    if given_t_arr is None:
        t_arr = linspace(t0, tf, n_interp)
    else:
        t_arr = given_t_arr

    if x_interp == "pol":
        q_and_der_polys = Polynomial_interpolations_TD(
            q_constr,
            None,
            t0,
            tf,
            n_coll,
            scheme,
            scheme_order,
        )
        D_nu = matrix_D_nu(N)

        for jj in range(deriv_order - 1):
            ii = jj + problem_order + 1

            coefs = (2 / h) ** ii * matrix_power(D_nu, ii) @ q_constr
            q_and_der_polys.append(bary_poly_2d(LGL_points, coefs))

        q_and_der_arrs = []
        for ii in range(problem_order):
            _newarr = force2d(q_and_der_polys[ii + deriv_order](t_arr))
            q_and_der_arrs.append(_newarr)
        x_deriv_arr = concatenate(tuple(q_and_der_arrs), axis=1)

    elif x_interp == "lin":
        if deriv_order == 0:
            x_deriv_arr = interp_2d(t_arr, t_x, xx)
        elif deriv_order == 1:
            x_deriv_arr = find_der_polyline(t_arr, t_x, xx)
        else:
            x_deriv_arr = zeros_like(xx)

    elif x_interp == "Hermite":
        herm = hermite(t_x, xx, xx_dot)
        for ii in range(deriv_order):
            herm = herm.derivative()
        x_deriv_arr = herm(t_arr)
    else:
        raise NameError(
            'Invalid interpolation method for x.\n valid options are "pol", "lin", "Hermite"'
        )
    return x_deriv_arr


def dynamic_error_TD(
    q_constr,
    xx,
    xx_dot,
    uu,
    params,
    scheme,
    tf,
    F,
    problem_order=2,
    t0=0,
    scheme_order=2,
    u_interp="pol",
    x_interp="pol",
    n_interp=2000,
    scheme_params=None,
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


    Parameters
    ----------
    xx : Numpy Array
        Values known of x(t)
    xx_dot : Numpy Array
        Values known of x'(t)
    uu : Numpy Array
        Values known of u(t)
    params : list
        Physical problem parameters to be passed to F
    scheme : str
        Pseudospectral scheme used in the optimization.
        Acceptable values are:
            'LG'
            'LGR'
            'LGR_inv'
            'LGL'
            'JG'
            'JGR'
            'JGR_inv'
            'JGL'
            'CG'
            'CGR'
            'CGR_inv'
            'CGL'
    t1 : float
        ending time of interval of analysis
    F : Function of (x, u, params)
        A function of a dynamic sistem, so that
            x' = F(x, u, params)
    t0 : float, default = 0
        starting time of interval of analysis
    problem_order : int, default 2
        differential order of the problem. It will be used for dividing
        x but not for calculating the interpolation.
    u_interp :  string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "smooth": 3d order spline interpolation
    x_interp : string, optional
        Model of the interpolation that must be used. The default is "pol".
        Acceptable values are:
            "pol": corresponding polynomial interpolation
            "lin": lineal interpolation
            "Hermite": Hermite's 3d order spline interpolation
    n_interp : int, default 2000
        number of interpolation points
    scheme_params :dict or none, optional
        Aditional parameters of the scheme. The default is None.
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
    if scheme[:3] == "TD_":
        scheme = scheme[3:]

    # N = q_constr.shape[0]
    n_x = xx.shape[-1]
    n_q = n_x // problem_order

    # interpolation_order = n_x// q_constr.shape[-1]
    n_coll = uu.shape[0]
    # assert N == order + n_coll
    # coll_points = tau_to_t_points(BU_coll_points(n_coll, scheme, order), t0, tf)
    # t_x = tau_to_t_points(TD_construction_points(n_coll, scheme, order=scheme_order), t0, tf)
    # LGL_points = tau_to_t_points(LGL(N), t0, tf)

    if scheme_params is None:
        scheme_params = {}
    x_and_derivs = []
    # scheme_params["order"] = scheme_order

    x_interp_arr, x_dot_interp_arr, u_interp_arr = interpolations_TD_pseudospectral(
        q_constr,
        xx,
        xx_dot,
        uu,
        scheme,
        t0,
        tf,
        scheme_order,
        u_interp,
        x_interp,
        n_interp,
    )

    x_and_derivs.append(get_x_divisions(x_interp_arr, problem_order))
    for jj in range(1, problem_order + 1):
        x_and_derivs.append(
            get_x_divisions(
                interpolations_deriv_TD_pseudospectral(
                    q_constr,
                    xx,
                    xx_dot,
                    scheme,
                    jj,
                    t0,
                    tf,
                    n_coll,
                    scheme_order,
                    x_interp,
                    n_interp,
                ),
                problem_order,
            )
        )

    q_and_d_interp = copy(x_interp_arr)
    for jj in range(problem_order):
        q_and_d_interp[:, n_q * jj : n_q * (jj + 1)] = x_and_derivs[jj][0]

    if mode == "q":
        x_in_f = q_and_d_interp
    elif mode == "x":
        x_in_f = x_interp
    else:
        raise ValueError(
            f"Value of mode {mode} not valid. Valid values are 'q' and 'x'."
        )

    f_interp = zeros([n_interp, n_q])
    for ii in range(n_interp):
        f_interp[ii, :] = F(x_in_f[ii], u_interp_arr[ii], params)[-n_q:]
    x_and_derivs[0].append(f_interp)

    dyn_errs = []
    for jj in range(problem_order):
        dyn_errs_order = []
        for ii in range(problem_order - jj):
            dyn_errs_order.append(
                x_and_derivs[jj + 1][ii] - x_and_derivs[0][ii + jj + 1]
            )
        dyn_errs.append(dyn_errs_order)
    return dyn_errs
