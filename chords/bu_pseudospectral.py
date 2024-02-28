#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:35:30 2023

@author: Siro Moreno

Here we define functions needed to operate with bottom-up pseudospectral
collocations schemes. In order to keep the best accuracy in interpolations, 
barycentric formulas are constructed.
"""

from .pseudospectral import (
    LG,
    LGL,
    LGR,
    JG,
    JGR,
    JGR_inv,
    JGL,
    CG,
    CGL,
    CGR,
    bary_poly,
    bary_poly_2d,
    extend_u_array,
    find_der_polyline,
)
from .util import gauss_rep_integral, poly_integral_2d, Lag_integ_2d
from .piecewise import interp_2d, is2d, get_x_divisions, force2d
from .numpy import store_results
from functools import lru_cache
from numpy import (
    zeros,
    array,
    concatenate,
    interp,
    gradient,
    linspace,
    # eye,
    arange,
    expand_dims,
)

# from sympy import jacobi_poly
from scipy.special import factorial
from copy import copy


_gauss_like_schemes = ["LG", "JG", "CG"]
_radau_like_schemes = ["LGR", "JGR", "CGR"]
_radau_inv_schemes = [_sch + "_inv" for _sch in _radau_like_schemes]
_lobato_like_schemes = ["LGL", "JGL", "CGL"]
_other_schemes = []
_implemented_schemes = (
    _gauss_like_schemes
    + _radau_like_schemes
    + _radau_inv_schemes
    + _lobato_like_schemes
    + _other_schemes
)


@lru_cache(maxsize=2000)
def BU_coll_points(N, scheme, order=2, precission=16):
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
            'LG2'
            'JG'
            'JGR'
            'JGR_inv'
            'JGL'
            'CG'
            'CGR'
            'CGR_inv'
            'CGL'
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
    elif scheme in ["LGL"]:
        return LGL(N, precission)
    elif scheme == "JG":
        return JG(N, order, precission)
    elif scheme == "JGR":
        return JGR(N, order, precission)
    elif scheme == "JGR_inv":
        return JGR_inv(N, order, precission)
    elif scheme == "JGL":
        return JGL(N, order, precission)
    elif scheme == "CG":
        return CG(N)
    elif scheme == "CGR":
        return CGR(N)
    elif scheme == "CGR_inv":
        return [-ii for ii in CGR(N)[::-1]]
    elif scheme == "CGL":
        return CGL(N)
    else:
        raise ValueError(
            f"Unsupported scheme {scheme}, valid schemes are: {_implemented_schemes}"
        )


@lru_cache(maxsize=2000)
def BU_construction_points(N, scheme, order=2, precission=16):
    """
    Generates a list of with values of tau for construction points for schemes
    with N collocation points. Construction points are the collocation points,
    excluding the initial point if it is a collocation point, and
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
        return coll_p + [1.0]
    elif scheme in _radau_like_schemes:
        return coll_p[1:] + [1.0]
    elif scheme in _radau_inv_schemes:
        return coll_p
    elif scheme in _lobato_like_schemes:
        return coll_p[1:]
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


def get_coll_indices(scheme):
    """
    returns a slice that can be used to separate collocation points from
    an array that includes first and last point

    Parameters
    ----------
    scheme : str
        the scheme used.

    Raises
    ------
    NotImplementedError
        When the scheme is not recognized.

    Returns
    -------
    coll_index : slice
        slice to be used as index in the array to extract the collocation points.

    """
    if scheme in _gauss_like_schemes:
        coll_index = slice(1, -1)
    elif scheme in _radau_like_schemes:
        coll_index = slice(None, -1)
    elif scheme in _radau_inv_schemes:
        coll_index = slice(1, None)
    elif scheme in _lobato_like_schemes:
        coll_index = slice(None, None)
    elif scheme in _other_schemes:
        if scheme in []:
            pass
        else:
            raise NotImplementedError(
                f"scheme {scheme} in category 'others' is not yet implemented"
            )
    else:
        raise NotImplementedError(f"Scheme {scheme} not implemented yet")
    return coll_index


@lru_cache(maxsize=2000)
def BU_unit_Lag_pol(N, scheme, n, order=2, precission=16):
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
        Barycentric Lagrange Polynomial L_n(x) for x in [-1,1].

    """
    x = BU_coll_points(N, scheme, order, precission)

    y = zeros(N)
    y[n] = 1
    return bary_poly(x, y)


def tau_to_t_points(points, t0, tf):
    """
    converts a point or series of points from tau in [-1,1] to t in [t0, tf]

    Parameters
    ----------
    points : number, list, array or tuple
        points in tau
    t0 : float
        initial t
    tf : float
        final t

    Returns
    -------
    new_points : number, list, array or tuple
        points in t

    """
    points_arr = array(points)
    h = tf - t0
    new_points = t0 + h * (points_arr + 1) / 2
    if type(points) == list:
        new_points = list(new_points)
    elif type(points) == tuple:
        new_points = tuple(new_points)
    elif type(points) == float:
        new_points = float(new_points)
    return new_points


def tau_to_t_function(f, t0, tf):
    """
    converts a function from f(tau), tau in [-1,1] to f(t), t in [t0, tf]

    Parameters
    ----------
    f : function
        function of tau: f(tau)
    t0 : float
        initial t
    tf : float
        final t

    Returns
    -------
    new_f : function
        function of t: f(t)

    """
    h = tf - t0

    def new_F(t):
        tau = 2 * (t - t0) / h - 1
        return f(tau)

    try:
        old_docstring = str(f.__doc__)
    except:
        old_docstring = "function of tau"
    try:
        old_f_name = str(f.__name__)
    except:
        old_f_name = "Unnamed Function"

    new_docstring = f"""
    This is a time t based version of function {old_f_name}.
    This expanded function is designed to operate with time t: F(t)
    While the old function was designed for tau: F(tau)
    Old function documentation:
    """
    new_docstring += old_docstring
    new_F.__doc__ = new_docstring
    new_F.__name__ = old_f_name + " of tau"
    return new_F


@lru_cache(maxsize=2000)
def BU_unit_Lag_pol_t(N, scheme, n, t0, tf, order=2, precission=16):
    """
    Generate a barycentric numeric Lagrange polynomial over N node points.
    This will create a function for use with t in [t0, tf].
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
        Barycentric Lagrange Polynomial L_n(t) for t in [t0, tf].

    """
    L_tau = BU_unit_Lag_pol(N, scheme, n, order, precission)

    return tau_to_t_function(L_tau, t0, tf)


@lru_cache(maxsize=2000)
@store_results
def _Lag_integ(
    N_coll,
    n_lag_pol,
    scheme,
    deriv_order,
    t_constr_index,
    scheme_order=2,
    precission=16,
):
    """
    Calculate a definite integral of a Lagrange polynomial to use in the
    definition of an integration matrix.

    Parameters
    ----------
    N_coll : int
        Number of collocation points
    n_lag_pol : int between 0 and N_coll-1
        current lagrange polynomial will be 1 for tau_{n_lag_pol}, and 0 for
        all the other collocation points tau_i for i = 0 ... N_coll-1
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
    deriv_order : int
        derivation level for which the element in the matrix is being
        calculated. For example:
            for q, deriv_order = 0
            for v, deriv_order = 1
    t_constr_index : int
        index of the construction point for which the element is being calculated
    scheme_order : int, optional
        Differential order of the problem. The default is 2.
    precission : int, optional
        Precission in colloction point calculation. The default is 20.

    Returns
    -------
    integral : float
        result of the numerical integration.

    """
    constr_points = BU_construction_points(N_coll, scheme, scheme_order, precission)
    tf = constr_points[t_constr_index]
    lag_pol = BU_unit_Lag_pol(N_coll, scheme, n_lag_pol, scheme_order, precission)
    integration_order = scheme_order - deriv_order
    integral = gauss_rep_integral(lag_pol, -1, tf, N_coll - 1, integration_order)
    return integral


@lru_cache(maxsize=2000)
@store_results
def Integration_Polynomial_Matrix(
    N_coll, deriv_order, scheme, scheme_order, precission=16
):
    M = scheme_order
    N = N_coll
    P = deriv_order
    constr_points = array(
        BU_construction_points(N, scheme, scheme_order, precission),
        dtype="float64",
    )
    int_pol = Lag_integ_2d(N, scheme, M - P, M)
    return int_pol(constr_points)


@lru_cache(maxsize=2000)
@store_results
def Integration_Matrix(N_coll, scheme, deriv_order, h, scheme_order=2, precission=16):
    assert (
        deriv_order < scheme_order
    ), "derivation order must be smaller than differential order of the problem"
    assert deriv_order >= 0
    assert scheme_order >= 1
    constr_points = array(
        BU_construction_points(N_coll, scheme, scheme_order, precission),
        dtype="float64",
    )
    n_t = len(constr_points)
    M = scheme_order
    N = N_coll
    P = deriv_order
    matrix = zeros([n_t, M + N])
    # for t_index in range(n_t):
    #     t = tau_to_t_points(constr_points[t_index], 0, h)
    #     for ii in range(M - P):
    #         matrix[t_index, P + ii] = t**ii / factorial(ii)
    #     for ii in range(N):
    #         matrix[t_index, M + ii] = (h / 2) ** (M - P) * _Lag_integ(
    #             N_coll, ii, scheme, P, t_index, scheme_order, precission
    #         )
    matrix[:, P] = 1.0
    if M - P > 1:
        constr_t = (1 + constr_points) * h / 2
        ii_arr = expand_dims(arange(1, M - P), 0)
        t_exp_mat = expand_dims(constr_t, 1) ** ii_arr / factorial(ii_arr)
        matrix[:, P + 1 : M] = t_exp_mat
    _M = Integration_Polynomial_Matrix(
        N_coll, deriv_order, scheme, scheme_order, precission
    )
    int_pol_mat = (h / 2) ** (M - P) * _M
    matrix[:, M:] = int_pol_mat
    return matrix


@lru_cache(maxsize=2000)
@store_results
def Extreme_Matrix(N_coll, scheme, point, scheme_order=2, precission=16):
    matrix = zeros([1, N_coll])
    for ii in range(N_coll):
        pol = BU_unit_Lag_pol(
            N_coll, scheme, ii, order=scheme_order, precission=precission
        )
        if point == "start":
            matrix[0, ii] = pol(-1)
        elif point == "end":
            matrix[0, ii] = pol(1)
        else:
            raise ValueError(f"point must be 'start' or 'end', not {point}")
    return matrix


def Polynomial_interpolations_BU(
    xx_dot, x_0, uu, scheme, problem_order, t0, tf, N_coll, scheme_order=2
):
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    n_q = len(x_0) // problem_order
    highest_der = xx_dot[:, -n_q:]
    coll_index = get_coll_indices(scheme)
    highest_der_col = highest_der[coll_index, :]
    n_col = highest_der_col.shape[0]

    q_and_ders = []
    for _ii in range(problem_order):
        q_and_ders.append(x_0[n_q * _ii : n_q * (_ii + 1)])

    coll_points = tau_to_t_points(BU_coll_points(n_col, scheme, scheme_order), t0, tf)

    highest_der_poly = bary_poly_2d(coll_points, highest_der_col)

    q_and_der_polys = [
        highest_der_poly,
    ]
    for ii in range(problem_order):
        _prev_poly = q_and_der_polys[-1]
        _new_poly = poly_integral_2d(
            _prev_poly, n_col - 1 + ii, t0, tf, q_and_ders[problem_order - ii - 1]
        )
        q_and_der_polys.append(_new_poly)

    if uu is None:
        return q_and_der_polys[::-1]
    else:
        u_poly = bary_poly_2d(coll_points, uu)
        return u_poly, q_and_der_polys[::-1]


def interpolations_BU_pseudospectral(
    xx,
    xx_dot,
    uu,
    scheme,
    interp_order,
    t0,
    tf,
    scheme_order=2,
    u_interp="pol",
    x_interp="pol",
    n_interp=5000,
):
    """
    Generates arrays of equispaced points with values of interpolations.

    'x_interp' and 'u_interp' define the way in which we interpolate the values
    of q, v and u between the given points.

    Parameters
    ----------
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
    interp_order : int
        differential order of the problem that will be used for
        generating the interpolations
    t0 : float
        starting time of interval of analysis
    tf : float
        ending time of interval of analysis
    scheme_order : int, default 2
        For Jacobi schemes, order of the scheme. otherwise, not used
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
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    if is2d(xx):
        x_0 = xx[0, :]
    else:
        x_0 = xx[0]
    n_q = len(x_0) // interp_order
    highest_der = xx_dot[:, -n_q:]
    coll_index = get_coll_indices(scheme)
    highest_der_col = highest_der[coll_index, :]
    n_col = highest_der_col.shape[0]

    if x_interp == "Hermite":
        from scipy.interpolate import CubicHermiteSpline as hermite

    if scheme not in _implemented_schemes:
        NameError(f"Invalid scheme.\n valid options are {_implemented_schemes}")

    t_arr = linspace(t0, tf, n_interp)
    t_x = tau_to_t_points(
        array([-1.0] + BU_construction_points(n_col, scheme, scheme_order)), t0, tf
    )

    if "pol" in [x_interp, u_interp]:
        u_pol, q_and_der_polys = Polynomial_interpolations_BU(
            xx_dot, x_0, uu, scheme, interp_order, t0, tf, n_col
        )

    if u_interp == "pol":
        u_arr = u_pol(t_arr)

    elif u_interp == "lin":
        tau_u, uu = extend_u_array(uu, scheme, n_col, scheme_order)
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
        tau_u, uu = extend_u_array(uu, scheme, n_col, scheme_order)
        t_u = tau_to_t_points(tau_u, t0, tf)
        uu_dot = gradient(uu, t_u)
        u_arr = hermite(t_u, uu, uu_dot)(t_arr)
    else:
        raise NameError(
            f'Invalid interpolation method for u:{u_interp}.\n valid options are "pol", "lin", "smooth"'
        )
    if x_interp == "pol":
        q_and_der_arrs = []
        for ii in range(interp_order + 1):
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


def _v_sum(t_arr, i):
    """
    Generates the coefficient V for barycentric coordinates

    Parameters
    ----------
    t_arr : iterable of floats
        values of t
    i : int
        index of current point.

    Returns
    -------
    v_i : float
        coefficient V.

    """
    n = len(t_arr)
    prod_coef = [ii for ii in range(n)]
    prod_coef.pop(i)
    v_i = 1.0
    for jj in prod_coef:
        v_i *= t_arr[i] - t_arr[jj]
    return 1.0 / v_i


def _matrix_D_bary(t_arr):
    """
    Generates the Derivation Matrix for the given scheme from
    barycentric coordinates

    Parameters
    ----------
    t_arr: array of times

    Returns
    -------
    M : NumPy Array
        Derivation Matrix.

    """
    N = len(t_arr)
    M = zeros((N, N), dtype="float64")
    v_arr = [_v_sum(t_arr, ii) for ii in range(N)]
    for i in range(N):
        j_range = [j for j in range(N)]
        j_range.pop(i)
        for j in j_range:
            M[i, j] = (v_arr[j] / v_arr[i]) / (t_arr[i] - t_arr[j])
    for j in range(N):
        M[j, j] = -sum(M[j, :])
    return M


def interpolations_deriv_BU_pseudospectral(
    xx,
    xx_dot,
    scheme,
    problem_order,
    deriv_order,
    t0,
    tf,
    scheme_order=2,
    x_interp="pol",
    n_interp=5000,
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
    interp_order : int
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
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    if is2d(xx):
        x_0 = xx[0, :]
    else:
        x_0 = xx[0]
    n_q = len(x_0) // problem_order
    highest_der = xx_dot[:, -n_q:]
    coll_index = get_coll_indices(scheme)
    highest_der_col = highest_der[coll_index, :]
    n_col = highest_der_col.shape[0]

    if x_interp == "Hermite":
        from scipy.interpolate import CubicHermiteSpline as hermite

    if scheme not in _implemented_schemes:
        NameError(f"Invalid scheme.\n valid options are {_implemented_schemes}")

    t_arr = linspace(t0, tf, n_interp)
    t_x = tau_to_t_points(
        array([-1.0] + BU_construction_points(n_col, scheme, scheme_order)), t0, tf
    )

    if x_interp == "pol":
        D = _matrix_D_bary(t_x)
        q_and_der_polys = Polynomial_interpolations_BU(
            xx_dot, x_0, None, scheme, problem_order, t0, tf, n_col
        )
        for jj in range(deriv_order - 1):
            highest_der = D @ highest_der
            highest_der_poly = bary_poly_2d(t_x, highest_der)
            q_and_der_polys.append(highest_der_poly)

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
            x_deriv_arr = zeros((len(t_arr), len(x_0)), dtype="float64")

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


def dynamic_error_BU(
    x_arr,
    x_dot_arr,
    u_arr,
    params,
    scheme,
    tf,
    F,
    t0=0,
    problem_order=2,
    scheme_order=2,
    u_interp="pol",
    x_interp="pol",
    n_interp=2000,
    interp_order=None,
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
    x_arr : Numpy Array
        Values known of x(t)
    x_dot_arr : Numpy Array
        Values known of x'(t)
    u_arr : Numpy Array
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
        differential order of the problem
    scheme_order : int, default 2
        differential order of the scheme if scheme is Jacobi Gauss
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
    interp_order : int or None, default None
        differential order of the problem interpolations.
        If None, problem_order will be used
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
    if scheme[:3] == "BU_":
        scheme = scheme[3:]
    if is2d(x_arr):
        x_0 = x_arr[0, :]
    else:
        x_0 = x_arr[0]
    n_q = len(x_0) // problem_order

    if interp_order is None:
        interp_order = problem_order
    x_and_derivs = []

    x_interp_arr, x_dot_interp_arr, u_interp_arr = interpolations_BU_pseudospectral(
        x_arr,
        x_dot_arr,
        u_arr,
        scheme,
        interp_order,
        t0,
        tf,
        scheme_order=scheme_order,
        u_interp=u_interp,
        x_interp=x_interp,
        n_interp=n_interp,
    )

    x_and_derivs.append(get_x_divisions(x_interp_arr, problem_order))
    for jj in range(1, problem_order + 1):
        x_and_derivs.append(
            get_x_divisions(
                interpolations_deriv_BU_pseudospectral(
                    x_arr,
                    x_dot_arr,
                    scheme,
                    interp_order,
                    jj,
                    t0,
                    tf,
                    scheme_order=scheme_order,
                    x_interp=x_interp,
                    n_interp=n_interp,
                ),
                scheme_order,
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
