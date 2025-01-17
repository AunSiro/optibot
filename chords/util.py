# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:52:01 2023

@author: Siro Moreno

Here we present some useful functions that whose objective is to simplify
the workflow and to streamline the utilization of Chords and plotting the results.
"""
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

# from sympy import legendre_poly
from functools import lru_cache
from math import ceil, factorial
from .pseudospectral import (
    bary_poly,
    coll_points,
    node_points,
    bary_poly_2d,
    _gauss_like_schemes,
    _gauss_inv_schemes,
)
from .numpy import combinefunctions, store_results

# Uniform output style functions


oct_fig_size = [10, 6]


def sch_to_lab(sch):
    label_dict = {
        "hs": "HS-1-Topputo",
        "hs_mod": "HS-2-Topputo",
        "hs_parab": "HS-1",
        "hs_mod_parab": "HS-2",
        "trapz": "TZ-1",
        "trapz_mod": "TZ-2",
        "hsj": "HSJ-Topputo",
        "hsj_parab": "HSJ",
        "hsj_parab_mod": "HSJ (sep $u_c$)",
        "rk4": "rk4",
        "hsn": "HS-M-Topputo",
        "hsn_parab": "HS-M",
        "trapz_n": "TZ-M",
        "LG": "LG",
        "LG2": "LG2",
        "LGL": "LGL",
        "LGR": "LGR",
        "LGR_inv": "LGR_inv",
        "JG": "JG",
        "JGR": "JGR",
        "JGR_inv": "JGR_inv",
        "JGL": "JGL",
        "CG": "Cheb",
        "CG_inv": "Cheb_inv",
        "CGL": "Cheb-L",
        "CGR": "Cheb-R",
        "CGR_inv": "Cheb-R_inv",
    }
    if sch[:3] == "ph_":
        ph = True
        sch = sch[3:]
    else:
        ph = False

    if sch[:3] == "BU_":
        sch = sch[3:]
        label = label_dict[sch]
        label += " D-A"
    elif sch[:3] == "TD_":
        sch = sch[3:]
        label = label_dict[sch]
        label += " I-A"
    else:
        label = label_dict[sch]

    if ph:
        label = "adaptive " + label
    return label


def sch_to_long_label(sch):
    titles = [
        "Hermite Simpson",
        "2nd order Hermite Simpson",
        "Trapezoidal",
        "2nd order Trapezoidal",
        "Hermite Simpson (Topputo)",
        "2nd order Hermite Simpson (Topputo)",
        "Hermite-Simpson-Jacobi (Topputo)",
        "Hermite-Simpson-Jacobi",
        "Hermite-Simpson-Jacobi (mod_u)",
        "Runge-Kutta 4",
        "Mth order Hermite Simpson",
        "Mth  order Hermite Simpson (Topputo)",
        "Mth  order Trapezoidal",
        "LG",
        "LG2",
        "LGL",
        "LGR",
        "LGR_inv",
        "JG",
        "JGR",
        "JGR_inv",
        "JGL",
        "Chebyshev",
        "Chebyshev_inv",
        "Chebyshev-Lobato",
        "Chebyshev-Radau",
        "Chebyshev-Radau_inv",
    ]
    schemes = [
        "hs_parab",
        "hs_mod_parab",
        "trapz",
        "trapz_mod",
        "hs",
        "hs_mod",
        "hsj",
        "hsj_parab",
        "hsj_parab_mod",
        "rk4",
        "hsn_parab",
        "hsn",
        "trapz_n",
        "LG",
        "LG2",
        "LGL",
        "LGR",
        "LGR_inv",
        "JG",
        "JGR",
        "JGR_inv",
        "JGL",
        "CG",
        "CG_inv",
        "CGL",
        "CGR",
        "CGR_inv",
    ]
    lname_dict = {}
    for ii in range(len(titles)):
        lname_dict[schemes[ii]] = titles[ii]

    if sch[:3] == "ph_":
        ph = True
        sch = sch[3:]
    else:
        ph = False

    if sch[:3] == "BU_":
        sch = sch[3:]
        label = lname_dict[sch]
        label += " integrative"
    elif sch[:3] == "TD_":
        sch = sch[3:]
        label = lname_dict[sch]
        label += " derivative"
    else:
        label = lname_dict[sch]

    if ph:
        label = "ph adaptive " + label
    return label


def sch_to_color(sch):
    color_dict = {}
    piece_sch = [
        "hs",
        "trapz_mod",
        "trapz",
        "hs_mod",
        "hsj",
        "rk4",
        "hsj_mod",
    ]
    pseud_sch = ["LG", "LGL", "LGR", "JG", "JGR", "JGL", "CG", "CGL", "CGR", "LG2"]
    n_sch = len(pseud_sch)

    for ii, sc_name in enumerate(piece_sch):
        color_dict[sc_name] = f"C{ii}"

    for ii, sc_name in enumerate(pseud_sch):
        color_dict[sc_name] = f"C{ii+4}"

    c_ch = colormaps["cubehelix"](np.linspace(0.2, 0.9, n_sch))
    c_gp2 = colormaps["gnuplot2"](np.linspace(0.85, 0.2, n_sch))

    for ii, sc_name in enumerate(pseud_sch):
        color_dict["BU_" + sc_name] = c_ch[ii]
    for ii, sc_name in enumerate(pseud_sch):
        color_dict["TD_" + sc_name] = c_gp2[ii]

    color_dict["BU_LG"] = colormaps["cubehelix"](0.45)
    color_dict["TD_LG"] = colormaps["gist_stern"](0.12)

    color_dict["BU_LGR"] = colormaps["gnuplot2"](0.2)
    color_dict["TD_LGR"] = colormaps["terrain"](0.6)

    color_dict["BU_LGL"] = colormaps["cubehelix"](0.7)
    color_dict["TD_LGL"] = colormaps["cubehelix"](0.3)

    sch = sch.replace("_parab", "")
    sch = sch.replace("ph_", "")
    sch = sch.replace("_inv", "")
    sch = sch.replace("trapz_n", "trapz_mod")
    sch = sch.replace("hsn", "hs_mod")
    # sch = sch.replace("BU_", "")
    # sch = sch.replace("TD_", "")
    return color_dict[sch]


def scheme_kwargs(sch, longlabel=False, colors_for_parab=False, order=None):
    if colors_for_parab:
        color = sch_to_color(sch)
        ls = "-"
    else:
        color = sch_to_color(sch)
        if "hs" in sch and "parab" not in sch:
            ls = "--"
        else:
            ls = "-"

    if longlabel:
        label = sch_to_long_label(sch)
    else:
        label = sch_to_lab(sch)

    if order is None:
        pass
    else:
        if order == 1:
            ls = "-"
        elif order == 2:
            ls = ":"
        else:
            ls = "--"

        label = label + " order " + str(order)

    kwargs = {"marker": "o", "c": color, "ls": ls, "label": label}

    return kwargs


def set_fonts():
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.rcParams.update({"font.size": 15})


def plot_by_segments(t_arr, y_plot, label, N, end_t, plot_dict=None):
    if plot_dict is None:
        plot_dict = {"c": "gray"}
    interv_n = (N * t_arr) / end_t
    cut_p = 0
    for ll in range(1, N + 1):
        jj = np.searchsorted(interv_n, ll)
        plt.plot(
            t_arr[cut_p:jj],
            y_plot[cut_p:jj],
            label=label if cut_p == 0 else None,
            **plot_dict,
        )
        cut_p = jj


def graph_by_segments(
    results, schemes, N, thing_to_plot, title, ylabel, component="all"
):
    plt.figure(figsize=oct_fig_size)
    plot_coll_p = False
    for scheme in schemes:
        t_arr = np.linspace(
            0, results[scheme][N]["t"][-1], results[scheme][N][thing_to_plot].shape[0]
        )
        interv_n = (N * t_arr) / results[scheme][N]["t"][-1]
        cut_p = 0
        for ll in range(1, N + 1):
            jj = np.searchsorted(interv_n, ll)
            y_plot = results[scheme][N][thing_to_plot]
            if component != "all":
                y_plot = y_plot[:, component]
            plt.plot(
                t_arr[cut_p:jj],
                y_plot[cut_p:jj],
                "-",
                c=sch_to_color(scheme),
                label=sch_to_lab(scheme) if cut_p == 0 else None,
            )
            cut_p = jj
        if "hs" in scheme:
            plot_coll_p = True
    plt.plot(
        results[scheme][N]["t"],
        np.zeros(N + 1),
        "ok",
        ms=5,
        label="knot & collocation points",
    )
    if plot_coll_p:
        plt.plot(
            results[scheme][N]["t_c"],
            np.zeros(N),
            "ow",
            ms=5,
            markeredgecolor="k",
            label="collocation points",
        )
    plt.legend()
    plt.grid()
    # plt.ylim([-0.01,y_max_list[ii]])
    plt.title(title)
    plt.xlabel("Time(s)")
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.0)


def gen_fig_filename(
    problem_name,
    graph_name,
    schemes_printed,
    save_format="pdf",
    N=None,
    q_counter=None,
    add=None,
):
    from os.path import join

    folder_name = problem_name + "_figs"
    schemes = schemes_printed.copy()  # avoid modifying original list
    schemes.sort()
    title = f"{problem_name}_{graph_name}"
    if not (q_counter is None):
        title += f"_q_{q_counter+1}"
    if N is None:
        title += "_vs_N"
    else:
        title += f"_N{N}"
    for scheme in schemes:
        title += f"--{scheme}"
    if add is not None:
        title += f"--{add}"
    title += f".{save_format}"
    filename = join(folder_name, title)
    return filename


def save_fig(
    problem_name,
    graph_name,
    schemes_printed,
    save_format="pdf",
    N=None,
    q_counter=None,
    add=None,
):
    from os.path import isdir
    from os import makedirs

    filename = gen_fig_filename(
        problem_name,
        graph_name,
        schemes_printed,
        save_format,
        N,
        q_counter,
        add,
    )
    folder_name = problem_name + "_figs"
    if not isdir(folder_name):
        makedirs(folder_name)
    plt.savefig(filename, format=save_format)


# --------------------------- Gauss Integration -------------------------------


# @lru_cache(maxsize=2000)
# def LG_weight(N, i, precission=16):
#     Pn = legendre_poly(N, polys=True)
#     Pn_d = Pn.diff()
#     xi = LG(N, precission)[i]
#     wi = 2 / ((1 - xi**2) * (Pn_d.eval(xi) ** 2))
#     return wi


@lru_cache(maxsize=2000)
@store_results
def leggauss(N):
    return np.polynomial.legendre.leggauss(N)


def gauss_integral(f, N, t0, t1):
    scale = t1 - t0
    points, weights = leggauss(N)
    # points = (np.array(LG(N)) + 1) / 2
    points = (points + 1.0) / 2.0
    points = t0 + scale * points
    # weights = [LG_weight(N, ii) for ii in range(N)]
    f_vals = np.array([f(points[ii]) for ii in range(N)], dtype="float64").flatten()
    _a = weights * f_vals
    return scale * np.sum(_a) / 2.0


def gauss_integral_2d(f, N, t0, t1):
    scale = t1 - t0
    points, weights = leggauss(N)
    # points = (np.array(LG(N)) + 1) / 2
    points = (points + 1.0) / 2.0
    points = t0 + scale * points
    # weights = [LG_weight(N, ii) for ii in range(N)]
    f_vals = f(points)
    _a = np.expand_dims(weights, 1) * f_vals
    return scale * np.sum(_a, axis=0) / 2.0


def gauss_rep_integral(f, t0, t1, n_pol, n_integ=1, f2d=False):
    n_pol_cauchy = n_pol + n_integ - 1
    n_gauss = ceil((n_pol_cauchy + 1) / 2)
    cauchy_f = lambda t: (t1 - t) ** (n_integ - 1) * f(t)
    integ_f = gauss_integral_2d if f2d else gauss_integral
    return 1 / factorial(n_integ - 1) * integ_f(cauchy_f, n_gauss, t0, t1)


def poly_integral(f, n_pol, t0, t1, y0=0):
    scale = t1 - t0

    points = (np.array(leggauss(n_pol + 1)[0], dtype="float64") + 1) / 2
    points = t0 + scale * points
    points = list(points)

    N_gauss = ceil((n_pol + 1) / 2)
    y = [y0] + [y0 + gauss_integral(f, N_gauss, t0, ii) for ii in points]

    points = np.array(
        [
            t0,
        ]
        + points,
        dtype="float64",
    )
    y = np.array(y, dtype="float64").flatten()
    return bary_poly(points, y)


def poly_integral_2d(f, n_pol, t0, t1, y0=0):
    y_example = f(t0)
    if len(y_example.shape) >= 2:
        raise NotImplementedError(
            f"The output of f has shape {y_example.shape} but implemented "
            + "methods only allow for shape (n,)"
        )
    if y_example.shape == () or len(y_example) == 1:
        return poly_integral(f, n_pol, t0, t1, y0)
    dim = y_example.shape[0]

    y0 = np.array(y0)
    if y0.shape != y_example.shape:
        if y0.shape == () or len(y0) == 1:
            y0 = y0 * np.ones(dim)
        else:
            raise ValueError(
                f"y0 has unexpected shape {y0.shape}, expected was {y_example.shape}"
            )
    scale = t1 - t0
    nq = y_example.shape[0]

    points = (np.array(leggauss(n_pol + 1)[0], dtype="float64") + 1) / 2
    points = t0 + scale * points

    mat = np.zeros([n_pol + 2, nq], dtype="float64")

    N_gauss = ceil((n_pol + 1) / 2)

    for ii in range(n_pol + 1):
        mat[ii + 1, :] = gauss_integral_2d(f, N_gauss, t0, points[ii])
    mat = np.expand_dims(y0, 0) + mat
    points = np.concatenate((np.expand_dims(t0, 0), points))

    return bary_poly_2d(points, mat)


@lru_cache(maxsize=2000)
def Lag_pol_2d(N, scheme, order=2):
    tau_arr = np.array(coll_points(N, scheme, order=order), dtype="float64")
    return bary_poly_2d(tau_arr, np.eye(N))


@lru_cache(maxsize=2000)
def Lag_pol_2d_x(N, scheme, order=2):
    if scheme in _gauss_like_schemes:
        arr_points_tau = node_points(N - 1, scheme)
        arr_points_tau = arr_points_tau + [
            1.0,
        ]
    elif scheme in _gauss_inv_schemes:
        arr_points_tau = node_points(N - 1, scheme)
        arr_points_tau = [
            -1.0,
        ] + arr_points_tau
    else:
        arr_points_tau = node_points(N, scheme)

    tau_arr = np.array(arr_points_tau, dtype="float64")

    return bary_poly_2d(tau_arr, np.eye(N))


@lru_cache(maxsize=2000)
def Lag_integ_2d(N, scheme, integ_order, order=2):
    if integ_order == 0:
        return Lag_pol_2d(N, scheme, order)
    deriv_poly = Lag_integ_2d(N, scheme, integ_order - 1, order)
    poly_deg = N - 2 + integ_order
    new_poly = poly_integral_2d(deriv_poly, poly_deg, -1, 1)
    return new_poly


@lru_cache(maxsize=2000)
@store_results
def get_weights(N, scheme, order=2, mode="u"):
    """
    Generate weights for quadrature integration. If an closed formula
    is known, it will be used. If not, weights will be calculated
    by gauss integration of barycentric lagrange polynomials.

    Parameters
    ----------
    N : int
        number of points.
    scheme : str
        scheme used.
    order : int, optional
        if the scheme requires a differential order, like jacobi-gauss.
        The default is 2.
    mode : str, optional, default 'u'
        kind of points that make the Lagrange basis:
            'u' : collocation points
            'x' : array points

    Returns
    -------
    numpy array
        weights.

    """
    if mode == "u":
        if scheme == "LG":
            return leggauss(N)[1]
        pol = Lag_pol_2d(N, scheme, order)
    elif mode == "x":
        pol = Lag_pol_2d_x(N, scheme)
    else:
        raise ValueError(f'Invalid mode: "{mode}", valid modes are "u" and "x".')

    N_gauss = ceil((N + 1) / 2)

    return gauss_integral_2d(pol, N_gauss, -1.0, 1.0)


# ----------------- Miscelanea -----------------------


def is_integer(x):
    try:
        result = int(x) == x
        return result
    except TypeError:
        return False


def log_base(x, base):
    aa = np.log(x)
    bb = np.log(base)
    return aa / bb
