# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:52:01 2023

@author: Siro Moreno

Here we present some useful functions that whose objective is to simplify
the workflow and to streamline the utilization of Chords and plotting the results.
"""
import matplotlib.pyplot as plt
import numpy as np
from sympy import legendre_poly
from functools import lru_cache
from math import ceil, factorial
from .pseudospectral import LG, bary_poly

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
        "hsn": "HS-3-Topputo",
        "hsn_parab": "HS-3",
        "trapz_n": "TZ-3",
    }
    return label_dict[sch]


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
        "3rd order Hermite Simpson",
        "3rd order Hermite Simpson (Topputo)",
        "3rd order Trapezoidal",
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
    ]
    lname_dict = {}
    for ii in range(len(titles)):
        lname_dict[schemes[ii]] = titles[ii]
    return lname_dict[sch]


def sch_to_color(sch):
    color_dict = {}
    for ii, sc_name in enumerate(
        [
            "hs",
            "trapz_mod",
            "trapz",
            "hs_mod",
            "hsj",
            "rk4",
            "hsj_mod",
        ]
    ):
        color_dict[sc_name] = f"C{ii}"
    sch = sch.replace("_parab", "")
    sch = sch.replace("trapz_n", "trapz_mod")
    sch = sch.replace("hsn", "hs_mod")
    return color_dict[sch]


def scheme_kwargs(sch, longlabel=False, colors_for_parab=False):
    if colors_for_parab:
        color = sch_to_color(sch)
        ls = "-"
    else:
        color = sch_to_color(sch)
        if "hs" in sch and "parab" not in sch:
            ls = "--"
        else:
            ls = "-"
    kwargs = {"marker": "o", "c": color, "ls": ls}
    if longlabel:
        kwargs["label"] = sch_to_long_label(sch)
    else:
        kwargs["label"] = sch_to_lab(sch)
    return kwargs


def set_fonts():
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.rcParams.update({"font.size": 15})


def plot_by_segments(
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


# --------------------------- Gauss Integration -------------------------------


@lru_cache(maxsize=2000)
def LG_weight(N, i, precission=20):
    Pn = legendre_poly(N, polys=True)
    Pn_d = Pn.diff()
    xi = LG(N, precission)[i]
    wi = 2 / ((1 - xi**2) * (Pn_d.eval(xi) ** 2))
    return wi


def gauss_integral(f, N, t0, t1):
    scale = t1 - t0
    points = (np.array(LG(N)) + 1) / 2
    points = t0 + scale * points
    weights = [LG_weight(N, ii) for ii in range(N)]
    _a = [weights[ii] * f(points[ii]) for ii in range(N)]
    return scale * np.sum(_a) / 2


def poly_integral(f, n_pol, t0, t1, y0=0):
    scale = t1 - t0

    points = (np.array(LG(n_pol)) + 1) / 2
    points = t0 + scale * points
    points = list(points)

    N_gauss = ceil((n_pol + 1) / 2)
    y = [gauss_integral(f, N_gauss, t0, ii) for ii in points]
    points = [
        t0,
    ] + points
    y = [
        y0,
    ] + y
    return bary_poly(points, y)


def gauss_rep_integral(f, t0, t1, n_pol, n_integ=1):
    n_pol_cauchy = n_pol + n_integ - 1
    n_gauss = ceil((n_pol_cauchy + 1) / 2)
    cauchy_f = lambda t: (t1 - t) ** (n_integ - 1) * f(t)
    return 1 / factorial(n_integ - 1) * gauss_integral(cauchy_f, n_gauss, t0, t1)
