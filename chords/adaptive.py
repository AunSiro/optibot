# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 01:01:35 2024

@author: Siro Moreno
"""

from numpy import (
    array,
    concatenate,
    ceil,
    linspace,
)

from .pseudospectral import coll_points, tau_to_t_points
from .analysis import dynamic_errors, interpolation, total_state_error
from .util import log_base

from numpy import sum as npsum
from numpy import abs as npabs
from numpy import max as npmax
from numpy import int_ as npint

from time import time


def calc_test_point_arr(res):
    new_point_str = res["point_structure"] + 1
    sch = res["scheme"]
    limits = res["t_knot_ext"]
    n_seg = res["n_segments"]
    t_list = []
    for seg_ii in range(n_seg):
        t0 = limits[seg_ii]
        t1 = limits[seg_ii + 1]
        N = new_point_str[seg_ii]
        t_list.append(
            array(tau_to_t_points(coll_points(N, sch), t0, t1), dtype="float64")
        )
    return t_list


def analyze_res_err(res, F, save_in_res=True):
    new_point_str = res["point_structure"] + 1
    n_seg = res["n_segments"]
    test_points_arr = calc_test_point_arr(res)
    _err = dynamic_errors(
        res=res,
        F=F,
        dynamics_error_mode="q",
        problem_order=2,
        scheme_order=2,
        x_interp=None,
        u_interp=None,
        n_interp=None,
        save_in_res=save_in_res,
        given_t_array=test_points_arr,
    )
    dyn_err = _err["dyn_err_interp"]
    t_err = concatenate(test_points_arr)
    dyn_err_list = []
    arr_index = 0
    for seg_ii in range(n_seg):
        int_len = new_point_str[seg_ii]
        dyn_err_list.append(dyn_err[arr_index : arr_index + int_len])
        arr_index += int_len
    if save_in_res:
        res["error"]["t_err"] = t_err
        res["error"]["t_err_list"] = test_points_arr
        res["error"]["dyn_err_interp_list"] = dyn_err_list
    return test_points_arr, dyn_err_list


def max_err(res):
    n_seg = res["n_segments"]
    err_list = res["error"]["dyn_err_interp_list"]
    test_points_arr = res["error"]["t_err_list"]
    interpolations = interpolation(
        res,
        problem_order=2,
        scheme_order=2,
        x_interp=None,
        u_interp=None,
        n_interp=None,
        save_in_res=False,
        given_t_array=test_points_arr,
    )
    q_list = interpolations["q_list"]
    q_list_max = [npmax(npabs(_qq)) for _qq in q_list]
    q_list_max = array(q_list_max, dtype="float64")
    res["error"]["max_q"] = q_list_max
    max_err_list = []
    for seg_ii in range(n_seg):
        max_err_list.append(npmax(npabs(err_list[seg_ii])))
    max_err_list = array(max_err_list, dtype="float64")
    res["error"]["max_abs_err"] = max_err_list
    max_err_rel = max_err_list / (1 + q_list_max)
    res["error"]["max_rel_err"] = max_err_rel
    return max_err_rel


def add_points(res, epsilon):
    m_err = res["error"]["max_rel_err"]
    rel_err_quot = m_err / epsilon
    return npint(ceil(log_base(rel_err_quot, res["point_structure"])))


def calc_new_struct(min_p, max_p, p_str, add_p, t_knot_ext):
    p_str = array(p_str, dtype="int")
    add_p = array(add_p, dtype="int")
    t_knot_ext = array(t_knot_ext, dtype="float64")
    assert p_str.shape == add_p.shape
    n_seg = len(p_str)
    n_p_str = p_str + add_p
    new_t_knot_ext = [
        t_knot_ext[0],
    ]
    new_p_str = []
    for seg_ii in range(n_seg):
        new_n = n_p_str[seg_ii]
        min_t = t_knot_ext[seg_ii]
        max_t = t_knot_ext[seg_ii + 1]
        if new_n <= max_p:
            new_p_str.append(new_n)
            new_t_knot_ext.append(max_t)
        else:
            div_n = int(ceil(new_n / max_p))
            new_n_each = int(ceil(new_n / div_n))
            new_n_each = max(new_n_each, min_p)
            new_knots = linspace(min_t, max_t, div_n + 1)[1:]
            new_t_knot_ext += list(new_knots)
            new_p_str += [
                new_n_each,
            ] * div_n

    return array(new_t_knot_ext[1:-1], dtype="float64"), array(new_p_str, dtype="int")


def adaptive_refinement_system(
    problem,
    F,
    scheme,
    order=2,
    problem_kwargs=None,
    initial_t_knots=None,
    initial_point_structure=array([10]),
    min_p=3,
    max_p=25,
    epsilon=1e-9,
    max_iter=100,
    max_coll_points=500,
    silent=False,
):
    ph_run = []
    point_structure = array(initial_point_structure, dtype="int")
    if initial_t_knots is None:
        t_knots = array([], dtype="int")
    else:
        t_knots = initial_t_knots

    if problem_kwargs is None:
        problem_kwargs = {
            "ini_guess": "lin",
            "solve_repetitions": 1,
            "silent": True,
            "verbose": False,
        }

    for iter in range(max_iter):
        N = npsum(point_structure)
        if not silent:
            print(
                f"iter: {iter}, t_knots: {t_knots}, point_structure: {point_structure}",
                time.strftime("%H:%M:%S ", time.localtime(time.time())),
            )
        try:
            _res = problem(
                scheme,
                N=N,
                t_knots_arr=t_knots,
                point_structure=point_structure,
                order=order,
            )
        except RuntimeError:
            point_structure = point_structure + 1
            if not silent:
                print("attempt failed, tring again")
            continue
        t_err_list, err_list = analyze_res_err(_res, F)
        max_err_rel = max_err(_res)
        print(f"\tMax rel err: {max_err_rel}")
        if npmax(max_err_rel) < epsilon:
            print("Convergencia Alcanzada!")
            ph_run.append(_res)
            break
        add_p = add_points(_res, epsilon)
        if not silent:
            print(f"\tadditional points: {add_p}")
        t_knots, point_structure = calc_new_struct(
            min_p, max_p, point_structure, add_p, _res["t_knot_ext"]
        )
        if npsum(point_structure) > max_coll_points:
            print("Máximo número de collocation points Alcanzado!")
            ph_run.append(_res)
            break
        ph_run.append(_res)

    return ph_run


def analyze_ph_run(ph_run, F, n_interp=2000):
    t_arr = linspace(0, 2, n_interp)
    _c = []
    _cpudt = []
    _iters = []
    _err_2 = []
    _err_q = []
    _N = []
    for case in ph_run:
        _c.append(case["cost"])
        _cpudt.append(case["cpudt"])
        _iters.append(case["iter_count"])
        _N.append(case["n_coll_total"])
        _errors = dynamic_errors(
            case,
            F,
            dynamics_error_mode="q",
            problem_order=2,
            scheme_order=2,
            x_interp=None,
            u_interp=None,
            n_interp=n_interp,
        )
        dyn_err_2 = _errors["dyn_err_interp"]
        dyn_err_q = _errors["compat_err_1_interp"]
        tot_dyn_err_2 = total_state_error(t_arr, dyn_err_2)
        tot_dyn_err_q = total_state_error(t_arr, dyn_err_q)
        _err_2.append(tot_dyn_err_2)
        _err_q.append(tot_dyn_err_q)
    results = {
        "cost": array(_c),
        "cpudt": array(_cpudt),
        "iter_count": array(_iters),
        "err_2_acum": array(_err_2),
        "err_q_acum": array(_err_q),
        "N_arr": array(_N),
    }
    return results
