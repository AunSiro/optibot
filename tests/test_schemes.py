#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:15:30 2021

@author: Siro Moreno
"""
import pytest

import numpy as np
import casadi as cas
from optibot import schemes as sch


@pytest.mark.parametrize(
    "maybe_iterable, expected",
    [
        ([1, 2], True),
        ((1, 2), True),
        (np.array([1, 2]), True),
        (3, False),
        (2.5, False),
        (cas.DM([1, 2]), False),
    ],
)
def test_things_are_iterable(maybe_iterable, expected):
    assert sch.is_iterable(maybe_iterable) == expected


@pytest.mark.parametrize(
    "maybe_2d, expected",
    [
        ([1, 2], False),
        (1, False),
        ("hello, i am a test, thank you for reading", False),
        ((2.5, 45), False),
        (np.array([1, 2]), False),
        (np.array([[1, 2], [2, 1]]), True),
        (cas.DM([[1, 2], [2, 1]]), True),
        (np.array([[[1, 2], [2, 1]], [[1, 2], [2, 1]]]), False),
    ],
)
def test_things_are_2D(maybe_2d, expected):
    assert sch.is2d(maybe_2d) == expected


@pytest.mark.parametrize(
    "array_like, expected_length",
    [
        (1, 1),
        (1.0, 1),
        (np.array(1), 1),
        ([1, 2], 2),
        ((2.5, 45), 2),
        (np.array([1, 2]), 2),
        (cas.DM([1, 2]), 2),
        (np.array([[1, 2, 3], [2, 1, 3]]), 2),
        (cas.DM([[1, 2, 3], [2, 1, 3]]), 2),
        ([[1, 2, 3], [2, 1, 3]], 2),
    ],
)
def test_length_of_things(array_like, expected_length):
    assert sch.vec_len(array_like) == expected_length


def test_interp_2d():
    old_t_array = np.array([1.0, 2.0])
    t_array = np.array([1.0, 1.5, 2.0])
    Y = np.array([[0.0, 0.0], [1.0, 1.0]])
    expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    result = sch.interp_2d(t_array, old_t_array, Y)
    assert np.all(result == expected)


@pytest.mark.parametrize(
    "array_case, expected",
    [
        (np.array([1, 2]), np.array([1, 2, 2])),
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4], [3, 4]])),
    ],
)
def test_extend_array(array_case, expected):
    assert np.all(sch.extend_array(array_case) == expected)


@pytest.mark.parametrize(
    "x_test, u_test, f_x_u",
    [
        ([1.0, 2.0], 3.0, np.array([2.0, 3.0])),
        (np.array([1.0, 2.0]), 3.0, np.array([2.0, 3.0])),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([5.0, 6.0]),
            np.array([[2.0, 5.0], [4.0, 6.0]]),
        ),
        (
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([3.0, 4.0, 5.0, 6.0]),
        ),
        (
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            np.array([5.0, 6.0]),
            np.array([[3.0, 4.0, 5.0, 6.0]]),
        ),
    ],
)
def test_expand_F_numpy(x_test, u_test, f_x_u):
    F = sch.expand_F(lambda x, u, params: u, mode="numpy")
    params = []
    result = F(x_test, u_test, params)
    assert np.all(result == f_x_u)


@pytest.mark.parametrize(
    "x_test, u_test, f_x_u",
    [
        ([1.0, 2.0], 3.0, cas.DM([2.0, 3.0])),
        (cas.DM([1.0, 2.0]), 3.0, cas.DM([2.0, 3.0])),
        (
            cas.DM([[1.0, 2.0], [3.0, 4.0]]),
            cas.DM([5.0, 6.0]),
            cas.DM([[2.0, 5.0], [4.0, 6.0]]),
        ),
        (
            cas.DM([1.0, 2.0, 3.0, 4.0]),
            cas.DM([5.0, 6.0]),
            cas.DM([3.0, 4.0, 5.0, 6.0]),
        ),
        (
            cas.DM([[1.0, 2.0, 3.0, 4.0]]),
            cas.DM([5.0, 6.0]),
            cas.DM([[3.0, 4.0, 5.0, 6.0]]),
        ),
    ],
)
def test_expand_F_casadi(x_test, u_test, f_x_u):
    F = sch.expand_F(lambda x, u, params: u, mode="casadi")
    params = []
    result = F(x_test, u_test, params)
    assert np.all(result == f_x_u)


@pytest.mark.parametrize(
    "mode", ["numpy", "casadi"],
)
def test_expand_F_keeps_doc(mode):
    def F(x, u, params):
        """This is a documentation"""
        return u

    new_F = sch.expand_F(F, mode)
    assert F.__doc__ in new_F.__doc__


# --- Integration Steps ---


def generate_step_parameters(scheme):
    x_0_opts = [
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0, 0.0]),
    ]
    u_opts = [
        np.array([1.0,]),
        np.array([-1.0,]),
        np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]),
    ]
    u_n_opts = [
        np.array([2.0,]),
        np.array([-2.0,]),
        np.array([2.0, -2.0]),
        np.array([-2.0, 2.0]),
    ]
    results_euler = [
        np.array([0.5, 1.5]),
        np.array([3.5, 2.5]),
        np.array([1.0, 2.5, 2.5, 2.5]),
        np.array([3.5, 2.0, 0.5, 0.5]),
    ]
    results_rk4 = [
        np.array([0.625, 1.5]),
        np.array([3.375, 2.5]),
        np.array([1.125, 2.375, 2.5, 2.5]),
        np.array([3.375, 2.125, 0.5, 0.5]),
    ]
    results_trapz = [
        np.array([0.6875, 1.75]),
        np.array([3.3125, 2.25]),
        np.array([1.1875, 2.3125, 2.75, 2.25]),
        np.array([3.3125, 2.1875, 0.25, 0.75]),
    ]
    results_trapz_mod = [
        np.array([0.6666, 1.75]),
        np.array([3.3333, 2.25]),
        np.array([1.1666, 2.3333, 2.75, 2.25]),
        np.array([3.3333, 2.1666, 0.25, 0.75]),
    ]
    results_hs = [
        np.array([0.6666, 1.75]),
        np.array([3.3333, 2.25]),
        np.array([1.1666, 2.3333, 2.75, 2.25]),
        np.array([3.3333, 2.1666, 0.25, 0.75]),
    ]
    results_hs_mod = [
        np.array([0.6666, 1.75]),
        np.array([3.3333, 2.25]),
        np.array([1.1666, 2.3333, 2.75, 2.25]),
        np.array([3.3333, 2.1666, 0.25, 0.75]),
    ]
    if scheme == "euler":
        return [
            (x_0_opts[ii], u_opts[ii], results_euler[ii]) for ii in range(len(x_0_opts))
        ]
    elif scheme == "rk4":
        return [
            (x_0_opts[ii], u_opts[ii], results_rk4[ii]) for ii in range(len(x_0_opts))
        ]
    elif scheme == "trapz":
        return [
            (x_0_opts[ii], u_opts[ii], u_n_opts[ii], results_trapz[ii])
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "trapz_mod":
        return [
            (x_0_opts[ii], u_opts[ii], u_n_opts[ii], results_trapz_mod[ii])
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "hs":
        return [
            (x_0_opts[ii], u_opts[ii], u_n_opts[ii], results_hs[ii])
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "hs_mod":
        return [
            (x_0_opts[ii], u_opts[ii], u_n_opts[ii], results_hs_mod[ii])
            for ii in range(len(x_0_opts))
        ]
    else:
        raise ValueError(f"Unrecognized scheme: {scheme}")


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_step_parameters("euler"),
)
def test_euler_step(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.euler_step(x_0, u, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_step_parameters("rk4"),
)
def test_rk4_step(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.rk4_step(x_0, u, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, u_n, expected_result", generate_step_parameters("trapz"),
)
def test_trapz_step(x_0, u, u_n, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.trapz_step(x_0, u, u_n, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, u_n, expected_result", generate_step_parameters("trapz_mod"),
)
def test_trapz_mod_step(x_0, u, u_n, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.trapz_mod_step(x_0, u, u_n, F, dt, params)
    assert np.all(np.abs(result - expected_result) < 0.0002)


@pytest.mark.parametrize(
    "x_0, u, u_n, expected_result", generate_step_parameters("hs"),
)
def test_hs_step(x_0, u, u_n, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.hs_step(x_0, u, u_n, F, dt, params)
    assert np.all(np.abs(result - expected_result) < 0.0002)


@pytest.mark.parametrize(
    "x_0, u, u_n, expected_result", generate_step_parameters("hs_mod"),
)
def test_hs_mod_step(x_0, u, u_n, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.hs_mod_step(x_0, u, u_n, F, dt, params)
    assert np.all(np.abs(result - expected_result) < 0.0002)


# --- Array Integrations ---


def generate_array_integration_parameters(scheme):
    x_0_opts = [
        [0.0, 1.0],
        [0.0, 1.0],
        [[0.0, 1.0],],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    u_opts = [
        1.0,
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [[0.0, 1.0], [2.0, 3.0]],
    ]
    results_euler = [
        [[0.0, 1.0], [0.5, 1.5]],
        [[0.0, 1.0], [0.5, 1.0], [1.0, 1.5]],
        [[0.0, 1.0], [0.5, 1.0], [1.0, 1.5]],
        [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.5]],
        [[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.5], [1.0, 1.25, 2.0, 3.0]],
    ]
    results_rk4 = [
        [[0.0, 1.0,], [0.625, 1.5,],],
        [[0.0, 1.0,], [0.5, 1.0,], [1.125, 1.5,],],
        [[0.0, 1.0,], [0.5, 1.0,], [1.125, 1.5,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,], [1.25, 1.75, 2.0, 3.0,],],
    ]
    results_trapz = [
        [[0.0, 1.0,], [0.625, 1.5,],],
        [[0.0, 1.0,], [0.5625, 1.25,], [1.3125, 1.75,],],
        [[0.0, 1.0,], [0.5625, 1.25,], [1.3125, 1.75,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.625, 0.75, 1.5, 2.0,], [1.625, 2.125, 2.5, 3.5,],],
    ]
    results_trapz_mod = [
        [[0.0, 1.0,], [0.625, 1.5,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,],],
        [
            [0.0, 0.0, 1.0, 1.0,],
            [0.5833, 0.7083, 1.5, 2.0,],
            [1.5833, 2.0833, 2.5, 3.5,],
        ],
    ]
    results_hs = [
        [[0.0, 1.0,], [0.625, 1.5,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,],],
        [
            [0.0, 0.0, 1.0, 1.0,],
            [0.5833, 0.7083, 1.5, 2.0,],
            [1.5833, 2.0833, 2.5, 3.5,],
        ],
    ]
    results_hs_mod = [
        [[0.0, 1.0,], [0.625, 1.5,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 1.0,], [0.5416, 1.25,], [1.2916, 1.75,],],
        [[0.0, 0.0, 1.0, 1.0,], [0.5, 0.625, 1.0, 1.5,],],
        [
            [0.0, 0.0, 1.0, 1.0,],
            [0.5833, 0.7083, 1.5, 2.0,],
            [1.5833, 2.0833, 2.5, 3.5,],
        ],
    ]
    if scheme == "euler":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_euler[ii]))
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "rk4":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_rk4[ii]))
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "trapz":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_trapz[ii]))
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "trapz_mod":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_trapz_mod[ii]))
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "hs":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_hs[ii]))
            for ii in range(len(x_0_opts))
        ]
    elif scheme == "hs_mod":
        return [
            (x_0_opts[ii], u_opts[ii], np.array(results_hs_mod[ii]))
            for ii in range(len(x_0_opts))
        ]
    else:
        raise ValueError(f"Unrecognized scheme: {scheme}")


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("euler"),
)
def test_integrate_euler(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_euler(x_0, u, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("rk4"),
)
def test_integrate_rk4(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_rk4(x_0, u, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("trapz"),
)
def test_integrate_trapz(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_trapz(x_0, u, F, dt, params)
    assert np.all(result == expected_result)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("trapz_mod"),
)
def test_integrate_trapz_mod(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_trapz_mod(x_0, u, F, dt, params)
    assert np.all((result - expected_result) < 0.0002)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("hs"),
)
def test_integrate_hs(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_hs(x_0, u, F, dt, params)
    assert np.all((result - expected_result) < 0.0002)


@pytest.mark.parametrize(
    "x_0, u, expected_result", generate_array_integration_parameters("hs_mod"),
)
def test_integrate_hs_mod(x_0, u, expected_result):
    F = sch.expand_F(lambda x, u, params: u)
    dt = 0.5
    params = []
    result = sch.integrate_hs_mod(x_0, u, F, dt, params)
    assert np.all((result - expected_result) < 0.0002)
