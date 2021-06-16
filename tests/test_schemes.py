#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:15:30 2021

@author: Siro Moreno
"""
import pytest

import numpy as np
from optibot import schemes as sch


@pytest.mark.parametrize(
    "maybe_iterable, expected",
    [
        ([1, 2], True),
        ((1, 2), True),
        (np.array([1, 2]), True),
        (3, False),
        (2.5, False),
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
        (np.array([[[1, 2], [2, 1]], [[1, 2], [2, 1]]]), False),
    ],
)
def test_things_are_2D(maybe_2d, expected):
    assert sch.is2d(maybe_2d) == expected


@pytest.mark.parametrize(
    "array_like, expected_length",
    [
        ([1, 2], 2),
        ((2.5, 45), 2),
        (np.array([1, 2]), 2),
        (np.array([[1, 2, 3], [2, 1, 3]]), 2),
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
    def F(x, u, params):
        return u

    params = []
    new_F = sch.expand_F(F, mode="numpy")
    result = new_F(x_test, u_test, params)
    assert np.all(result == f_x_u)
