# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:21:32 2020

@author: Siro Moreno
"""

import numpy as np
from numpy import sin, cos, tan


# --- Double Pendulum ---


def doub_pend_f_0(q_0, q_1, v_0, v_1, u_0, u_1, g=1, l_0=1, l_1=1, m_0=1, m_1=1):
    result = (
        l_0
        * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0 ** 2 * sin(q_0 - q_1)) - u_1)
        * cos(q_0 - q_1)
        + l_1
        * (
            -l_0
            * (
                g * m_0 * sin(q_0)
                + g * m_1 * sin(q_0)
                + l_1 * m_1 * v_1 ** 2 * sin(q_0 - q_1)
            )
            + u_0
        )
    ) / (l_0 ** 2 * l_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    return result


def doub_pend_f_1(q_0, q_1, v_0, v_1, u_0, u_1, g=1, l_0=1, l_1=1, m_0=1, m_1=1):
    result = (
        -l_0
        * (m_0 + m_1)
        * (l_1 * m_1 * (g * sin(q_1) - l_0 * v_0 ** 2 * sin(q_0 - q_1)) - u_1)
        + l_1
        * m_1
        * (
            l_0
            * (
                g * m_0 * sin(q_0)
                + g * m_1 * sin(q_0)
                + l_1 * m_1 * v_1 ** 2 * sin(q_0 - q_1)
            )
            - u_0
        )
        * cos(q_0 - q_1)
    ) / (l_0 * l_1 ** 2 * m_1 * (m_0 - m_1 * cos(q_0 - q_1) ** 2 + m_1))
    return result
