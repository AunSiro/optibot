#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:37:36 2022

@author: Siro Moreno

Here we will define functions designed to interface with 
Robotic Toolbox models.
"""

from sympy import symbols, simplify, lambdify
from sympy.physics.mechanics import dynamicsymbols, init_vprinting
from sympy.physics.mechanics import (
    Lagrangian,
    ReferenceFrame,
    Point,
    Particle,
    inertia,
    RigidBody,
    angular_momentum,
)
from optibot.symbolic import (
    lagrange,
    diff_to_symb,
    SimpLagrangesMethod,
    ImplicitLagrangesMethod,
)
from optibot.numpy import unpack
from numpy import zeros, array


def matrix_to_6_vect(M):
    I = zeros(6)
    I[0] = M[0, 0]
    I[1] = M[1, 1]
    I[2] = M[2, 2]
    I[3] = M[0, 1]
    I[4] = M[1, 2]
    I[5] = M[0, 2]
    return I


def arr_to_vect(v, frame):
    return v[0] * frame.x + v[1] * frame.y + v[2] * frame.z


def make_search_func(link, replacedict):
    def search(varname):
        if varname in replacedict.keys():
            return replacedict[varname]
        else:
            return getattr(link, varname)

    return search


def dhlink_rot_mod_to_symbody(
    dhlink,
    ref_frame,
    ref_point,
    N_inert,
    P_inert,
    q,
    name="Body",
    g=9.8,
    replacedict={},
):
    search = make_search_func(dhlink, replacedict)
    theta = search("theta")
    alpha = search("alpha")
    d = search("d")
    a = search("a")
    m = search("m")
    inert = matrix_to_6_vect(search("I"))

    Nint = ref_frame.orientnew("Nint", "Axis", [alpha, ref_frame.x])
    N1 = Nint.orientnew("N1", "Axis", [theta, Nint.z])
    Nbody = N1.orientnew("NB", "Axis", [q, N1.z])
    P1 = ref_point.locatenew("P1", d * N1.z + a * ref_frame.x)
    P1.set_vel(N_inert, P1.pos_from(P_inert).dt(N_inert))

    cm_pos = arr_to_vect(search("r"), Nbody)
    CM0 = P1.locatenew("CM0", cm_pos)
    CM0.set_vel(N_inert, CM0.pos_from(P_inert).dt(N_inert))
    I_0 = inertia(Nbody, *inert)
    body0 = RigidBody(name, CM0, Nbody, m, (I_0, CM0))
    body0.potential_energy = m * g * CM0.pos_from(P_inert).dot(N_inert.z)

    return body0, Nbody, P1


def robot_to_sympy(robot, replacedict_list, end_effector=None, simplif=True):

    if end_effector is None:
        pass
    else:
        raise NotImplementedError("This kind of end effector is not implemented yet.")

    N_in = ReferenceFrame("N")
    P0 = Point("P0")
    P0.set_vel(N_in, 0)

    points = [P0]
    frames = [N_in]
    bodies = []

    N_Links = len(robot.links)
    uu = list(symbols(f"u_:{ N_Links}"))
    qq = dynamicsymbols(f"q_:{ N_Links}")

    for ii, link in enumerate(robot.links):
        replacedict = replacedict_list[ii]
        if "q" in replacedict.keys():
            q = replacedict["q"]
        else:
            q = qq[ii]
        if link.mdh:
            if link.isrevolute:
                _b1, _N1, _P1 = dhlink_rot_mod_to_symbody(
                    link,
                    frames[-1],
                    points[-1],
                    N_in,
                    P0,
                    q,
                    replacedict=replacedict,
                )
            else:
                raise NotImplementedError("Only revolute links are already implemented")
        else:
            raise NotImplementedError("Only Modified DH links are already implemented")
        points.append(_P1)
        frames.append(_N1)
        bodies.append(_b1)

    Lag = Lagrangian(N_in, *bodies)

    if simplif:
        Lag = Lag.simplify()

    FL = []
    for ii in range(N_Links):
        fr = bodies[ii].frame
        FL.append((fr, uu[ii] * fr.z))
    for ii in range(N_Links - 1):
        fr = bodies[ii].frame
        FL.append((fr, -uu[ii + 1] * bodies[ii + 1].frame.z))

    LM_small = ImplicitLagrangesMethod(Lag, qq, forcelist=FL, frame=N_in)

    LM_small.robot = robot

    return LM_small


def Panda_Simp(
    end_effector=None, simplif=True, replacedict_list=[{} for j in range(7)]
):
    from sympy import pi

    hpi = pi / 2
    alpha_arr = [0, -hpi, hpi, hpi, -hpi, hpi, hpi]
    for ii, replacedict in enumerate(replacedict_list):
        replacedict["alpha"] = alpha_arr[ii]

    from roboticstoolbox.models.DH import Panda

    panda = Panda()

    # Centers of masses of Panda not yet implemented in Robotic Toolbox
    r_arr = array(
        [
            [3.875e-03, 2.081e-03, 0],
            [-3.141e-03, -2.872e-02, 3.495e-03],
            [2.7518e-02, 3.9252e-02, -6.6502e-02],
            [-5.317e-02, 1.04419e-01, 2.7454e-02],
            [-1.1953e-02, 4.1065e-02, -3.8437e-02],
            [6.0149e-02, -1.4117e-02, -1.0517e-02],
            [1.0517e-02, -4.252e-03, -4.5403e-02],
        ]
    )

    for ii, link in enumerate(panda.links):
        link.r = r_arr[ii, :]

    return robot_to_sympy(panda, replacedict_list, end_effector, simplif)
