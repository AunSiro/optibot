# CHORDS
**Collocation methods for second or higher order systems**

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://raw.githubusercontent.com/AunSiro/optibot/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Chords* is a python package for trajectory optimization and control of second or higher order dynamic systems, such as robots. It provides collocation schemes specifically developed to deal with such systems.

An easy interface is provided to create Casadi opti problems and apply automatically different collocation schemes to a variety of dynamics models, including Sympy Lagrangian objects and function-defined physics, both explicit and implicit.

Local (or equispaced) collocation schemes included:
- Trapezoidal
- Trapezoidal for second order
- Hermite-Simpson (With and without linear action restriction versions)
- Hermite-Simpson for second order (With and without linear action restriction versions)

Global pseudospectral collocation schemes included:
- Lehendre-Gauss
- Lehendre-Gauss-Lobatto
- Lehendre-Gauss-Radau (normal and inverse)
- Lehendre-Gauss for second order

Future planned features include:
- Trapezoidal for order M 
- Hermite-Simpson for oder M (With and without linear action restriction versions)

The library contains a set of different tool for easy operation of this kind of systems, such as:
- Conversion from Sympy expressions to Numpy and Casadi functions with standarized calls
- Interpolation functions adapted to each collocation scheme
- Modified Lagrangian Sympy object for creating implicit formulation dynamics or constrained systems reducible to fewer dimensions
- Dynamic error analysis of the obtained solutions

