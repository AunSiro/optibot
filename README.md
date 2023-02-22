# CHORDS
**Collocation methods for second or higher order systems**

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://raw.githubusercontent.com/AunSiro/optibot/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Chords* is a python package for trajectory optimization on second or higher order systems. A simple interface is provided to create Casadi opti problems and to apply different collocation schemes to dynamical models described by user-defined functions or via Sympy Lagrangian objects.

The package implements direct collocation methods that (1) respect the differential relationships between the state variables and (2) ensure that the actual dynamics is imposed at the collocation points, two properties that are not guaranteed when conventional collocation methods are applied to such systems. In most cases, the new methods in Chords reduce significantly the dynamic error of the trajectories, without noticeably increasing the cost of solving the associated NLP problems.

The following collocation schemes are implemented in Chords.

Piecewise polynomial schemes:

- Trapezoidal
- Hermite-Simpson

Pseudospectral schemes:

- Legendre-Gauss
- Legendre-Gauss-Lobatto
- Legendre-Gauss-Radau

For the piecewise polynomial schemes, and for the pseudospectral Legendre-Gauss scheme, the package implements the methods for second order systems as described in [1,2,3]. Conventional versions for first order systems are available for all methods. In the future, we plan to add the trapezoidal and Hermite-Simpson methods for Mth order systems given in [2].

The library also contains tools for:

- Conversion from Sympy expressions to Numpy and Casadi
  functions that use standardized calls
- Interpolation functions adapted to each collocation scheme
- Creation of implicit dynamics formulations through
  modified Lagrangian Sympy objects
- Reduction of dimensionality in dynamical systems
  under certain conditions
- Dynamic error analysis of the obtained solutions


## Related Papers
**Second or higher order piecewise polynomial collocation methods:**

1. Siro Moreno-Martín, Lluís Ros and Enric Celaya,
"Collocation Methods for Second Order Systems",
*XVIII Robotics: Science and Systems Conference, 2022, New York, pp. 1-11.*  
[<ins>Full Text</ins>](http://www.roboticsproceedings.org/rss18/p038.html)
[<ins>Bibtex</ins>](https://raw.githubusercontent.com/AunSiro/optibot/main/bibtex/Collocation-Moreno-RSS22.bib)


2. Siro Moreno-Martín, Lluís Ros and Enric Celaya,
"Collocation methods for second and higher order systems",
preprint  
[<ins>Full Text (preprint)</ins>](https://arxiv.org/abs/2302.09056) 
[<ins>Bibtex</ins>](https://raw.githubusercontent.com/AunSiro/optibot/main/bibtex/Collocation-Moreno-preprint23.bib)

**Second Order Pseudospectral Collocation Methods:**

3. Siro Moreno-Martín, Lluís Ros and Enric Celaya,
 "A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems",
 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Kyoto, Japan, 2022, pp. 13335-13340  
[<ins>Full Text (preprint)</ins>](https://arxiv.org/abs/2302.09036) 
[<ins>Full Text (IEEE)</ins>](https://ieeexplore.ieee.org/document/9981255)
[<ins>Bibtex</ins>](https://raw.githubusercontent.com/AunSiro/optibot/main/bibtex/Pseudospectral-Moreno-IROS22.bib)
