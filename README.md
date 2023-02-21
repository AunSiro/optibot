# CHORDS
**Collocation methods for second or higher order systems**

[![license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://raw.githubusercontent.com/AunSiro/optibot/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Chords* is a python package for trajectory optimization and control of second or higher order dynamic systems, such as robots. It provides collocation schemes specifically developed to deal with such systems.

Using our methods for second or higher order methods can present several benefits when compared with the standard technique of order reduction, such as a reduction of the dynamic error, satisfaction of the dynamics equation at the collocation points and consistent trajectories generation. Further information is provided in the related papers section.

An easy interface is provided to create Casadi opti problems and apply automatically different collocation schemes to a variety of dynamics models, including Sympy Lagrangian objects and function-defined physics, both explicit and implicit.

Local (or equispaced) collocation schemes included:
- Trapezoidal (for first or second order)
- Hermite-Simpson (for first or second order) (With and without linear action restriction versions)

Global pseudospectral collocation schemes included:
- Legendre-Gauss (for first or second order)
- Legendre-Gauss-Lobatto
- Legendre-Gauss-Radau (normal and inverse)

Future planned features include:
- Trapezoidal for order M 
- Hermite-Simpson for oder M (With and without linear action restriction versions)

The library contains a set of different tool for easy operation of this kind of systems, such as:
- Conversion from Sympy expressions to Numpy and Casadi functions with standarized calls
- Interpolation functions adapted to each collocation scheme
- Modified Lagrangian Sympy object for creating implicit formulation dynamics or constrained systems reducible to fewer dimensions
- Dynamic error analysis of the obtained solutions


## Related Papers
**Second or higher order local (or equispaced) collocation methods:**

[Collocation Methods for Second Order Systems](http://www.roboticsproceedings.org/rss18/p038.html)
```bibtex
@INPROCEEDINGS{Moreno-Martin-RSS-22, 
    AUTHOR    = {Siro Moreno-Martin AND Lluís Ros AND Enric Celaya}, 
    TITLE     = {{Collocation Methods for Second Order Systems}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2022}, 
    ADDRESS   = {New York City, NY, USA}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2022.XVIII.038} 
}
```
[Collocation methods for second and higher order systems](https://arxiv.org/abs/2302.09056) (PREPRINT)
```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.09056,
  doi = {10.48550/ARXIV.2302.09056},
  url = {https://arxiv.org/abs/2302.09056},
  author = {Moreno-Martín, Siro and Ros, Lluís and Celaya, Enric},
  title = {Collocation methods for second and higher order systems},
  publisher = {arXiv},
  year = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```

**Second Order Pseudospectral Collocation Methods:**

[A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems](https://arxiv.org/abs/2302.09036) (PREPRINT version)

[A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems](https://ieeexplore.ieee.org/document/9981255) (PUBLISHED version)
```bibtex
@INPROCEEDINGS{9981255,
author={Moreno-Martín, Siro and Ros, Lluís and Celaya, Enric},
booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
title={A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems}, 
year={2022},
volume={},
number={},
pages={13335-13340},
doi={10.1109/IROS47612.2022.9981255}}
```
