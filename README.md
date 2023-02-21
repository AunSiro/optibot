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

Collocation Methods for Second Order Systems 

[Full Text](http://www.roboticsproceedings.org/rss18/p038.html)
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


Collocation methods for second and higher order systems

[Full Text (preprint)](https://arxiv.org/abs/2302.09056) 
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

A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems

[Full Text (preprint)](https://arxiv.org/abs/2302.09036) 
[Full Text (IEEE)](https://ieeexplore.ieee.org/document/9981255)
```bibtex
@INPROCEEDINGS{9981255,
    author={Moreno-Martín, Siro and Ros, Lluís and Celaya, Enric},
    booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
    title={A Legendre-Gauss Pseudospectral Collocation Method for Trajectory Optimization in Second Order Systems}, 
    year={2022},
    volume={},
    number={},
    pages={13335-13340},
    doi={10.1109/IROS47612.2022.9981255}
}
```
