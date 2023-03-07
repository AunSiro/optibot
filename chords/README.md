# CHORDS
**Collocation methods for second or higher order systems**

Brief explanation of the contents of each file:

- **analysis.py:** Functions dedicated to analysis and post-processing
of obtained solutions.

- **casadi.py:** Tools and functions dedicated to operate with and
convert to Casadi objects. Also contains a version of the sympy2casadi function adapted from Joris Gillis:
https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8

- **numpy.py:** Tools and functions designed to operate with and
convert to NumPy arrays.

- **opti.py:** Functions and classes designed to create, contain and interface
easily with a Casadi Opti problem.

- **piecewise.py:** A collection of functions that describe different piecewise polynomial
schemes, expressed both with explicit and implicit dynamics formulations. Related interpolation and 
auxiliary functions are also defined here.

- **pseudospectral.py:** Functions needed to operate with pseudospectral collocation
schemes. In order to keep the best accuracy in interpolations, barycentric
formulas are constructed.

- **robots.py:** Functions designed to operate and interface with 
models from the Robotic Toolbox by P. Corke.

- **symbolic.py:** Functions and classes that operate with SymPy symbolic objects, 
including auxiliary functions, functions that convert expressions
between different notations, and classes that inherit from `sympy.physics.mechanics.LagrangesMethod` and expand
it in various ways. 
