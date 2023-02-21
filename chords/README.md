# CHORDS
**Collocation methods for second or higher order systems**

Here we offer a brief explanation of each file in 
alphabetical order and what you can expect to find in them.

- **analysis.py:** Contains functions dedicated to analysis and post-processing
of obtained solutions.

- **casadi.py:** Here there are some tools and functions dedicated to operate with and
convert to casadi objects. Also contains a version of sympy2casadi function adapted from Joris Gillis:
https://gist.github.com/jgillis/80bb594a6c8fcf55891d1d88b12b68b8

- **numpy.py:** Some tools and functions designed to operate with and
convert to numpy arrays are defined here.

- **opti.py:** Here there are functions and classes designed to create, contain and interface
easily with a casadi opti problem.

- **pseudospectral.py:** Contains functions needed to operate with pseudospectral collocations
schemes. In order to keep the best accuracy in interpolations, barycentric
formulas are constructed.

- **robots.py:** Functions designed to operate and interface with 
Robotic Toolbox models can be found here.

- **schemes.py:** Here, a collection of functions that describe different numerical
schemes, expressed explicitly and implicitly can be found. Related interpolation and 
auxiliar functions are also defined here.

- **symbolic.py:** Contains functions and classes that operate with SymPy symbolic objects.
There are handy auxiliary functions, functions that convert expressions
between notations, and classes that inherit from Lagranges Method and expand
it in various ways. 
