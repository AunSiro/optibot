# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:52:20 2021

@author: Siro Moreno
"""
from sympy import (
    symbols,
    pi,
    cos,
    sin,
    simplify,
    integrate,
    Eq,
    solve,
    dsolve,
    Matrix,
    preorder_traversal,
    Float,
    solve_linear_system,
    eye,
    zeros,
    lambdify,
    sqrt,
    Symbol,
    Derivative,
    Function,
)
from sympy.physics.mechanics import dynamicsymbols
from sympy.functions import sign
from sympy.physics.mechanics import LagrangesMethod, find_dynamicsymbols


def get_str(x):
    return x.__str__()


def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False
    except Exception:
        return False


def make_list(x):
    if is_iterable(x):
        return list(x)
    else:
        return [
            x,
        ]


def q_2_x(expr, qs, qdots):
    n = len(qs)
    subs_list = []
    for i in range(n):
        item = [qdots[i], dynamicsymbols("x_" + str(n + i))]
        subs_list.append(item)

    for i in range(n):
        item = [qs[i], dynamicsymbols("x_" + str(i))]
        subs_list.append(item)
    return expr.subs(subs_list)


def derivative_level(symb):
    dot_level = 0
    while type(symb) == Derivative:
        symb = symb.integrate()
        dot_level += 1
    return dot_level


def deriv_base(symb):
    while type(symb) == Derivative:
        symb = symb.integrate()
    return symb


def find_dyn_dependencies(expr):
    sym_list = find_dynamicsymbols(expr)
    sym_base_list = []
    for sym in sym_list:
        base = deriv_base(sym)
        if not base in sym_base_list:
            sym_base_list.append(base)
    return sym_base_list


def sorted_dynamic_symbols(expr):
    """
    In a given expression, finds dynamic symbols, and returns them in a
    list ordered from higher to lower derivation order.

    Parameters
    ----------
    expr : sympy expression
        DESCRIPTION.

    Returns
    -------
    dyn_vars : list
        List of found dynamic symbols, ordered from higher to lower
        derivation order

    """

    dyn_vars = find_dynamicsymbols(expr)
    dyn_vars = sorted(dyn_vars, key=derivative_level, reverse=True)
    return dyn_vars


def integerize(expr):
    expr2 = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr2 = expr2.subs(a, round(a))
    return expr2


def roundize(expr, n=4):
    expr2 = expr
    for a in preorder_traversal(expr):
        if isinstance(a, Float):
            expr2 = expr2.subs(a, round(a, n))
    return expr2


def lagrange(L_expr, var):
    """
    Parameters
    ----------
    L_expr : Symbolic Expression of Lagrange Function
        
    var : Symbol of a variable
        

    Returns
    -------
    Symbolic Expressions
        Lagrange equation respect to the variable.

    """
    t = symbols("t")
    vardot = var.diff(t)
    lag1 = L_expr.diff(vardot).simplify().diff(t).simplify()
    lag2 = L_expr.diff(var).simplify()
    lag = lag1 - lag2
    return lag.simplify().expand()


def get_lagr_eqs(T, U, n_vars):
    """
    Get a list of lagrange equations. T and U are Kinetic energy
    and Potential energy, as functions of coordinates q, its
    derivatives and other parameters

    Parameters
    ----------
    T : Symbolic Expression
        Kinetic Energy as function of q_i with i in (0,n_vars).
    U : Symbolic Expression
        Potential Energy as function of q_i with i in (0,n_vars).
    n_vars : int
        Amount of variables.

    Returns
    -------
    res : List of symbolic expressions
        List of simbolic lagrange equations.

    """
    L = T - U
    res = []
    for ii in range(n_vars):
        q = dynamicsymbols(f"q_{ii}")
        res.append(simplify(lagrange(L, q)))
    return res


def diff_to_symb(symb):
    q = symb
    dot_level = 0
    while type(q) == Derivative:
        q = q.integrate()
        dot_level += 1
    q_name = q.__str__()[:-3]
    if dot_level == 0:
        pass
    elif q_name[:2] == "q_" or (q_name[0] == "q" and q_name[1].isdigit()):
        if dot_level == 1:
            q_name = "v" + q_name[1:]
        elif dot_level == 2:
            q_name = "a" + q_name[1:]
        else:
            q_name = "a" + "_dot" * (dot_level - 2) + q_name[1:]
    else:
        q_name += "_dot" * (dot_level)

    return symbols(q_name)


def diff_to_symb_expr(expr):
    """Transform an expression with derivatives to symbols"""
    dyn_vars = sorted_dynamic_symbols(expr)
    subs_list = [[dvar, diff_to_symb(dvar)] for dvar in dyn_vars]
    new_expr = expr.subs(subs_list)
    return new_expr


def standard_notation(expr):
    """
    Finds symbols formuled as "nii" and replaces them with symbols of 
    the form n_ii, where n = [q, v, a, u, x] and ii is a number.

    Parameters
    ----------
    expr : symbolic expression

    Returns
    -------
    expr : updated symbolic expression.

    """
    var_set = expr.atoms(Symbol)
    subs_list = []
    for var in var_set:
        varname = str(var)
        tail = varname[1:]
        if tail.isnumeric():
            if varname[0] == "q":
                subs_list.append([var, symbols(f"q_{tail}")])
            elif varname[0] == "v":
                subs_list.append([var, symbols(f"v_{tail}")])
            elif varname[0] == "a":
                subs_list.append([var, symbols(f"a_{tail}")])
            elif varname[0] == "u":
                subs_list.append([var, symbols(f"u_{tail}")])
            elif varname[0] == "x":
                subs_list.append([var, symbols(f"x_{tail}")])
    expr = expr.subs(subs_list)
    return expr


def lagr_to_RHS(lagr_eqs, output_msgs=True):
    """
    Takes lagrangian equations,
    Calculates the Right Hand Side functions of
    the second order derivatives as used in optimal control

    Parameters
    ----------
    lagr_eqs : List of symbolic expressions
        Lagrange Equations.
        
    output_msgs : Bool
        Wether to print information messages during execution

    Returns
    -------
    Symbolic Matrix of Nx1 size
        Contains the symbolic expressions of the right hand side
        of system equations for the second derivative of coordinates
        as used in optimal control problems

    """
    n_var = len(lagr_eqs)
    coeff_mat = []
    acc_mat = []
    c_mat = []
    u_mat = []
    if output_msgs:
        print("Calculating matrices")
    for ii in range(n_var):
        expr = diff_to_symb_expr(lagr_eqs[ii])
        coeff_line = []
        rest = expr
        for jj in range(n_var):
            a = symbols(f"a_{jj}")
            expr2 = expr.expand().collect(a).coeff(a)
            coeff_line.append(expr2)
            rest = rest - a * expr2
        coeff_mat.append(coeff_line)
        acc_mat.append(
            [symbols(f"a_{ii}"),]
        )
        u_mat.append(
            [symbols(f"u_{ii}"),]
        )
        c_mat.append(
            [simplify(rest),]
        )
    coeff_mat = Matrix(coeff_mat)
    acc_mat = Matrix(acc_mat)
    c_mat = Matrix(c_mat)
    u_mat = Matrix(u_mat)
    if output_msgs:
        print("Inverting matrix")
    coeff_mat_inv = coeff_mat.inv()
    if output_msgs:
        print("Simplifying result expressions")
    RHS = simplify(coeff_mat_inv @ (u_mat - c_mat))
    return RHS


class SimpLagrangesMethod(LagrangesMethod):
    def __init__(
        self,
        Lagrangian,
        qs,
        forcelist=None,
        bodies=None,
        frame=None,
        hol_coneqs=None,
        nonhol_coneqs=None,
        simplif=True,
        verbose=True,
    ):
        super().__init__(
            Lagrangian, qs, forcelist, bodies, frame, hol_coneqs, nonhol_coneqs,
        )
        self._verbose = verbose
        self._simplif = simplif
        self._rhs = None
        self._rhs_reduced = None
        self._rhs_full = None
        self.R = None
        self._lambda_vec = None

    def _simplyprint(self, expr, verbose=True, simplif=True, name=""):
        if simplif:
            if verbose:
                print("simplifying " + name)
            return simplify(expr)
        else:
            return expr

    def _invsimplyprint(self, expr, verbose=True, simplif=True, name=""):
        if verbose:
            print("Generating " + name)
        expr = expr.inv()
        return self._simplyprint(expr, verbose, simplif, name)

    @property
    def mass_matrix(self):
        """Returns the mass matrix.
        Explanation
        ===========
        If the system is described by 'n' generalized coordinates
        then an n X n matrix is returned.
        """

        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        return self._m_d

    @property
    def rhs(self):
        """Returns equations that can be solved numerically.
        Parameters
        ==========
        
        """
        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        if not self._rhs is None:
            return self._rhs

        n = len(self.q)
        t = dynamicsymbols._t
        M = self._m_d
        self.Q = self.forcing
        self.q_dot = self._qdots
        coneqs = self.coneqs
        _invsimplyprint = self._invsimplyprint
        _simplyprint = self._simplyprint
        verbose = self._verbose
        simplif = self._simplif

        # print(self.coneqs,len(self.coneqs))
        if len(coneqs) > 0:
            m = len(coneqs)
            n_ind = n - m

            self.M_in = M[:n_ind, :n_ind]
            self.M_de = M[n_ind:, n_ind:]
            self.M_con = M[:n_ind, n_ind:]

            self.phi_q = self.lam_coeffs
            self.phi_q_in = self.phi_q[:, :n_ind]
            self.phi_q_de = self.phi_q[:, n_ind:]

            self.Q_in = Matrix(self.Q[:n_ind])
            self.Q_de = Matrix(self.Q[n_ind:])

            self.q_dot_in = Matrix(self.q_dot[:n_ind])

            self.phi_q_de_inv = _invsimplyprint(
                self.phi_q_de, verbose, simplif, name="Phi_q_de_inv"
            )

            self.R = _simplyprint(
                -self.phi_q_de_inv @ self.phi_q_in, verbose, simplif, name="R"
            )
            self.R_dot = self.R.diff(t)
            self.q_dot_de = self.R @ self.q_dot_in
            H_con = self.M_con @ self.R
            H = self.M_in + H_con + H_con.T + self.R.T @ self.M_de @ self.R
            self.H = _simplyprint(H, verbose, simplif, name="H")
            K = self.R.T @ self.M_de @ self.R_dot + self.M_con @ self.R_dot
            self.K = _simplyprint(K, verbose, simplif, name="K")
            Fa = self.Q_in + self.R.T @ self.Q_de
            self.Fa = _simplyprint(Fa, verbose, simplif, name="Fa")

            h_inv = _invsimplyprint(self.H, verbose, simplif, name="H_inv")
            q_dotdot_in = h_inv @ (self.Fa - self.K @ self.q_dot_in)
            self.q_dotdot_in = _simplyprint(
                q_dotdot_in, verbose, simplif, name="RHS_in"
            )

            dyn_deps = sum(
                [find_dyn_dependencies(expr) for expr in self.q_dotdot_in], start=[]
            )
            q_de = self.q[n_ind:]

            if all([not symb in dyn_deps for symb in q_de]):
                if verbose:
                    print("Reduced model found and completed")
                self._rhs_reduced = Matrix(list(self.q_dot_in) + list(self.q_dotdot_in))
                self._rhs = self._rhs_reduced
                return self._rhs
            else:
                if verbose:
                    print(
                        "Dependencies found in simplified model on dependent coordinates,"
                    )
                    print("Calculating complete model.")
                self.calculate_RHS_complete()
                return self._rhs
        else:
            M_inv = _invsimplyprint(self._m_d, verbose, simplif, name="M_inv")
            self.q_dotdot = _simplyprint(M_inv @ self.Q, verbose, simplif, name="RHS")
            if verbose:
                print("Model completed")
            self._rhs = Matrix(list(self.q_dot) + list(self.q_dotdot))
            return self._rhs

    def calculate_RHS_complete(self):
        verbose = self._verbose
        simplif = self._simplif
        _simplyprint = self._simplyprint

        if self.R is None:
            self.rhs

        if (self._rhs is None) or (len(self.coneqs) > 0):
            q_dotdot_de = self.R_dot @ self.q_dot_in + self.R @ self.q_dotdot_in

            self.q_dotdot_de = _simplyprint(
                q_dotdot_de, verbose, simplif, name="Dependent Variables"
            )
            self._rhs = Matrix(
                list(self.q_dot) + list(self.q_dotdot_in) + list(self.q_dotdot_de)
            )

        return self._rhs

    @property
    def rhs_reduced(self):
        """Returns equations that can be solved numerically.
        Parameters
        ==========
        
        """
        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        if self._rhs is None:
            self.rhs
        if self._rhs_reduced is None:
            raise ValueError(
                "System could not be reduced to a lower number of variables, use rhs instead"
            )
        else:
            return self._rhs_reduced

    @property
    def lambda_vector(self):
        if self._lambda_vec is None:
            if len(self.coneqs) == 0:
                self._lambda_vec = []
            else:
                if self._verbose:
                    print("Generating and simplifiying Lambdas vector")
                self._lambda_vec = simplify(
                    self.phi_q_de_inv @ (self.Q_de - self.M_de @ self.q_dotdot_de)
                )
        return self._lambda_vec

    @property
    def rhs_full(self):
        """Returns equations that can be solved numerically.
        Parameters
        ==========
        
        """
        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        if self._rhs_full is None:
            self.calculate_RHS_complete()

            self._rhs_full = Matrix(list(self.rhs) + list(self.lambda_vector))
        return self._rhs_full


class ImplicitLagrangesMethod(LagrangesMethod):
    @property
    def mass_matrix_square(self):
        """Augments the coefficients of restrictions to the mass_matrix.
        Returns:
            | M    A_c|
            |m_cd   0 |
        So that the dynamics can be written as 
            | M    A_c|   | q''  |   |f_d |
            |         | @ |      | = |    |
            |m_cd   0 |   |lambda|   |f_dc|
        """

        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        m = len(self.coneqs)
        row1 = self.mass_matrix
        if self.coneqs:
            row2 = self._m_cd.row_join(zeros(m, m))
            return row1.col_join(row2)
        else:
            return row1

    @property
    def implicit_dynamics_q(self):
        """Returns a vector of implicit dynamics.
        Given that the dynamics can be written as:
            | M    A_c|   | q''  |   |f_d |
            |         | @ |      | = |    |
            |m_cd   0 |   |lambda|   |f_dc|
        Returns a vector D equal to:
            | M    A_c|   | q''  |   |f_d |
        D = |         | @ |      | - |    |
            |m_cd   0 |   |lambda|   |f_dc|
        so that the dynamics can be defined as:
            D(q, q', q'', u, lambdas, params) = 0
        """
        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        M = self.mass_matrix_square
        if self.coneqs:
            F = self.forcing.col_join(self._f_cd)
            Q_exp = self._qdoubledots.col_join(self.lam_vec)
        else:
            F = self.forcing
            Q_exp = self._qdoubledots

        return M @ Q_exp - F

    @property
    def implicit_dynamics_x(self):
        """Returns a vector of implicit dynamics.
        Given that the dynamics can be written as:
            | M    A_c|   | q''  |   |f_d |
            |         | @ |      | = |    |
            |m_cd   0 |   |lambda|   |f_dc|
        And transforming the variables q and v into x:
                | q |   | x_q |
            x = |   | = |     |
                | v |   | x_v |
        Expressing the dynammics in terms of x:
            | I   0    0 |   |      |   |x_v |
            |            |   | x'   |   |    |
            | 0   M   A_c| @ |      | = |f_d |
            |            |   |      |   |    |
            | 0  m_cd  0 |   |lambda|   |f_dc|
            
        Function returns a vector D equal to:
            | I   0    0 |   |      |   |x_v |
            |            |   | x'   |   |    |
        D = | 0   M   A_c| @ |      | - |f_d |
            |            |   |      |   |    |
            | 0  m_cd  0 |   |lambda|   |f_dc|
        so that the dynamics can be defined as:
            D(x, x', u, lambdas, params) = 0
        """
        if self.eom is None:
            raise ValueError("Need to compute the equations of motion first")
        M = q_2_x(self.mass_matrix_full, self.q, self._qdots)
        n = len(self.q)
        X = Matrix(dynamicsymbols("x_0:" + str(2 * n)))
        X_dot = X.diff(dynamicsymbols._t)
        forcing = q_2_x(self.forcing, self.q, self._qdots)
        if self.coneqs:
            f_cd = q_2_x(self._f_cd, self.q, self._qdots)
            F = Matrix(X[n:]).col_join(forcing).col_join(f_cd)
            X_exp = X_dot.col_join(self.lam_vec)
        else:
            F = Matrix(X[n:]).col_join(forcing)
            X_exp = X_dot

        return M @ X_exp - F


def add_if_not_there(x, container):
    if not x in container:
        container.append(x)


def is_lambda(symb):
    name = symb.__str__()
    if name[-3:] == "(t)":
        name = name[:-3]
    if name[:3] == "lam" and name[3:].isdigit():
        return True
    else:
        return False


def find_arguments(
    expr_list,
    q_vars,
    u_vars=None,
    separate_lambdas=False,
    separate_as=False,
    verbose=False,
):
    """
    Given an iterable of sympy expressions, search in it looking for 
    certain symbols.
    

    Parameters
    ----------
    expr_list : iterable of sympy expressions
        Iterable whose symbols will be extracted
    q_vars : int or list of dynamic symbols
        Determine the symbols that will be searched
        if int, the program will assume q as q_i for q in [0,q_vars]
    u_vars : None, in or list of symbols. Default is None.
        Symbols that will be sarched and separated. 
        If None, symbols of the form u_ii where ii is a number will be 
        assumed
    separate_lambdas : Bool, optional
        Wether to separate symbols of the form "lamNN" from the rest of 
        parameters, where NN is a number. The default is False.
    separate_as : Bool, optional
        Wether to separate symbols of the form "a_NN" from the rest of 
        parameters, where NN is a number. The default is False.
    verbose : Bool, optional
        wether to print aditional information of expected and found variables
        in the given expression

    Raises
    ------
    TypeError
        If inputs are different types than supported.

    Returns
    -------
    q_args : list of symbols
        symbols equal to q_vars if q_var is a list, or symbols determined from
        q_var id q_var is an integer
    v_args : list of symbols
        if q_args are symbols of the form q_ii, where ii is a number, 
        v_args will be a list o symbols of the form v_ii where ii is the same
        numbers in q_args.
        if not, v_args will be a list of symbols of the form "S_dot", where 
        S is the name of each symbol in q_args.
    a_args : list of symbols
        if separate_as is True:
            If elements of q_args are q_ii, will be a list of symbols a_ii where ii is 
            a number, the same as in q_args.
            If not, will be a list of symbols of the form "S_dot_dot", where 
            S is the name of each symbol in q_args.
        else, will be an empty list
    u_args : list os symbols
        If u_vars is a list of symbols, will be the same list.
        If u_vars is an int, will be a list of symbols of the form u_ii where 
        ii is a number in [0, u_vars]
        If u_vars is None (default), will be a list of all found symbols
        of the form u_ii where ii is a number
    params : List of symbols
       A list of all the symbols in the expressions that don't fit in any 
       other category
    lambda_args_found : list of symbols
        If separate_lambdas is True, a list of all found symbols
        of the form lamii where ii is a number
        If not, will be an empty list

    """

    expr_list = list(expr_list)
    expr_list = [standard_notation(diff_to_symb_expr(expr)) for expr in expr_list]
    max_n_var = sum([len(expr.atoms(Symbol)) for expr in expr_list])
    params = []
    args = []
    u_args_found = []
    x_args_found = []
    a_args_found = []
    lambda_args_found = []

    if type(q_vars) == int:
        q_args = list(symbols(f"q_0:{q_vars}"))
        v_args = list(symbols(f"v_0:{q_vars}"))
        a_args = list(symbols(f"a_0:{q_vars}"))
    elif type(q_vars) == list:
        q_args = [diff_to_symb(var) for var in q_vars]
        v_args = [diff_to_symb(var.diff()) for var in q_vars]
        a_args = [diff_to_symb(var.diff().diff()) for var in q_vars]
    else:
        raise TypeError(
            "data type not undersood for q_vars, must be an integer or list of symbols"
        )

    if u_vars is None:
        u_args = list(symbols("u_0:" + str(max_n_var)))
    elif type(u_vars) == list:
        u_args = u_vars
    elif type(u_vars) == int:
        u_args = list(symbols("u_0:" + str(u_vars)))
    else:
        raise TypeError(
            "data type not undersood for u_vars, must be None, an integer or list of symbols"
        )

    x_args = q_args + v_args
    args = x_args + u_args + a_args
    for ii in range(len(expr_list)):
        expr = expr_list[ii]
        var_set = expr.atoms(Symbol)
        for symb in var_set:
            if not symb in args:
                if is_lambda(symb):
                    add_if_not_there(symb, lambda_args_found)
                else:
                    add_if_not_there(symb, params)
            elif symb in u_args:
                add_if_not_there(symb, u_args_found)
            elif symb in x_args:
                add_if_not_there(symb, x_args_found)
            elif symb in a_args:
                add_if_not_there(symb, a_args_found)

    lambda_args_found = sorted(lambda_args_found, key=get_str)
    u_args_found = sorted(u_args_found, key=get_str)
    x_args_found = sorted(x_args_found, key=get_str)
    a_args_found = sorted(a_args_found, key=get_str)

    if verbose:
        print("x vars expected:", x_args)
        print("x vars found:", x_args_found)
        print("u vars found:", u_args_found)
        if separate_lambdas:
            print("Lambda variables are separated from parameters")
            print("lambda vars found:", lambda_args_found)
        else:
            print("Lambda variables are not separated from parameters")
        if separate_as:
            print("a variables are separated from parameters")
            print("a vars expected:", a_args)
            print("a vars found:", a_args_found)
        else:
            print("a variables are not separated from parameters")
    if u_vars is None:
        u_args = sorted(u_args_found, key=get_str)
    if not separate_lambdas:
        params += lambda_args_found
        lambda_args_found = []
    if not separate_as:
        params += a_args_found
        a_args = []
    params = sorted(params, key=get_str)
    if verbose:
        print("Parameters found:", params)

    return q_args, v_args, a_args, u_args, params, lambda_args_found


def printer_function(flavour):
    if flavour == "numpy":
        from sympy.printing.numpy import NumPyPrinter

        def printer(x):
            np_printer = NumPyPrinter()
            return np_printer.doprint(x)

        return printer
    elif flavour == "np":
        from sympy.printing.numpy import NumPyPrinter
        from re import sub

        def printer(x):
            np_printer = NumPyPrinter()
            x_out = np_printer.doprint(x)
            return sub("numpy", "np", x_out)

        return printer
    elif flavour == "casadi":
        from sympy.printing.numpy import NumPyPrinter
        from re import sub

        def printer(x):
            np_printer = NumPyPrinter()
            x_out = np_printer.doprint(x)
            return sub("numpy", "cas", x_out)

        return printer
    else:
        return str


def print_funcs_RHS(RHS, q_vars, u_vars=None, flavour="np", verbose=False):
    """
    Prints the Right Hand Side of the control ecuations, formatted
    to be used as a python function to solve a system like:
        x' = F(x, u, params)

    Parameters
    ----------
    RHS : Matrix or list of symbolic expressions
        
    q_vars : int or list of symbols
        Number of q variables or list of q variables as symbols.
        If int, will search variables of form q_i
        
    u_vars : None or list of symbols
        Number of u variables or list of q variables as symbols.
        If None, will search variables of form u_i
    
    flavour : str in ["numpy", "np", "casadi"], default = "np"
        experimental feature, converts common functions like sin(x)
        to numpy.sin(x), np.sin(x) or cas.sin(x) respectively
    verbose : Bool, default = False
        wether to print aditional information of expected and found variables
        in the given expression

    Returns
    -------
    string
        when outputted by print(), can be copypasted to define a function
        associated with RHS: x' = F(x, u, params)

    """
    RHS = make_list(RHS)
    RHS = [standard_notation(diff_to_symb_expr(expr)) for expr in RHS]
    arguments = find_arguments(RHS, q_vars, u_vars, verbose=verbose)
    q_args, v_args, a_args, u_args_found, params, lambda_args = arguments
    x_args = q_args + v_args

    printer = printer_function(flavour)

    msg = "def F(x, u, params):\n"
    msg += f"    {x_args.__str__()[1:-1]} = unpack(x)\n"
    msg += f"    {u_args_found.__str__()[1:-1]} = unpack(u)\n"
    msg += f"    {params.__str__()[1:-1]} = params\n"
    msg += f"    result = [{v_args.__str__()[1:-1]},]\n"
    for expr in RHS:
        msg += "    result.append(" + printer(expr) + ")\n"
    msg += "\n    return result\n"

    print(msg)
    return msg


def print_funcs(expr_list, q_vars=0, flavour="np", verbose=False):
    """
    Prints the given expression list or matrix as a function of x-variables,
    u-variables and parameters. X-variables are either the dynamic symbols
    of q_vars and their derivatives, or variables of the form q_i or v_i 
    detected in the expressions up until i = q_vars if q_vars is an integer.

    Parameters
    ----------
    expr_list : Matrix or list of Sympy symbolic expressions
        
    q_vars : int or list of symbols, default = 0
        Number of q variables or list of q variables as symbols.
        If int, will search variables of form q_i and qi
        If set to 0, all detected variables will be considered parameters
    
    flavour : str in ["numpy", "np", "casadi"], default = "np"
        experimental feature, converts common functions like sin(x)
        to numpy.sin(x), np.sin(x) or cas.sin(x) respectively
    verbose : Bool, default = False
        wether to rint aditional information of expected and found variables
        in the given expression

    Returns
    -------
    string
        when outputted by print(), can be copypasted to define a function
        associated with RHS: x' = F(x, u, params)

    """
    expr_list = make_list(expr_list)
    expr_list = [standard_notation(diff_to_symb_expr(expr)) for expr in expr_list]
    arguments = find_arguments(expr_list, q_vars, verbose=verbose)
    q_args, v_args, a_args, u_args_found, params, lambda_args = arguments
    x_args = q_args + v_args

    printer = printer_function(flavour)

    msg = "def F("
    if len(x_args) > 0:
        msg += "x, "
    if len(u_args_found) > 0:
        msg += "u, "
    if len(params) > 0:
        msg += "params, "
    msg += "):\n"
    if len(x_args) > 0:
        msg += f"    {x_args.__str__()[1:-1]} = unpack(x)\n"
    if len(u_args_found) > 0:
        msg += f"    {u_args_found.__str__()[1:-1]} = unpack(u)\n"
    if len(lambda_args) > 0:
        msg += f"    {lambda_args.__str__()[1:-1]} = unpack(lambda)\n"
    if len(params) > 0:
        msg += f"    {params.__str__()[1:-1]} = params\n"
    if len(expr_list) == 1:
        msg += "    result = " + printer(expr_list[0]) + "\n"
    else:
        msg += "    result = []\n"
        for expr in expr_list:
            msg += "    result.append(" + printer(expr) + ")\n"
    msg += "\n    return result\n"

    print(msg)
    return msg
