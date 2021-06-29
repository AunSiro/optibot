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
from sympy.physics.mechanics import LagrangesMethod


def get_str(x):
    return x.__str__()


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


def sorted_dynamic_symbols(expr):
    """
    In a given expression, finds dynamic symbols, understanding them as either
    functions that only depend on t or derivatives of such functions.

    Parameters
    ----------
    expr : sympy expression
        DESCRIPTION.

    Returns
    -------
    dyn_vars : list
        List of found dynamic symbols, ordered from biggest to smallest
        derivation order

    """
    t = symbols("t")
    dyn_vars = []
    func_set = expr.atoms(Function)
    deriv_set = expr.atoms(Derivative)
    for func in func_set:
        if func.args == (t,):
            dyn_vars.append(func)
    for deriv in deriv_set:
        if deriv_base(deriv).args == (t,):
            dyn_vars.append(deriv)
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
    var_set = expr.atoms(Symbol)
    n_vars_max = len(var_set)
    subs_list = []
    for jj in range(n_vars_max):
        subs_list.append([symbols(f"q{jj}"), symbols(f"q_{jj}")])
        subs_list.append([symbols(f"v{jj}"), symbols(f"v_{jj}")])
        subs_list.append([symbols(f"a{jj}"), symbols(f"a_{jj}")])
        subs_list.append([symbols(f"u{jj}"), symbols(f"u_{jj}")])
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


class SimpLagrangesMethod:
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
        print_status=True,
    ):

        self.print_status = print_status
        self.LM = LagrangesMethod(
            Lagrangian, qs, forcelist, bodies, frame, hol_coneqs, nonhol_coneqs
        )
        if print_status:
            print("Generating Lagrange Equations")
        self.LM.form_lagranges_equations()

        n = len(qs)
        t = symbols("t")
        self.M = self.LM.mass_matrix[:, :n]
        self.Q = self.LM.forcing
        self.q_dot = self.LM.q.diff(t)
        self.forcelist = forcelist
        self.coneqs = self.LM.coneqs

        # print(self.coneqs,len(self.coneqs))
        if len(self.coneqs) > 0:
            m = len(self.coneqs)
            n_ind = n - m

            self.M = self.LM.mass_matrix[:, :n]
            self.M_in = self.M[:n_ind, :n_ind]
            self.M_de = self.M[n_ind:, n_ind:]

            self.phi_q = self.LM.lam_coeffs
            self.phi_q_in = self.phi_q[:, :n_ind]
            self.phi_q_de = self.phi_q[:, n_ind:]

            self.Q_in = Matrix(self.Q[:n_ind])
            self.Q_de = Matrix(self.Q[n_ind:])

            self.q_dot_in = Matrix(self.q_dot[:n_ind])

            if print_status:
                print("Generating and simplifiying Phi_q_de_inv")
            self.phi_q_de_inv = simplify(self.phi_q_de.pinv())
            if print_status:
                print("Generating and simplifiying R")
            self.R = simplify(-self.phi_q_de_inv @ self.phi_q_in)
            self.R_dot = self.R.diff(t)
            self.q_dot_de = self.R @ self.q_dot_in
            if print_status:
                print("Generating and simplifiying H")
            self.H = simplify(self.M_in + self.R.T @ self.M_de @ self.R)
            if print_status:
                print("Generating and simplifiying K")
            self.K = simplify(self.R.T @ self.M_de @ self.R_dot)
            if print_status:
                print("Generating and simplifiying Fa")
            self.Fa = simplify(self.Q_in + self.R.T @ self.Q_de)
            if print_status:
                print("Generating and simplifiying reduced q_dot_dot")
            self.q_dotdot_in_expr = simplify(
                self.H.pinv() @ (self.Fa - self.K @ self.q_dot_in)
            )
            if print_status:
                print("Reduced model completed")
            self.RHS_reduced = Matrix(list(self.q_dot_in) + list(self.q_dotdot_in_expr))
        else:
            if print_status:
                print("Generating and simplifiying reduced q_dot_dot")
            self.q_dotdot_expr = simplify(self.M.pinv() @ self.Q)
            if print_status:
                print("Reduced model completed")
            self.RHS = Matrix(list(self.q_dot) + list(self.q_dotdot_expr))

    def calculate_RHS(self):
        if not hasattr(self, "RHS"):
            if self.print_status:
                print("Generating and simplifiying Right Hand Side")
            self.q_dotdot_de_expr = simplify(
                self.R_dot @ self.q_dot_in + self.R @ self.q_dotdot_in_expr
            )
            self.RHS = Matrix(
                list(self.q_dot)
                + list(self.q_dotdot_in_expr)
                + list(self.q_dotdot_de_expr)
            )
        return self.RHS

    def calculate_lambda_vec(self):
        if not hasattr(self, "lambda_vec"):
            if len(self.coneqs) == 0:
                self.lambda_vec = []
            else:
                if self.print_status:
                    print("Generating and simplifiying Lambdas vector")
                self.lambda_vec = simplify(
                    self.phi_q_de_inv @ (self.Q_de - self.M_de @ self.q_dotdot_de_expr)
                )
        return self.lambda_vec

    def calculate_RHS_full(self):
        if not hasattr(self, "RHS_full"):
            if not hasattr(self, "RHS"):
                self.calculate_RHS()
            if not hasattr(self, "lambda_vec"):
                self.calculate_lambda_vec()
            self.RHS_full = Matrix(list(self.RHS) + list(self.lambda_vec))
        return self.RHS_full


def find_arguments(expr_list, q_vars, u_vars=None):

    expr_list = list(expr_list)
    expr_list = [standard_notation(diff_to_symb_expr(expr)) for expr in expr_list]
    max_n_var = max([len(expr.atoms(Symbol)) for expr in expr_list])
    u_args = []
    params = []
    args = []
    u_args_found = []
    x_args_found = []

    if type(q_vars) == int:
        q_args = []
        v_args = []
        for jj in range(q_vars):
            q = symbols(f"q_{jj}")
            q_args.append(q)
            v = symbols(f"v_{jj}")
            v_args.append(v)
    elif type(q_vars) == list:
        q_args = [diff_to_symb(var) for var in q_vars]
        v_args = [diff_to_symb(var.diff()) for var in q_vars]
    else:
        raise TypeError(
            "data type not undersood for q_vars, must be an integer or list of symbols"
        )

    if u_vars is None:
        for jj in range(max_n_var):
            u = symbols(f"u_{jj}")
            u_args.append(u)
    elif type(u_vars) == list:
        u_args = u_vars
    else:
        raise TypeError(
            "data type not undersood for u_vars, must be an integer or list of symbols"
        )

    x_args = q_args + v_args
    args = x_args + u_args
    for ii in range(len(expr_list)):
        expr = expr_list[ii]
        var_set = expr.atoms(Symbol)
        for symb in var_set:
            if not symb in args:
                if not symb in params:
                    params.append(symb)
            elif symb in u_args:
                if not symb in u_args_found:
                    u_args_found.append(symb)
            elif symb in x_args:
                if not symb in x_args_found:
                    x_args_found.append(symb)

    params = sorted(params, key=get_str)
    if u_vars is None:
        u_args_found = sorted(u_args_found, key=get_str)
    else:
        u_args_found = u_vars

    return q_args, v_args, x_args_found, u_args, u_args_found, params


def print_funcs_RHS(RHS, q_vars, u_vars=None, flavour="numpy"):
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
    
    flavour : str in ["numpy", "casadi"], default = "numpy"
        experimental feature, converts common functions like sin(x)
        to np.sin(x) or cas.sin(x) respectively

    Returns
    -------
    string
        when outputted by print(), can be copypasted to define a function
        associated with RHS: x' = F(x, u, params)

    """
    RHS = list(RHS)
    RHS = [standard_notation(diff_to_symb_expr(expr)) for expr in RHS]
    arguments = find_arguments(RHS, q_vars, u_vars)
    q_args, v_args, x_args_found, u_args, u_args_found, params = arguments
    x_args = q_args + v_args
    msg = "def F(x, u, params):\n"
    msg += f"    {x_args.__str__()[1:-1]} = unpack(x)\n"
    msg += f"    {u_args_found.__str__()[1:-1]} = unpack(u)\n"
    msg += f"    {params.__str__()[1:-1]} = params\n"
    msg += f"    result = [{v_args.__str__()[1:-1]},]\n"
    for expr in RHS:
        msg += "    result.append(" + expr.__str__() + ")\n"
    msg += "\n    return result\n"

    print(msg)
    return msg


def print_funcs(expr_list, q_vars=0, flavour="numpy"):
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
    
    flavour : str in ["numpy", "casadi"], default = "numpy"
        experimental feature, converts common functions like sin(x)
        to np.sin(x) or cas.sin(x) respectively

    Returns
    -------
    string
        when outputted by print(), can be copypasted to define a function
        associated with RHS: x' = F(x, u, params)

    """
    expr_list = list(expr_list)
    expr_list = [standard_notation(diff_to_symb_expr(expr)) for expr in expr_list]
    arguments = find_arguments(expr_list, q_vars)
    q_args, v_args, x_args_found, u_args, u_args_found, params = arguments
    x_args = q_args + v_args

    msg = "def F("
    if len(x_args_found) > 0:
        msg += "x, "
    if len(u_args_found) > 0:
        msg += "u, "
    if len(params) > 0:
        msg += "params, "
    msg += "):\n"
    if len(x_args_found) > 0:
        msg += f"    {x_args.__str__()[1:-1]} = unpack(x)\n"
    if len(u_args_found) > 0:
        msg += f"    {u_args_found.__str__()[1:-1]} = unpack(u)\n"
    if len(params) > 0:
        msg += f"    {params.__str__()[1:-1]} = params\n"
    if len(expr_list) == 1:
        msg += "    result = " + expr_list[0].__str__() + "\n"
    else:
        msg += "    result = []\n"
        for expr in expr_list:
            msg += "    result.append(" + expr.__str__() + ")\n"
    msg += "\n    return result\n"

    print(msg)
    return msg
