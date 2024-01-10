import torch
from utils import *
from tqdm import trange
from math import *
import numpy as np

# Following code for FASTA algorithm is taken from: https://github.com/phasepack/fasta-python/tree/master/fasta

__all__ = ['Param', 'asParam'] 

PARAM_PREFIX = 'pars'


class Param(dict):
    """
    Convenience class: a dictionary that gives access to its keys
    through attributes.
    
    Note: dictionaries stored in this class are also automatically converted
    to Param objects:
    >>> p = Param()
    >>> p.x = {}
    >>> p
    Param({})
    
    While dict(p) returns a dictionary, it is not recursive, so it is better in this case
    to use p.todict(). However, p.todict does not check for infinite recursion. So please
    don't store a dictionary (or a Param) inside itself.
    
    BE: Please note also that the recursive behavior of the update function will create
    new references. This will lead inconsistency if other objects refer to dicts or Params
    in the updated Param instance. 
    """
    _display_items_as_attributes = True
    _PREFIX = PARAM_PREFIX

    def __init__(self, __d__=None, **kwargs):
        """
        A Dictionary that enables access to its keys as attributes.
        Same constructor as dict.
        """
        dict.__init__(self)
        if __d__ is not None: self.update(__d__)
        self.update(kwargs)

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, dict.__repr__(self))

    # def __str__(self):
    #     from .verbose import report
    #     return report(self,depth=7,noheader=True)

    def __setitem__(self, key, value):
        # BE: original behavior modified as implicit conversion may destroy references
        # Use update(value,Convert=True) instead
        # return super(Param, self).__setitem__(key, Param(value) if type(value) == dict else value)
        return super(Param, self).__setitem__(key, value)

    def __getitem__(self, name):
        # item = super(Param, self).__getitem__(name)
        # return Param(item) if type(item) == dict else item
        return super(Param, self).__getitem__(name)

    def __delitem__(self, name):
        return super(Param, self).__delitem__(name)

    def __delattr__(self, name):
        return super(Param, self).__delitem__(name)

    # __getattr__ = __getitem__
    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError as ke:
            raise AttributeError(ke)

    __setattr__ = __setitem__

    def copy(self, depth=0):
        """
        :returns Param: A (recursive) copy of P with depth `depth` 
        """
        d = Param(self)
        if depth > 0:
            for k, v in d.iteritems():
                if isinstance(v, self.__class__): d[k] = v.copy(depth - 1)
        return d

    def __dir__(self):
        """
        Defined to include the keys when using dir(). Useful for
        tab completion in e.g. ipython.
        If you do not wish the dict key's be displayed as attributes
        (although they are still accessible as such) set the class 
        attribute `_display_items_as_attributes` to False. Default is
        True.
        """
        if self._display_items_as_attributes:
            return self.keys()
            # return [item.__dict__.get('name',str(key)) for key,item in self.iteritems()]
        else:
            return []

    def update(self, __d__=None, in_place_depth=0, Convert=False, **kwargs):
        """
        Update Param - almost same behavior as dict.update, except
        that all dictionaries are converted to Param if `Convert` is set 
        to True, and update may occur in-place recursively for other Param
        instances that self refers to.
        
        Parameters
        ----------
        Convert : bool 
                  If True, convert all dict-like values in self also to Param.
                  *WARNING* 
                  This mey result in misdirected references in your environment
        in_place_depth : int 
                  Counter for recursive in-place updates 
                  If the counter reaches zero, the Param to a key is
                  replaced instead of updated
        """

        def _k_v_update(k, v):
            # If an element is itself a dict, convert it to Param
            if Convert and hasattr(v, 'keys'):
                # print 'converting'
                v = Param(v)
            # new key 
            if not k in self:
                self[k] = v
            # If this key already exists and is already dict-like, update it
            elif in_place_depth > 0 and hasattr(v, 'keys') and isinstance(self[k], self.__class__):
                self[k].update(v, in_place_depth - 1)
                """
                if isinstance(self[k],self.__class__):
                    # Param gets recursive in_place updates
                    self[k].update(v, in_place_depth - 1)
                else:
                    # dicts are only updated in-place once
                    self[k].update(v)
                """
            # Otherwise just replace it
            else:
                self[k] = v

        if __d__ is not None:
            if hasattr(__d__, 'keys'):
                # Iterate through dict-like argument
                for k, v in __d__.items():
                    _k_v_update(k, v)

            else:
                # here we assume a (key,value) list.
                for (k, v) in __d__:
                    _k_v_update(k, v)

        for k, v in kwargs.items():
            _k_v_update(k, v)

        return None

    def _to_dict(self, Recursive=False):
        """
        Convert to dictionary (recursively if needed).
        """
        if not Recursive:
            return dict(self)
        else:
            d = dict(self)
            for k, v in d.items():
                if isinstance(v, self.__class__): d[k] = v._to_dict(Recursive)
        return d

    @classmethod
    def _from_dict(cls, dct):
        """
        Make Param from dict. This is similar to the __init__ call
        """
        # p=Param()
        # p.update(dct.copy())
        return Param(dct.copy())


def set_defaults_opts_base(opts, A, At, x0, gradf):
    valid_options = ['max_iters', 'tol', 'verbose', 'record_objective',
                     'record_iterates', 'adaptive', 'accelerate', 'restart', 'backtrack',
                     'stepsize_shrink', 'window', 'eps_r', 'eps_n', 'L', 'tau', 'function',
                     'string_header', 'stop_rule', 'stop_now', 'mode']
    for k in opts.keys():
        if not k in valid_options:
            raise RuntimeError(
                f'invalid option supplied to fasta: {k}.   Valid choices are:  max_iters, tol, verbose, record_objective, record_iterates,  adaptive, accelerate, restart, backtrack, stepsize_shrink,  window, eps_r, eps_n, L, tau, function, string_header,  stop_rule, stop_now.')

    if not 'max_iters' in opts:
        opts.max_iters = 1000
    if not 'tol' in opts:
        opts.tol = 1e-3
    if not 'verbose' in opts:
        opts.verbose = False
    if not 'record_objective' in opts:
        opts.record_objective = False
    if not 'record_iterates' in opts:
        opts.record_iterates = False
    if not 'adaptive' in opts:
        opts.adaptive = True
    if not 'accelerate' in opts:
        opts.accelerate = False
    if not 'restart' in opts:
        opts.restart = True
    if not 'backtrack' in opts:
        opts.backtrack = True
    if not 'stepsize_shrink' in opts:
        opts.stepsize_shrink = 0.2
        if ~opts.adaptive or opts.accelerate:
            opts.stepsize_shrink = 0.5
    if not 'window' in opts:
        opts.window = 10
    if not 'eps_r' in opts:
        opts.eps_r = 1e-8
    if not 'eps_n' in opts:
        opts.eps_n = 1e-8
    opts.mode = 'plain'
    if opts.adaptive:
        opts.mode = 'adaptive'
    if opts.accelerate:
        if opts.restart:
            opts.mode = 'accelerated(FISTA)+restart'
        else:
            opts.mode = 'accelerated(FISTA)'
    if not 'function' in opts:
        opts.function = lambda x: 0
    if not 'string_header' in opts:
        opts.string_header = ''
    if not 'stop_now' in opts:
        opts.stop_rule = 'hybridResidual'
    if opts.stop_rule == 'hybridResidual':
        def residual(x1, iter, resid, normResid, maxResidual, opts):
            return resid < opts.tol

        opts.stop_now = residual
    if opts.stop_rule == 'iterations':
        def iterations(x1, iter, resid, normResid, maxResidual, opts):
            return iter > opts.maxIters

        opts.stop_now = iterations
    if opts.stop_rule == 'normalizedResidual':
        def normalizedResidual(x1, iter, resid, normResid, maxResidual, opts):
            return normResid < opts.tol

        opts.stop_now = normalizedResidual
    if opts.stop_rule == 'ratioResidual':
        def ratioResidual(x1, iter, resid, normResid, maxResidual, opts):
            return resid / (maxResidual + opts.eps_r) < opts.tol

        opts.stop_now = ratioResidual
    if opts.stop_rule == 'hybridResidual':
        def hybridResidual(x1, iter, resid, normResid, maxResidual, opts):
            return (resid / (maxResidual + opts.eps_r) < opts.tol) or normResid < opts.tol

        opts.stop_now = hybridResidual

    assert 'stop_now' in opts, f'Invalid choice of stopping rule: {opts.stop_rule}'

    return opts


def set_default_opts(opts, A, At, x0, gradf):
    device, _ = get_devices()
    opts = set_defaults_opts_base(opts, A, At, x0, gradf)
    if (not 'L' in opts or opts.L <= 0) and (not 'tau' in opts or opts.tau <= 0):
        x1 = torch.randn(x0.shape).to(device)
        x2 = torch.randn(x0.shape).to(device)
        gradf1 = At(gradf(A(x1)))
        gradf2 = At(gradf(A(x2)))
        opts.L = torch.norm(gradf1 - gradf2) / torch.norm(x2 - x1)
        opts.L = max(opts.L, 1e-6)
        opts.tau = 2 / opts.L / 10
    assert opts.tau > 0, f'Invalid step size: {opts.tau}'
    if not 'tau' in opts or opts.tau <= 0:
        opts.tau = 1.0 / opts.L
    else:
        opts.L = 1 / opts.tau
    return opts


def fasta(A, At, f, gradf, g, proxg, x0, opts):
    """
    :param A: A matrix (or optionally a function handle to a method) that
             returns A*x
    :param At: The adjoint (transpose) of 'A.' Optionally, a function handle
             may be passed.
    :param f: A function of x, computes the value of f
    :param gradf: A function of z, computes the gradient of f at z
    :param g: A function of x, computes the value of g
    :param proxg: A function of z and t, the proximal operator of g with
             stepsize t.
    :param x0: The initial guess, usually a vector of zeros
    :param opts: An optional struct with options.  The commonly used fields
             of 'opts' are:
               maxIters : (integer, default=1e4) The maximum number of iterations
                               allowed before termination.
               tol      : (double, default=1e-3) The stopping tolerance.
                               A smaller value of 'tol' results in more
                               iterations.
               verbose  : (boolean, default=false)  If true, print out
                               convergence information on each iteration.
               recordObjective:  (boolean, default=false) Compute and
                               record the objective of each iterate.
               recordIterates :  (boolean, default=false) Record every
                               iterate in a cell array.
            To use these options, set the corresponding field in 'opts'.
            For example:
                      >> opts.tol=1e-8;
                      >> opts.maxIters = 100;
    :return: a tuple (solution, out_dictionary, in_options)
    """
    opts = set_default_opts(opts, A, At, x0, gradf)
    device, _ = get_devices()

    tau1 = opts.tau
    max_iters = opts.max_iters
    W = opts.window

    residual = torch.zeros(max_iters).to(device)
    normalized_resid = torch.zeros(max_iters).to(device)
    taus = torch.zeros(max_iters).to(device)
    fvals = torch.zeros(max_iters).to(device)
    objective = torch.zeros(max_iters + 1).to(device)
    func_values = torch.zeros(max_iters).to(device)
    total_backtracks = 0
    backtrack_count = 0
    iterates = {}

    x1 = x0
    d1 = A(x1)
    f1 = f(d1)
    fvals[0] = f1
    gradf1 = At(gradf(d1))

    if opts.accelerate:
        x_accel1 = x0
        d_accel1 = d1
        alpha1 = 1

    max_residual = - np.inf
    min_objective_value = np.inf

    if opts.record_objective:
        with torch.no_grad():
          objective[0] = f1 + g(x0)

    for i in range(max_iters):
        

        x0 = x1
        gradf0 = gradf1
        tau0 = tau1
        

        x1hat = x0 - tau0 * gradf0
        
        x1 = proxg(x1hat, tau0)

        Dx = x1 - x0


        d1 = A(x1)
        f1 = f(d1)



        if opts.backtrack and i > 0:
            M = torch.max(fvals[max(i - W, 0):max(i, 0)])
            backtrack_count = 0
            while f1 - 1e-12 > M + torch.dot(Dx.flatten(), gradf0.flatten()) + torch.norm(Dx.flatten()) ** 2 / (
                    2 * tau0) and backtrack_count < 20:
                tau0 = tau0 * opts.stepsize_shrink
                x1hat = x0 - tau0 * gradf0
                x1 = proxg(x1hat, tau0)
                d1 = A(x1)
                f1 = f(d1)
                Dx = x1 - x0
                backtrack_count += 1
            total_backtracks += backtrack_count
        if opts.verbose and backtrack_count > 10:
            print(f'WARNING: excessive backtracking ({backtrack_count} steps, current stepsize is {tau0}')


        taus[i] = tau0
        residual[i] = torch.norm(Dx) / tau0
        max_residual = max(residual[i], max_residual)
        normalizer = max(torch.norm(gradf0), torch.norm(x1 - x1hat / tau0) + opts.eps_n)
        normalized_resid[i] = residual[i] / normalizer
        fvals[i] = f1
        func_values[i] = opts.function(x0)
        if opts.record_objective:
            with torch.no_grad():
              objective[i + 1] = f1 + g(x1)
            new_objective_value = objective[i + 1]
        else:
            new_objective_value = residual[i]

        if opts.record_iterates:
            iterates[f'{i}'] = x1

        if new_objective_value < min_objective_value:
            best_objective_iterate = x1
            min_objective_value = new_objective_value

        if opts.verbose:
            print()

        if opts.stop_now(x1, i, residual[i], normalized_resid[i], max_residual, opts) or i > max_iters:
            outs = Param()
            outs.solve_time = 0
            outs.residuals = residual[:i]
            outs.stepsizes = taus[1:i]
            outs.normalized_residuals = normalized_resid[1:i]
            outs.objective = objective[1:i]
            outs.func_values = func_values[1:i]
            outs.backtracks = total_backtracks
            outs.L = opts.L
            outs.initial_stepsize = opts.tau
            outs.iteration_count = i
            if opts.record_iterates:
                outs.iterates = iterates
            sol = best_objective_iterate
            return sol, outs, opts

        if opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1))
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = torch.dot(Dx.flatten(), Dg.flatten())
            tau_s = torch.norm(Dx) ** 2 / dotprod
            tau_m = dotprod / torch.norm(Dg) ** 2
            tau_m = max(tau_m, 0)
            if 2 * tau_m > tau_s:
                tau1 = tau_m
            else:
                tau1 = tau_s - 0.5 * tau_m
            if tau1 <= 0 or torch.isinf(tau1) or torch.isnan(tau1):
                tau1 = tau0 * 1.5


        if opts.accelerate:
            x_accel0 = x_accel1
            d_accel0 = d_accel1
            alpha0 = alpha1
            x_accel1 = x1
            d_accel1 = d1
            if opts.restart and torch.dot((x0 - x1).view(-1), (x_accel1 - x_accel0).view(-1)) > 0:
                alpha0 = 1
            alpha1 = (1 + sqrt(1 + 4 * alpha0 ** 2)) / 2
            x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)
            # Compute the gradient needed on the next iteration
            gradf1 = At(gradf(d1))
            func_values[i] = f(d1)
            tau1 = tau0

        if ~opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1))
            tau1 = tau0
    outs = Param()
    outs.solve_time = 0
    outs.residuals = residual[:i]
    outs.stepsizes = taus[1:i]
    outs.normalized_residuals = normalized_resid[1:i]
    outs.objective = objective[1:i]
    outs.func_values = func_values[1:i]
    outs.backtracks = total_backtracks
    outs.L = opts.L
    outs.initial_stepsize = opts.tau
    outs.iteration_count = i
    if opts.record_iterates:
        outs.iterates = iterates
    sol = best_objective_iterate
    return sol, outs, opts