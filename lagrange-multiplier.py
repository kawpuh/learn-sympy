import sympy
from sympy import symbols, log, diff, nonlinsolve, nsolve

# Using the maximum entropy problem to help build intuition for the method of lagrange multipliers

true_n = 2
n = true_n + 1 # our use of n is mostly in range bounds, which need n+1
avg = 1.3
p = symbols(f'p_1:{n}', real=True, nonnegative=True)
lmda = symbols('lmda', real=True) # lm for average constraint
mu = symbols('mu', real=True, nonzero=True) # lm for probability constraint
f = -sum(p_i * log(p_i) for p_i in p)
g = sum(i * p_i for i, p_i in enumerate(p, start=1)) - avg
h = sum(p_i for p_i in p) - 1
lg = f - h*lmda
syms = p + (lmda,)
lg_grad = [lg.diff(sym) for sym in syms]
[first_partial_soln, *other_partial_solns] = nonlinsolve(lg_grad, p)
assert(len(other_partial_solns) == 0) # if this fails, we have alternate solutions to check
substituted_lg_grad = lg_grad[-1].subs(p[0], first_partial_soln[0]).subs(p[1], first_partial_soln[1])
print(substituted_lg_grad)
[first_lmda_soln, *other_lmda_solns] = nsolve([substituted_lg_grad], [lmda], [0.5])
assert(len(other_lmda_solns) == 0)
print(lg_grad)

# sympy.plotting.plot3d(f, h, (p[0],0,1), (p[1],0,1))
# sympy.plotting.plot3d(f, h*first_lmda_soln, (p[0],0,1), (p[1],0,1))
