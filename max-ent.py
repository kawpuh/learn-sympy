import pprint
import sympy
from sympy import symbols, log, diff, nonlinsolve, nsolve

p = symbols('p_1:7', real=True, nonnegative=True)
lmda = symbols('lmda', real=True)
mu = symbols('mu', real=True, nonzero=True)
f = -sum(p_i * log(p_i) for p_i in p)
g = sum(i * p_i for i, p_i in zip(range(1, 7), p)) - 5
h = sum(p_i for p_i in p) - 1
lagrangian = f - lmda*g - mu*h
syms = p + (lmda, mu) # (p1, p2, p3, p4, p5, p6, lmda, mu)
lagrangian_grad = [lagrangian.diff(sym) for sym in syms]
partial_soln_p = nonlinsolve(lagrangian_grad, p)
new_system = lagrangian_grad[-2:]
for symbol, partial_soln in zip(p, next(iter(partial_soln_p))):
  for i, unsolved in enumerate(new_system):
    new_system[i] = unsolved.subs(symbol, partial_soln)
lmda_soln, mu_soln = nsolve(new_system, syms[-2:], [0, 0.792])
soln_p = list()
for p_i in partial_soln_p:
  soln_p.append(p_i.subs(lmda, lmda_soln).subs(mu, mu_soln))


for symbol, soln in zip(p, next(iter(soln_p))):
  print(symbol, " = ", soln)

print("lmda = ", lmda_soln)
print("mu   = ", mu_soln)
