"""Stage 1: logit Bertrand duopoly environment (Calvano et al. 2020 / FGS Section 2.1)."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

A = np.array([2.0, 2.0])   # product qualities a_i
A0 = 0.0                   # outside option a_0
MU = 0.25                  # horizontal differentiation
C = np.array([1.0, 1.0])   # marginal cost c_i (before alpha scaling)
BETA = 100.0               # quantity scale

def demand(p, alpha=1.0):
    p = np.asarray(p, float)
    e = np.exp((A - p / alpha) / MU)
    return BETA * e / (e.sum() + np.exp(A0 / MU))

def profit(p, alpha=1.0):
    p = np.asarray(p, float)
    return (p - alpha * C) * demand(p, alpha)

def best_response(p_other, alpha=1.0):
    r = minimize_scalar(lambda x: -profit([x, p_other], alpha)[0],
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x

def nash_price(alpha=1.0):
    return brentq(lambda p: best_response(p, alpha) - p, alpha * C[0], 10 * alpha)

def monopoly_price(alpha=1.0):
    r = minimize_scalar(lambda x: -profit([x, x], alpha).sum(),
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x

if __name__ == "__main__":
    for alpha in (1, 3.2, 10):
        pn, pm = nash_price(alpha), monopoly_price(alpha)
        print(f"alpha={alpha:>4}: Nash {pn/alpha:.3f}  Monopoly {pm/alpha:.3f}  "
              f"| profit/firm Nash {profit([pn,pn],alpha)[0]/alpha:.2f}  "
              f"Monopoly {profit([pm,pm],alpha)[0]/alpha:.2f}  (normalised)")


# ---------- Monopoly (single-product) environment, FGS Appendix C ----------
def demand_mono(p, alpha=1.0):
    e = np.exp((A[0] - p / alpha) / MU)
    return BETA * e / (e + np.exp(A0 / MU))

def profit_mono(p, alpha=1.0):
    return (p - alpha * C[0]) * demand_mono(p, alpha)

def monopoly_price_single(alpha=1.0):
    r = minimize_scalar(lambda x: -profit_mono(x, alpha),
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x
