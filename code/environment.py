"""Stage 1: logit Bertrand duopoly environment (Calvano et al. 2020 / FGS Section 2.1)."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

A = np.array([2.0, 2.0])   # product qualities a_i
A0 = 0.0                   # outside option a_0
MU = 0.25                  # horizontal differentiation
C = np.array([1.0, 1.0])   # marginal cost c_i (before alpha scaling)
BETA = 100.0               # quantity scale

def demand(p, alpha=1.0):
    """Logit demand: quantity sold by each firm given both prices p = [p1, p2].
    alpha is the currency scale; it rescales prices and costs without changing the economics."""
    p = np.asarray(p, float)
    e = np.exp((A - p / alpha) / MU)
    return BETA * e / (e.sum() + np.exp(A0 / MU))

def profit(p, alpha=1.0):
    """Profit of each firm: (price - unit cost) x quantity."""
    p = np.asarray(p, float)
    return (p - alpha * C) * demand(p, alpha)

def best_response(p_other, alpha=1.0):
    """Firm 1's profit-maximising price when the rival charges p_other (numerical search)."""
    r = minimize_scalar(lambda x: -profit([x, p_other], alpha)[0],
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x

def nash_price(alpha=1.0):
    """Symmetric Bertrand-Nash price: the price that is a best response to itself."""
    return brentq(lambda p: best_response(p, alpha) - p, alpha * C[0], 10 * alpha)

def monopoly_price(alpha=1.0):
    """Joint-monopoly price: the common price that maximises the two firms' combined profit."""
    r = minimize_scalar(lambda x: -profit([x, x], alpha).sum(),
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x

# Stage 1 check: print the benchmark prices and profits at each currency scale.
# Expected (normalised by alpha): Nash 1.473, joint monopoly 1.925.
if __name__ == "__main__":
    for alpha in (1, 3.2, 10):
        pn, pm = nash_price(alpha), monopoly_price(alpha)
        print(f"alpha={alpha:>4}: Nash {pn/alpha:.3f}  Monopoly {pm/alpha:.3f}  "
              f"| profit/firm Nash {profit([pn,pn],alpha)[0]/alpha:.2f}  "
              f"Monopoly {profit([pm,pm],alpha)[0]/alpha:.2f}  (normalised)")


# ---------- Monopoly (single-product) environment, FGS Appendix C ----------
def demand_mono(p, alpha=1.0):
    """Demand for a single firm facing only the outside option (Stage 2 monopoly check)."""
    e = np.exp((A[0] - p / alpha) / MU)
    return BETA * e / (e + np.exp(A0 / MU))

def profit_mono(p, alpha=1.0):
    """Profit of the single firm."""
    return (p - alpha * C[0]) * demand_mono(p, alpha)

def monopoly_price_single(alpha=1.0):
    """Profit-maximising price of the single firm (1.802 after normalising by alpha)."""
    r = minimize_scalar(lambda x: -profit_mono(x, alpha),
                        bounds=(alpha * C[0], 10 * alpha), method="bounded",
                        options={"xatol": 1e-10})
    return r.x
