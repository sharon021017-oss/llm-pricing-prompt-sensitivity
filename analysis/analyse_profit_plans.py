"""Profits, price stability and keyword analysis of the agents' reasoning.

Reads every complete duopoly run in results/ (duopoly_<arm>_run<k>.jsonl) and writes:
  results/v2_profit_stability.csv  one row per run: profits and stability measures
                                   over the terminal window (periods 51-100)
  results/v2_plans_keywords.csv    for selected terms, the share of agent-periods
                                   whose written text mentions the term, by arm,
                                   with Mann-Whitney p-values against P1

Run from the repository folder:  python analysis/analyse_profit_plans.py
"""
import glob, json
import numpy as np
import pandas as pd
from scipy import stats

ARMS = ["P1", "P1-trunc", "P1-para"]   # P1 is the baseline; the others are compared with it

rows = []    # one dict per run (profits and stability)
plans = []   # one dict per agent per period (the text the agent wrote)

for f in glob.glob("results/duopoly_*_run*.jsonl"):
    L = [json.loads(l) for l in open(f)]      # one record per period
    if len(L) < 100:                          # skip incomplete runs
        continue
    arm, run, al = L[0]["arm"], L[0]["run"], L[0]["alpha"]

    # Prices and profits as (period x firm) arrays, divided by the currency scale alpha
    # so that all runs are on the same scale.
    P = np.array([[a["price"] / r["alpha"] for a in r["agents"]] for r in L])
    Pi = np.array([[a["profit"] / r["alpha"] for a in r["agents"]] for r in L])

    w = slice(50, 100)    # terminal window: periods 51-100 (Python counts from 0)
    pm = P.mean(1)        # market price in each period = average of the two firms

    rows.append({
        "arm": arm, "run": run,
        # mean profit per firm, and mean combined profit of both firms
        "profit": Pi[w].mean(),
        "joint_profit": Pi[w].sum(1).mean(),
        # mean absolute difference between the two firms' prices and profits
        "firm_gap": np.abs(P[w, 0] - P[w, 1]).mean(),
        "profit_gap": np.abs(Pi[w, 0] - Pi[w, 1]).mean(),
        # share of firm-periods in which a firm changes its price by more than 0.005
        "changes": (np.abs(np.diff(P[w], axis=0)) > 0.005).mean(),
        # number of sharp price cuts: a firm's price falls by more than 0.05 in one period
        "big_cuts": (np.diff(P[w], axis=0) < -0.05).sum(),
        # number of periods in which the market price (average of both firms) falls by more than 0.05
        "war_episodes": int((np.diff(pm[w]) < -0.05).sum()),
        # share of periods in which the two firms' prices differ by more than 0.01
        "undercut_share": (np.abs(P[w, 0] - P[w, 1]) > 0.01).mean(),
    })

    # Collect everything each agent wrote in each period (all 100 periods) for the keyword analysis.
    for r in L:
        for a in r["agents"]:
            text = a["plans"] + " " + a["insights"] + " " + a.get("observations", "")
            plans.append({"arm": arm, "run": run, "t": r["period"], "text": text.lower()})

# ---------------- Profits and stability ----------------
R = pd.DataFrame(rows)
R["arm"] = pd.Categorical(R.arm, ARMS)
measures = ["profit", "joint_profit", "firm_gap", "profit_gap", "changes", "big_cuts",
            "war_episodes", "undercut_share"]
print(R.groupby("arm", observed=True)[measures].mean().round(4).T)   # arm means

# Compare each treatment with P1 across runs (the run is the unit of observation):
# Welch's unequal-variance t-test and the Mann-Whitney rank test.
b = R[R.arm == "P1"]
for arm in ["P1-trunc", "P1-para"]:
    g = R[R.arm == arm]
    for v in ["profit", "joint_profit", "firm_gap", "profit_gap", "changes", "big_cuts"]:
        print(arm, v, "diff", round(g[v].mean() - b[v].mean(), 4),
              "welch p", round(stats.ttest_ind(g[v], b[v], equal_var=False).pvalue, 4),
              "MW p", round(stats.mannwhitneyu(g[v], b[v]).pvalue, 4))
R.to_csv("results/v2_profit_stability.csv", index=False)

# ---------------- Keyword analysis of the agents' reasoning ----------------
T = pd.DataFrame(plans)

# Each term is a regular expression searched for in the lower-case text.
# \b marks the start of a word, so "\bexplor" matches "explore", "exploration", ...
kw = {
    "undercut": r"undercut",
    "price war": r"price war|pricing war",
    "aggressive": r"aggressiv",
    "explore/test": r"\bexplor|\btest",
    "match competitor": r"\bmatch",
    "collude/cooperate": r"collu|cooperat|tacit",
    "long run/term": r"long[- ](run|term)",
    "maximi[sz]e profit": r"maximi[sz]\w* (the )?(user'?s )?profit",
    "volume/market share": r"volume|market share",
    "stabil": r"stabil",
    "higher price": r"higher price|raise|increase (our )?price",
    "lower price": r"lower price|reduc\w* (our )?price|cut",
}

out = []
for k, pat in kw.items():
    # True/False for each agent-period: does the text mention the term?
    T[k] = T.text.str.contains(pat, regex=True)
    # Share of agent-periods mentioning the term, computed separately for each run ...
    runlvl = T.groupby(["arm", "run"])[k].mean().reset_index()
    # ... then averaged across runs within each arm (reported in %).
    m = runlvl.groupby("arm")[k].mean()
    p1 = runlvl[runlvl.arm == "P1"][k]
    row = {"term": k, **{a: round(100 * m[a], 1) for a in ARMS}}
    # Mann-Whitney test of the run-level shares, treatment vs P1
    for a in ["P1-trunc", "P1-para"]:
        row[a + "_p"] = round(stats.mannwhitneyu(runlvl[runlvl.arm == a][k], p1).pvalue, 4)
    out.append(row)

K = pd.DataFrame(out)
print("\n% of agent-periods mentioning (run-level average):\n", K.to_string(index=False))
K.to_csv("results/v2_plans_keywords.csv", index=False)

# Average length (characters) of what agents wrote, by arm
T["len"] = T.text.str.len()
print("\ntext length", T.groupby("arm").len.mean().round(0).to_dict())
