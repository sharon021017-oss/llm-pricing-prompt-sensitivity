# Supervisor feedback: P1-trunc vs P1-para directly, and multiple-testing adjustment.
# Run after analyse_v2.py and analyse_profit_plans.py (needs v2_runs.csv and v2_profit_stability.csv)
# Output: results/v3_trunc_para_and_adjusted.csv, results/v3_keywords_trunc_para.csv
import glob, json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

U = pd.read_csv("results/v2_runs.csv")
P = pd.read_csv("results/v2_profit_stability.csv")
D = U.merge(P, on=["arm", "run"])
rng = np.random.default_rng(12345)

def boot(a, b, n=10000):    # bootstrap CI for difference in means (resampling runs)
    a, b = np.asarray(a), np.asarray(b)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])

outs = [("mean", "run mean price"), ("p10", "P10"), ("p25", "P25"), ("p50", "median"), ("p75", "P75"),
        ("p90", "P90"), ("sd", "within-run SD"), ("iqr", "within-run IQR"), ("profit", "profit per firm"),
        ("changes", "share of periods with price change"), ("big_cuts", "sharp price cuts")]
rows = []
def comp(A, B, label):
    for c, name in outs:
        a = D[D.arm == A][c]; b = D[D.arm == B][c]
        lo, hi = boot(a, b)
        rows.append({"comparison": label, "outcome": name, "A": a.mean(), "B": b.mean(), "diff": a.mean() - b.mean(),
                     "ci_lo": lo, "ci_hi": hi, "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
                     "mw_p": stats.mannwhitneyu(a, b).pvalue})

comp("P1-trunc", "P1-para", "P1-trunc vs P1-para")
comp("P1-trunc", "P1", "P1-trunc vs P1")
comp("P1-para", "P1", "P1-para vs P1")
T = pd.DataFrame(rows)

# Holm and Benjamini-Hochberg: (a) the 22 comparisons with P1, (b) the 11 trunc vs para comparisons
fam = T[T.comparison != "P1-trunc vs P1-para"].copy()
fam["holm"] = multipletests(fam.welch_p, method="holm")[1]
fam["bh"] = multipletests(fam.welch_p, method="fdr_bh")[1]
tp = T[T.comparison == "P1-trunc vs P1-para"].copy()
tp["holm"] = multipletests(tp.welch_p, method="holm")[1]
tp["bh"] = multipletests(tp.welch_p, method="fdr_bh")[1]
out = pd.concat([tp, fam])
print(out.round(4).to_string(index=False))
out.to_csv("results/v3_trunc_para_and_adjusted.csv", index=False)

# keywords, P1-trunc vs P1-para (run-level shares, Mann-Whitney)
rows = []
for f in glob.glob("results/duopoly_*_run*.jsonl"):
    L = [json.loads(l) for l in open(f)]
    if len(L) < 100:
        continue
    for r in L:
        for a in r["agents"]:
            text = (a["plans"] + " " + a["insights"] + " " + a.get("observations", "")).lower()
            rows.append({"arm": L[0]["arm"], "run": L[0]["run"], "text": text})
X = pd.DataFrame(rows)
kw = {"explore/test": r"\bexplor|\btest", "aggressive": r"aggressiv", "undercut": r"undercut",
      "stabil": r"stabil", "long run/term": r"long[- ](?:run|term)"}
res = []
for k, pat in kw.items():
    X[k] = X.text.str.contains(pat, regex=True)
    R = X.groupby(["arm", "run"])[k].mean().reset_index()
    a = R[R.arm == "P1-trunc"][k]; b = R[R.arm == "P1-para"][k]
    res.append({"term": k, "P1-trunc": round(100 * a.mean(), 1), "P1-para": round(100 * b.mean(), 1),
                "mw_p": round(stats.mannwhitneyu(a, b).pvalue, 4)})
K = pd.DataFrame(res); print(K.to_string(index=False))
K.to_csv("results/v3_keywords_trunc_para.csv", index=False)
