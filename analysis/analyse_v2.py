"""Extended analysis (supervisor meeting, 21 Oct 2026).

Run from the experiment folder:   python analyse_v2.py
Reads results/duopoly_*_run*.jsonl (complete 100-period runs only) and writes
results/v2_*.csv and results/v2_*.png.

Per run, the price series is the mean of the two firms' prices each period,
normalised by alpha, over periods 51-100 (the "terminal window").
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
ARMS = ["P1", "P1-trunc", "P1-para"]
NASH, MONO = 1.4729, 1.9250
MARKUP = MONO - 1.0
LAST = 50
QS = [10, 25, 50, 75, 90]
rng = np.random.default_rng(12345)

# ---------------- load ----------------
series, rows = {}, []
for f in sorted(glob.glob(os.path.join(RES, "duopoly_*_run*.jsonl"))):
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if len(recs) < 100 or recs[0]["arm"] not in ARMS:
        continue
    arm, run, alpha = recs[0]["arm"], recs[0]["run"], recs[0]["alpha"]
    p = np.array([np.mean([a["price"] for a in r["agents"]]) / r["alpha"] for r in recs])
    w = p[-LAST:]
    series[(arm, run)] = p
    row = {"arm": arm, "run": run, "alpha": alpha, "mean": w.mean(), "sd": w.std(ddof=1),
           "iqr": np.percentile(w, 75) - np.percentile(w, 25),
           "range": w.max() - w.min()}
    for q in QS:
        row[f"p{q}"] = np.percentile(w, q)
    rows.append(row)
R = pd.DataFrame(rows).sort_values(["arm", "run"]).reset_index(drop=True)
R["arm"] = pd.Categorical(R["arm"], ARMS)
R.to_csv(os.path.join(RES, "v2_runs.csv"), index=False)
n = R.groupby("arm", observed=True).size()
print("Runs per arm:", dict(n))

# ---------------- 1. descriptive table ----------------
desc = R.groupby("arm", observed=True).agg(
    runs=("mean", "size"), mean_of_means=("mean", "mean"), sd_across_runs=("mean", "std"),
    min_mean=("mean", "min"), max_mean=("mean", "max"),
    within_sd=("sd", "mean"), within_sd_median=("sd", "median"),
    **{f"avg_p{q}": (f"p{q}", "mean") for q in QS})
desc["index"] = (desc["mean_of_means"] - NASH) / (MONO - NASH)
desc.round(4).to_csv(os.path.join(RES, "v2_descriptive.csv"))
print("\n1. DESCRIPTIVES (run level; terminal window)\n", desc.round(3).T.to_string())

# ---------------- 2. tests vs P1 ----------------
def boot_ci_diff(x, y, stat=np.mean, B=10000):
    bs = [stat(rng.choice(x, len(x))) - stat(rng.choice(y, len(y))) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])

def tost(x, y, m):
    d = x.mean() - y.mean()
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy); df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return max(1 - stats.t.cdf((d + m) / se, df), stats.t.cdf((d - m) / se, df))

base = R[R.arm == "P1"]
tests = []
for arm in ["P1-trunc", "P1-para"]:
    g = R[R.arm == arm]
    for var, label in [("mean", "run mean price"), ("p10", "P10"), ("p25", "P25"), ("p50", "median"),
                       ("p75", "P75"), ("p90", "P90"), ("sd", "within-run SD"), ("iqr", "within-run IQR")]:
        x, y = g[var].values, base[var].values
        lo, hi = boot_ci_diff(x, y)
        tests.append({"comparison": f"{arm} vs P1", "outcome": label,
                      "arm_value": x.mean(), "P1_value": y.mean(), "diff": x.mean() - y.mean(),
                      "boot_CI_low": lo, "boot_CI_high": hi,
                      "welch_p": stats.ttest_ind(x, y, equal_var=False).pvalue,
                      "mannwhitney_p": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue})
    x, y = g["mean"].values, base["mean"].values
    tests.append({"comparison": f"{arm} vs P1", "outcome": "across-run variance (Brown-Forsythe)",
                  "arm_value": x.std(ddof=1), "P1_value": y.std(ddof=1), "diff": x.std(ddof=1) / y.std(ddof=1),
                  "welch_p": stats.levene(x, y, center="median").pvalue})
    tests.append({"comparison": f"{arm} vs P1", "outcome": "distribution of run means (KS)",
                  "welch_p": stats.ks_2samp(x, y).pvalue})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ad = stats.anderson_ksamp([x, y]).pvalue
    tests.append({"comparison": f"{arm} vs P1", "outcome": "distribution of run means (Anderson-Darling)",
                  "welch_p": ad})
    tests.append({"comparison": f"{arm} vs P1", "outcome": "distribution of within-run SD (KS)",
                  "welch_p": stats.ks_2samp(g["sd"].values, base["sd"].values).pvalue})
    if arm == "P1-para":
        tests.append({"comparison": "P1-para vs P1", "outcome": "TOST equivalence of means (+/-10% markup)",
                      "welch_p": tost(x, y, 0.10 * MARKUP)})
T = pd.DataFrame(tests)
T.to_csv(os.path.join(RES, "v2_tests.csv"), index=False)
print("\n2. TESTS VS P1 (p-value column = test named in 'outcome' where not Welch)\n",
      T.round(4).to_string(index=False))

# ---------------- 3. variance decomposition ----------------
long = pd.DataFrame([{"arm": a, "run": r, "t": t + 1, "price": v}
                     for (a, r), p in series.items() for t, v in enumerate(p) if t >= 100 - LAST])
grand = long.price.mean()
arm_m = long.groupby("arm").price.transform("mean")
run_m = long.groupby(["arm", "run"]).price.transform("mean")
ss_tot = ((long.price - grand) ** 2).sum()
ss_arm = ((arm_m - grand) ** 2).sum()
ss_run = ((run_m - arm_m) ** 2).sum()
ss_within = ((long.price - run_m) ** 2).sum()
vd = pd.DataFrame({"component": ["between arms (prompt)", "between runs within arm (chance)",
                                 "within runs over time"],
                   "sum_of_squares": [ss_arm, ss_run, ss_within]})
vd["share_pct"] = 100 * vd.sum_of_squares / ss_tot
print("\n3a. VARIANCE DECOMPOSITION (all period-level prices, terminal window)\n", vd.round(3).to_string(index=False))
by_arm = []
for arm, g in long.groupby("arm"):
    rm = g.groupby("run").price.transform("mean")
    b, w = ((rm - g.price.mean()) ** 2).sum(), ((g.price - rm) ** 2).sum()
    by_arm.append({"arm": arm, "between_runs_pct": 100 * b / (b + w), "within_runs_pct": 100 * w / (b + w),
                   "total_var": g.price.var()})
VA = pd.DataFrame(by_arm).set_index("arm").loc[ARMS]
print("\n3b. DECOMPOSITION WITHIN EACH ARM\n", VA.round(4).to_string())
pd.concat([vd.set_index("component"), VA], axis=0).to_csv(os.path.join(RES, "v2_variance_decomposition.csv"))

# random-effects (mixed) model: price ~ arm + (1|run)
long["runid"] = long.arm + "_" + long.run.astype(str)
mm = smf.mixedlm("price ~ C(arm, Treatment('P1'))", long, groups=long["runid"]).fit(reml=True)
var_run, var_res = float(mm.cov_re.iloc[0, 0]), float(mm.scale)
print("\n3c. MIXED MODEL price ~ arm + (1|run)")
print(mm.summary().tables[1].round(4).to_string())
print(f"   run-level variance {var_run:.5f}; residual (within-run) variance {var_res:.5f}; "
      f"ICC = {var_run / (var_run + var_res):.3f}")

# ---------------- 4. regressions at run level ----------------
reg = []
for y in ["mean", "p10", "p50", "p90", "sd"]:
    d = R.copy()
    d["y"] = d[y]; lab = "within-run SD" if y == "sd" else y
    d["batch2"] = (d.run > 10).astype(int)
    m = smf.ols("y ~ C(arm, Treatment('P1')) + C(alpha) + batch2", d).fit(cov_type="HC3")
    for k in m.params.index:
        if "arm" in k:
            reg.append({"outcome": lab, "term": k.split("[T.")[1].rstrip("]"),
                        "coef": m.params[k], "se_HC3": m.bse[k], "p": m.pvalues[k], "n": int(m.nobs),
                        "R2": m.rsquared})
RG = pd.DataFrame(reg)
RG.to_csv(os.path.join(RES, "v2_regressions.csv"), index=False)
print("\n4. RUN-LEVEL OLS: outcome ~ arm + alpha dummies + batch dummy (HC3 SEs)\n", RG.round(4).to_string(index=False))

# ---------------- 5. robustness: original 10 vs new runs ----------------
R["batch"] = np.where(R.run <= 10, "runs 1-10", "runs 11-29")
print("\n5. BY BATCH (mean of run means / mean within-run SD)\n",
      R.groupby(["arm", "batch"], observed=True)[["mean", "sd"]].mean().round(3).to_string())

# ---------------- figures ----------------
cols = {"P1": "#3b6fb6", "P1-trunc": "#c0504d", "P1-para": "#4f9a5b"}

# (a) price paths
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, arm in zip(axes, ARMS):
    for (a, r), p in series.items():
        if a == arm:
            ax.plot(range(1, 101), p, lw=0.6, alpha=0.6, color=cols[arm])
    ax.axhline(NASH, ls="--", c="k", lw=0.8); ax.axhline(MONO, ls=":", c="k", lw=0.8)
    ax.axvspan(51, 100, color="0.92", zorder=0)
    ax.set_title(f"{arm} (n={n[arm]})"); ax.set_xlabel("period"); ax.set_ylim(1.3, 3.0)
axes[0].set_ylabel("price / α (mean of firms)")
fig.tight_layout(); fig.savefig(os.path.join(RES, "v2_price_paths.png"), dpi=200); plt.close(fig)

# (b) per-run percentile panels (supervisor request)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, arm in zip(axes, ARMS):
    g = R[R.arm == arm].sort_values("p50").reset_index(drop=True)
    xs = np.arange(len(g))
    ax.vlines(xs, g.p10, g.p90, color=cols[arm], lw=1.2, alpha=0.6, label="P10–P90")
    ax.vlines(xs, g.p25, g.p75, color=cols[arm], lw=4, label="P25–P75")
    ax.plot(xs, g.p50, "o", ms=3, color="k", label="median")
    ax.axhline(NASH, ls="--", c="k", lw=0.8); ax.axhline(MONO, ls=":", c="k", lw=0.8)
    ax.set_title(arm); ax.set_xlabel("runs (sorted by median)"); ax.set_xticks([])
axes[0].set_ylabel("price / α, periods 51–100"); axes[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "v2_run_percentiles.png"), dpi=200); plt.close(fig)

# (c) ECDF of run means + (d) within-run SD by arm
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for arm in ARMS:
    x = np.sort(R[R.arm == arm]["mean"].values)
    axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=cols[arm], label=arm, lw=1.6)
axes[0].axvline(NASH, ls="--", c="k", lw=0.8); axes[0].axvline(MONO, ls=":", c="k", lw=0.8)
axes[0].set_xlabel("run mean price (periods 51–100)"); axes[0].set_ylabel("cumulative share of runs")
axes[0].legend(); axes[0].set_title("Distribution of run means")
data = [R[R.arm == a]["sd"].values for a in ARMS]
bp = axes[1].boxplot(data, showfliers=False, patch_artist=True)
axes[1].set_xticks([1, 2, 3]); axes[1].set_xticklabels(ARMS)
for patch, a in zip(bp["boxes"], ARMS):
    patch.set_facecolor(cols[a]); patch.set_alpha(0.35)
for i, d in enumerate(data):
    axes[1].plot(np.full(len(d), i + 1) + rng.uniform(-0.12, 0.12, len(d)), d, "o", ms=3, color=cols[ARMS[i]])
axes[1].set_ylabel("within-run SD of price"); axes[1].set_title("Price volatility within runs")
fig.tight_layout(); fig.savefig(os.path.join(RES, "v2_distributions.png"), dpi=200); plt.close(fig)

print("\nSaved v2_*.csv and v2_*.png in", RES)
