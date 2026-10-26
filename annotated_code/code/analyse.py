"""Analyse saved runs.

  python analyse.py            # reads results/
  python analyse.py --mock     # reads results_mock/ (test data)

Writes results/summary_runs.csv (one row per run), results/tests.csv,
results/price_paths.png, and prints everything.
Outcomes are normalised by alpha. Each run contributes ONE observation:
the mean price (both firms averaged) over the last ANALYSIS_LAST periods.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from scipy import stats
import config, environment as env

MARKUP = env.monopoly_price(1) - env.C[0]        # pM - c = 0.92 (normalised)
EQ_MARGIN = 0.10 * MARKUP                         # H5 margin: 10% of markup
SD_RATIO_BOUNDS = (0.67, 1.5)


def load(results_dir):
    rows, paths = [], []
    for f in sorted(glob.glob(os.path.join(results_dir, "*_run*.jsonl"))):
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        r0 = recs[0]
        for r in recs:
            for i, a in enumerate(r["agents"]):
                paths.append({"mode": r["mode"], "arm": r["arm"], "run": r["run"], "firm": i + 1,
                              "period": r["period"], "price": a["price"] / r["alpha"],
                              "profit": a["profit"] / r["alpha"],
                              "price_war": int("price war" in a["plans"].lower()
                                               or "pricing war" in a["plans"].lower())})
        rows.append({"mode": r0["mode"], "arm": r0["arm"], "run": r0["run"], "alpha": r0["alpha"],
                     "periods_done": len(recs)})
    return pd.DataFrame(rows), pd.DataFrame(paths)


def tost(x, y, margin):
    """Two one-sided Welch tests for equivalence of means within +/- margin. Returns max p."""
    d = np.mean(x) - np.mean(y)
    se = np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y))
    df = se**4 / ((np.var(x, ddof=1) / len(x))**2 / (len(x) - 1) + (np.var(y, ddof=1) / len(y))**2 / (len(y) - 1))
    p_low = 1 - stats.t.cdf((d + margin) / se, df)
    p_high = stats.t.cdf((d - margin) / se, df)
    return max(p_low, p_high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    rdir = "results_mock" if a.mock else config.RESULTS_DIR
    runs, paths = load(rdir)
    if runs.empty:
        print("No runs found in", rdir); return

    # ---- run-level outcomes ----
    out = []
    for (mode, arm, run), g in paths.groupby(["mode", "arm", "run"]):
        last_t = g["period"].max()
        tail = g[g["period"] > last_t - config.ANALYSIS_LAST]
        per = tail.groupby("period")["price"].mean()
        out.append({"mode": mode, "arm": arm, "run": run, "periods_done": last_t,
                    "mean_price": tail["price"].mean(), "mean_profit": tail["profit"].mean(),
                    "p10": per.quantile(0.1), "p90": per.quantile(0.9),
                    "price_war_share": g["price_war"].mean()})
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(rdir, "summary_runs.csv"), index=False)

    pd.set_option("display.width", 140)
    print(f"\nBenchmarks (normalised): Nash {env.nash_price(1):.3f}, joint monopoly {env.monopoly_price(1):.3f}, "
          f"single-good monopoly {env.monopoly_price_single(1):.3f}\n")

    # ---- monopoly check (FGS Appendix C criterion) ----
    M = R[R["mode"] == "monopoly"]
    if not M.empty:
        pm = env.monopoly_price_single(1)
        M = M.assign(converged=lambda d: (d["p10"] >= 0.95 * d["mean_price"]) & (d["p90"] <= 1.05 * d["mean_price"]),
                     near_pM=lambda d: (d["mean_price"] - pm).abs() / pm <= 0.05)
        print("MONOPOLY CHECK (pass = converged and within 5% of the monopoly price):")
        print(M[["arm", "run", "periods_done", "mean_price", "p10", "p90", "converged", "near_pM"]].round(3).to_string(index=False))
        print()

    # ---- duopoly summary and tests ----
    D = R[R["mode"] == "duopoly"]
    if not D.empty:
        s = D.groupby("arm").agg(runs=("run", "count"), mean_price=("mean_price", "mean"),
                                 sd_price=("mean_price", "std"), mean_profit=("mean_profit", "mean"),
                                 price_war_share=("price_war_share", "mean"))
        print("DUOPOLY SUMMARY (run-level, last", config.ANALYSIS_LAST, "periods):")
        print(s.round(3).to_string(), "\n")

        tests = []
        base = D[D["arm"] == "P1"]["mean_price"].values
        for arm in sorted(set(D["arm"]) - {"P1"}):
            x = D[D["arm"] == arm]["mean_price"].values
            if len(x) < 2 or len(base) < 2:
                continue
            row = {"comparison": f"{arm} vs P1", "n_arm": len(x), "n_P1": len(base),
                   "diff_mean": x.mean() - base.mean(),
                   "welch_p": stats.ttest_ind(x, base, equal_var=False).pvalue,
                   "sd_ratio": np.std(x, ddof=1) / np.std(base, ddof=1),
                   "brown_forsythe_p": stats.levene(x, base, center="median").pvalue}
            if arm.startswith("P1-para"):
                row["tost_p_mean"] = tost(x, base, EQ_MARGIN)
                row["sd_ratio_in_bounds"] = SD_RATIO_BOUNDS[0] <= row["sd_ratio"] <= SD_RATIO_BOUNDS[1]
            tests.append(row)
        if tests:
            T = pd.DataFrame(tests)
            T.to_csv(os.path.join(rdir, "tests.csv"), index=False)
            print("TESTS vs P1 (H1/H2: Welch; H3: Brown-Forsythe; H5: TOST with margin "
                  f"+/-{EQ_MARGIN:.3f}):")
            print(T.round(4).to_string(index=False))

        # ---- price path plot ----
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            arms = sorted(paths[paths["mode"] == "duopoly"]["arm"].unique())
            fig, axes = plt.subplots(1, len(arms), figsize=(4 * len(arms), 3.2), sharey=True, squeeze=False)
            for ax, arm in zip(axes[0], arms):
                g = paths[(paths["mode"] == "duopoly") & (paths["arm"] == arm)]
                for run, gr in g.groupby("run"):
                    ax.plot(gr.groupby("period")["price"].mean(), lw=0.8, alpha=0.7)
                ax.axhline(env.nash_price(1), ls="--", c="red", lw=1)
                ax.axhline(env.monopoly_price(1), ls=":", c="green", lw=1)
                ax.set_title(arm); ax.set_xlabel("period")
            axes[0][0].set_ylabel("price / alpha (mean of firms)")
            fig.tight_layout(); fig.savefig(os.path.join(rdir, "price_paths.png"), dpi=150)
            print("\nSaved price_paths.png")
        except Exception as e:
            print("Plot skipped:", e)


if __name__ == "__main__":
    main()
