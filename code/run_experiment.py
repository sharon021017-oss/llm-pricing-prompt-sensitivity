"""Run LLM pricing experiments.

Examples (run from this folder):
  python run_experiment.py --mock --mode duopoly --arm P1 --runs 1 --periods 20   # free test, no API calls
  python run_experiment.py --mode monopoly --arm P0 --runs 3                      # Stage 2 monopoly check
  python run_experiment.py --mode duopoly --arm P1 --runs 8                       # main runs

Each run is saved to results/<mode>_<arm>_run<k>.jsonl, one line per period.
If a run is interrupted (closed laptop, spending cap, daily limit), just run the same
command again: finished runs are skipped and an unfinished run continues where it stopped.
"""
import argparse, json, os, random
from concurrent.futures import ThreadPoolExecutor
import config, environment as env, prompts, llm


def run_path(mode, arm, k):
    return os.path.join(config.RESULTS_DIR, f"{mode}_{arm}_run{k}.jsonl")


def load_run(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run_one(backend, ledger, mode, arm, k, n_periods):
    path = run_path(mode, arm, k)
    done = load_run(path)
    if len(done) >= n_periods:
        print(f"  run {k}: already complete")
        return
    duo = mode == "duopoly"
    n = 2 if duo else 1
    alpha = config.ALPHAS[(k - 1) % len(config.ALPHAS)]
    rng = random.Random(f"{mode}-{arm}-{k}")
    p_ref = env.monopoly_price(alpha) if duo else env.monopoly_price_single(alpha)
    ceiling = rng.uniform(1.5, 2.5) * p_ref
    cost = alpha * env.C[0]

    # rebuild state from what is already saved
    hist = [[] for _ in range(n)]
    plans, insights = [""] * n, [""] * n
    for rec in done:
        for i, a in enumerate(rec["agents"]):
            hist[i].append({"round": rec["period"], "my_price": a["price"],
                            "comp_price": rec["agents"][1 - i]["price"] if duo else None,
                            "quantity": a["quantity"], "profit": a["profit"]})
            plans[i], insights[i] = a["plans"], a["insights"]
    if done:
        print(f"  run {k}: resuming at period {len(done) + 1}")

    with open(path, "a", encoding="utf-8") as f:
        for t in range(len(done) + 1, n_periods + 1):
            ps = [prompts.build_prompt(arm, cost, ceiling, plans[i], insights[i],
                                       hist[i][-config.HISTORY_WINDOW:], duo) for i in range(n)]
            with ThreadPoolExecutor(n) as ex:
                outs = list(ex.map(lambda p: llm.ask(backend, ledger, p), ps))
            prices = [o["price"] for o in outs]
            if duo:
                q = env.demand(prices, alpha)
                pi = env.profit(prices, alpha)
            else:
                q = [env.demand_mono(prices[0], alpha)]
                pi = [env.profit_mono(prices[0], alpha)]
            rec = {"mode": mode, "arm": arm, "run": k, "period": t, "alpha": alpha,
                   "ceiling": round(ceiling, 4), "model": config.MODEL, "agents": []}
            for i, o in enumerate(outs):
                rec["agents"].append({"price": prices[i], "quantity": float(q[i]), "profit": float(pi[i]),
                                      "plans": o["plans"], "insights": o["insights"],
                                      "observations": o["observations"], "attempts": o["attempts"],
                                      "input_tokens": o["input_tokens"], "output_tokens": o["output_tokens"]})
                hist[i].append({"round": t, "my_price": prices[i],
                                "comp_price": prices[1 - i] if duo else None,
                                "quantity": float(q[i]), "profit": float(pi[i])})
                plans[i], insights[i] = o["plans"], o["insights"]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if t % 5 == 0 or t == n_periods or t == len(done) + 1:
                norm = ", ".join(f"{p / alpha:.2f}" for p in prices)
                print(f"  run {k} period {t}/{n_periods}: prices/alpha = {norm}  "
                      f"| spent ${ledger.data['cost_usd']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["duopoly", "monopoly"], default="duopoly")
    ap.add_argument("--arm", default="P1", choices=list(prompts.PREFIXES))
    ap.add_argument("--arms", default=None,
                    help="comma-separated arms run in rotation, e.g. P1,P1-trunc,P1-para "
                         "(run 1 of each, then run 2 of each, ...) so the arms stay balanced")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--start", type=int, default=1, help="first run number")
    ap.add_argument("--periods", type=int, default=config.N_PERIODS)
    ap.add_argument("--mock", action="store_true", help="fake model: tests the harness for free")
    a = ap.parse_args()

    if a.mock:
        config.RESULTS_DIR = "results_mock"
        config.MAX_REQUESTS_PER_MINUTE = 10**9
    backend = llm.MockBackend() if a.mock else llm.GeminiBackend()
    ledger = llm.Ledger()
    arms = a.arms.split(",") if a.arms else [a.arm]
    for x in arms:
        if x not in prompts.PREFIXES:
            raise SystemExit(f"Unknown arm {x}. Choose from {list(prompts.PREFIXES)}")
    print(f"{a.mode} | arms {', '.join(arms)} | runs {a.start}-{a.start + a.runs - 1} | {a.periods} periods "
          f"| model {'MOCK' if a.mock else config.MODEL} | spent so far ${ledger.data['cost_usd']:.2f} "
          f"of ${config.BUDGET_USD:.2f}")
    try:
        for k in range(a.start, a.start + a.runs):
            for arm in arms:
                print(f"--- {arm}, run {k}")
                run_one(backend, ledger, a.mode, arm, k, a.periods)
    except (llm.BudgetExceeded, llm.DailyLimitReached) as e:
        print(f"\nSTOPPED: {e}\nProgress is saved. Run the same command again later to continue.")
        return
    print(f"\nDone. Total spent: ${ledger.data['cost_usd']:.2f} "
          f"({ledger.data['requests']} requests).")


if __name__ == "__main__":
    main()
