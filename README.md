# Prompt Sensitivity in LLM-Based Pricing Agents

Code and data for the MSc Economics dissertation *Prompt Sensitivity in LLM-Based Pricing Agents: Underspecification, Average Prices and Outcome Dispersion in a Repeated Bertrand Game* (Sharon Thomas, The University of Manchester, 2026).

Two pricing agents based on Gemini 3.5 Flash-Lite set prices in a repeated Bertrand duopoly with logit demand (Calvano et al., 2020), following the design of Fish, Gonczarowski and Shorrer (2024). Three prompt arms are compared: P1 (baseline), P1-trunc (the clause "thus, you should not take actions which undermine profitability" removed) and P1-para (a paraphrase). Each arm was run 29 times for 100 periods.

## Contents

| Folder | Contents |
|---|---|
| `code/` | The experiment: `config.py` (model, budget and design settings), `environment.py` (market model and Nash/monopoly benchmarks), `prompts.py` (prompt texts), `llm.py` (model calls, parsing, retries, spending cap), `run_experiment.py` (simulation loop), `analyse.py` (first-round analysis) |
| `analysis/` | `analyse_v2.py` (run-level statistics, tests, variance decomposition, mixed model, regressions, Figures 4.2, 4.4, 4.5), `analyse_profit_plans.py` (profits, stability, keyword analysis), `extra_figs.py` (Figures 4.1, 4.3, 4.6, 4.7, 4.8) |
| `results/` | Raw data in `raw_runs.zip`: one JSON-lines file per run (`duopoly_<arm>_run<k>.jsonl`, 87 files; `monopoly_P0_run<k>.jsonl`, 3 files). Each line is one period with both firms' prices, quantities, profits and the full text each agent wrote. Also the output tables (`v2_*.csv`) and figures (`v2_*.png`). |

## Reproducing the analysis

Requires Python 3.10+ and `pip install numpy scipy pandas matplotlib statsmodels`. First unzip `results/raw_runs.zip` into the `results/` folder. Then, from this folder:

```
python analysis/analyse_v2.py results
python analysis/analyse_profit_plans.py
python analysis/extra_figs.py
```

## Re-running the experiment

Needs a Gemini API key (`pip install google-genai`). Put the key in `code/api_key.txt` (see `api_key_example.txt`) or set `GEMINI_API_KEY`. Never commit the key. Then, from `code/`:

```
python run_experiment.py --mode monopoly --arm P0 --runs 3
python run_experiment.py --mode duopoly --arms P1,P1-trunc,P1-para --runs 29
python run_experiment.py --mock --mode duopoly --arm P1 --runs 1 --periods 20   # free test without API calls
```

`config.py` holds a hard spending cap (`BUDGET_USD`). The full experiment cost about US$42. Model outputs are stochastic (temperature 1), so re-runs will not reproduce the data exactly.
