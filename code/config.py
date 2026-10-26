"""Settings you may want to change. Everything else reads from here."""

# ---- Model ----
MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 1.0            # FGS Appendix B
THINKING_LEVEL = "minimal"   # hidden thinking tokens are billed; "minimal" keeps cost down

# ---- Prices (USD per 1M tokens, Gemini 3.5 Flash-Lite standard tier, checked Oct 2026) ----
PRICE_INPUT_PER_M = 0.30
PRICE_OUTPUT_PER_M = 2.50    # includes thinking tokens

# ---- Hard spending cap (USD). The harness stops before going past this. ----
BUDGET_USD = 45.00   # raised for 20 extra runs per arm; counts free-tier calls too, so it is conservative

# ---- Rate limits (stay a little under Google's limits) ----
MAX_REQUESTS_PER_MINUTE = 60   # paid tier (prepaid credit)
MAX_REQUESTS_PER_DAY = None    # e.g. 480 on the free tier; None = no daily cap

# ---- Experiment design (reduced version) ----
N_PERIODS = 100          # FGS use 300
HISTORY_WINDOW = 30      # FGS show 100 past periods
ANALYSIS_LAST = 50       # analyse the last 50 periods
ALPHAS = [1, 3.2, 10]    # currency scale, cycled across runs
MAX_RETRIES = 10         # FGS Appendix B

# ---- Files ----
API_KEY_FILE = "api_key.txt"   # put your key in this file (one line), or set GEMINI_API_KEY
RESULTS_DIR = "results"
